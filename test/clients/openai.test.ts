// eslint-disable-next-line simple-import-sort/imports
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import createMock from './openai.mock';

import OpenAIClient, { stripAddedNulls, toStrictSchema } from '@/clients/openai';
import { Message, Tool } from '@/lib';

const openaiMock = createMock();

describe('OpenAIClient', () => {
    let client: OpenAIClient;

    beforeEach(() => {
        vi.resetAllMocks();
        client = new OpenAIClient({
            apiKey: 'test-api-key',
            baseURL: 'https://api.example.test',
            model: 'default-model'
        });
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('configures the OpenAI SDK with the provided credentials and base URL', () => {
        expect(openaiMock.constructor).toHaveBeenCalledWith({
            apiKey: 'test-api-key',
            baseURL: 'https://api.example.test'
        });
    });

    it('returns the models from the SDK', async () => {
        const modelsData = [{ id: 'model-a' }, { id: 'model-b' }];
        openaiMock.models.mockResolvedValue({ data: modelsData });

        const models = await client.getModels();

        expect(models).toBe(modelsData);
        expect(openaiMock.models).toHaveBeenCalledOnce();
    });

    it('uses the configured model and converts text, image, and file content', async () => {
        openaiMock.create.mockResolvedValue({
            output: [{ type: 'message' }],
            output_text: 'Done'
        });

        const messages: Message[] = [
            {
                role: 'user',
                content: [
                    { type: 'text', content: 'Describe these files' },
                    { type: 'image', url: 'https://example.test/image.png' },
                    {
                        type: 'file',
                        name: 'report.pdf',
                        content: 'data:application/pdf;base64,abc'
                    }
                ]
            }
        ];

        const response = await client.respond(messages);

        expect(response).toEqual({
            message: { role: 'assistant', content: 'Done' }
        });
        expect(openaiMock.create).toHaveBeenCalledWith({
            model: 'default-model',
            input: [{
                role: 'user',
                content: [
                    { type: 'input_text', text: 'Describe these files' },
                    { type: 'input_image', image_url: 'https://example.test/image.png' },
                    {
                        type: 'input_file',
                        filename: 'report.pdf',
                        file_url: undefined,
                        file_data: 'data:application/pdf;base64,abc'
                    }
                ]
            }],
            tools: []
        });
    });

    it('uses the model and tools provided in respond options', async () => {
        openaiMock.create.mockResolvedValue({
            output: [{ type: 'message' }],
            output_text: 'Custom response'
        });
        const tool = new Tool('lookup', '', { type: 'object' }, async () => ({
            content: []
        }));

        await client.respond([{ role: 'user', content: 'Look this up' }], {
            model: 'custom-model',
            tools: [tool]
        });

        expect(openaiMock.create).toHaveBeenCalledWith(expect.objectContaining({
            model: 'custom-model',
            tools: [{
                type: 'function',
                name: 'lookup',
                description: '',
                parameters: {
                    type: 'object',
                    properties: {},
                    required: [],
                    additionalProperties: false
                },
                strict: true
            }]
        }));
    });

    it('executes function calls and sends their results in a follow-up request', async () => {
        const call = vi.fn(async () => ({
            content: [{ type: 'text' as const, text: 'Found it' }]
        }));
        const tool = new Tool(
            'lookup',
            'Find a record',
            {
                type: 'object',
                properties: { query: { type: 'string' } },
                required: ['query']
            },
            call
        );
        openaiMock.create
            .mockResolvedValueOnce({
                output: [{
                    type: 'function_call',
                    call_id: 'call-1',
                    name: 'lookup',
                    arguments: '{"query":"hello"}'
                }],
                output_text: ''
            })
            .mockResolvedValueOnce({
                output: [{ type: 'message' }],
                output_text: 'The result is here'
            });

        const response = await client.respond([
            { role: 'user', content: 'Find hello' }
        ], { tools: [tool] });

        expect(response).toEqual({
            message: { role: 'assistant', content: 'The result is here' }
        });
        expect(call).toHaveBeenCalledWith({ query: 'hello' });
        expect(openaiMock.create).toHaveBeenCalledTimes(2);
        expect(openaiMock.create.mock.calls[1][0]).toMatchObject({
            input: expect.arrayContaining([
                {
                    type: 'function_call_output',
                    call_id: 'call-1',
                    output: JSON.stringify({ content: [{ type: 'text', text: 'Found it' }] })
                }
            ])
        });
    });

    it('reports tool errors in the follow-up function output', async () => {
        const error = new Error('Lookup failed');
        const tool = new Tool(
            'lookup',
            '',
            { type: 'object', properties: {}, required: [] },
            async () => { throw error; }
        );
        openaiMock.create
            .mockResolvedValueOnce({
                output: [{
                    type: 'function_call',
                    call_id: 'call-1',
                    name: 'lookup',
                    arguments: '{}'
                }],
                output_text: ''
            })
            .mockResolvedValueOnce({
                output: [{ type: 'message' }],
                output_text: 'Could not find it'
            });

        await client.respond([{ role: 'user', content: 'Look it up' }], { tools: [tool] });

        expect(openaiMock.create.mock.calls[1][0]).toMatchObject({
            input: expect.arrayContaining([
                { type: 'function_call_output', call_id: 'call-1', output: 'Error Lookup failed' }
            ])
        });
    });

    it('throws when the model returns no output', async () => {
        openaiMock.create.mockResolvedValue({ output: [], output_text: '' });

        await expect(client.respond([{ role: 'user', content: 'Hello' }]))
            .rejects.toThrow('The model returned no output');
    });

    it('propagates errors from the OpenAI SDK', async () => {
        openaiMock.create.mockRejectedValue(new Error('Request failed'));

        await expect(client.respond([{ role: 'user', content: 'Hello' }]))
            .rejects.toThrow('Request failed');
    });
});

describe('OpenAI strict schema helpers', () => {
    it('makes every object property required and nullable only when originally optional', () => {
        expect(toStrictSchema({
            type: 'object',
            properties: {
                required: { type: 'string' },
                optional: { type: 'string', enum: ['a', 'b'] },
                nested: {
                    type: 'object',
                    properties: { value: { type: 'number' } },
                    required: ['value']
                }
            },
            required: ['required']
        })).toEqual({
            type: 'object',
            properties: {
                required: { type: 'string' },
                optional: {
                    type: ['string', 'null'],
                    enum: ['a', 'b', null]
                },
                nested: {
                    type: ['object', 'null'],
                    properties: {
                        value: { type: 'number' }
                    },
                    required: ['value'],
                    additionalProperties: false
                }
            },
            required: ['required', 'optional', 'nested'],
            additionalProperties: false
        });
    });

    it('removes generated null values but preserves required nullable values', () => {
        const schema = {
            type: 'object',
            properties: {
                optional: { type: 'string' },
                required: { type: ['string', 'null'] }
            },
            required: ['required']
        };

        expect(stripAddedNulls({
            optional: null,
            required: null,
            extra: null
        }, schema)).toEqual({ required: null });
    });
});