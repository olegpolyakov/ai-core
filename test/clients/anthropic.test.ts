// eslint-disable-next-line simple-import-sort/imports
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import createMock from './anthropic.mock';

import AnthropicClient from '@/clients/anthropic';
import { Message, Tool } from '@/lib';

const anthropicMock = createMock();

describe('AnthropicClient', () => {
    let client: AnthropicClient;

    beforeEach(() => {
        vi.resetAllMocks();
        client = new AnthropicClient({
            apiKey: 'test-api-key',
            baseURL: 'https://api.example.test'
        });
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('configures the Anthropic SDK with the provided credentials and base URL', () => {
        expect(anthropicMock.constructor).toHaveBeenCalledWith({
            apiKey: 'test-api-key',
            baseURL: 'https://api.example.test'
        });
    });

    it('returns models from the SDK', async () => {
        const modelsData = { data: [{ id: 'model-a' }] };
        anthropicMock.models.mockResolvedValue(modelsData);

        const models = await client.getModels();

        expect(models).toBe(modelsData);
        expect(anthropicMock.models).toHaveBeenCalledOnce();
    });

    it('joins system messages, converts content blocks, and uses the default model', async () => {
        anthropicMock.create.mockResolvedValue({
            content: [
                { type: 'text', text: 'First part' },
                { type: 'text', text: 'Second part' }
            ]
        });
        const messages: Message[] = [
            { role: 'system', content: 'System one' },
            { role: 'user', content: [
                { type: 'text', content: 'Review these' },
                { type: 'image', url: 'https://example.test/image.png' },
                { type: 'file', name: 'report.pdf', url: 'https://example.test/report.pdf' }
            ] },
            { role: 'system', content: 'System two' },
            { role: 'assistant', content: 'Previous response' }
        ];

        const response = await client.respond(messages, { model: 'test-model' });

        expect(response).toEqual({
            message: { role: 'assistant', content: 'First part\nSecond part' }
        });
        expect(anthropicMock.create).toHaveBeenCalledWith({
            model: 'test-model',
            max_tokens: 1024,
            system: 'System one\nSystem two',
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'Review these' },
                        {
                            type: 'image',
                            source: { type: 'url', url: 'https://example.test/image.png' }
                        },
                        {
                            type: 'document',
                            source: { type: 'url', url: 'https://example.test/report.pdf' }
                        }
                    ]
                },
                { role: 'assistant', content: 'Previous response' }
            ],
            tools: []
        });
    });

    it('uses the requested model and converts tool definitions', async () => {
        anthropicMock.create.mockResolvedValue({
            content: [{ type: 'text', text: 'Done' }]
        });
        const tool = new Tool(
            'lookup',
            'Find a record',
            {
                type: 'object',
                properties: { query: { type: 'string' } },
                required: ['query']
            },
            async () => ({ content: [] })
        );
        client = new AnthropicClient({
            apiKey: 'test-api-key',
            baseURL: 'https://api.example.test'
        });

        await client.respond([
            { role: 'user', content: 'Look this up' }
        ], { model: 'custom-model', tools: [tool] });

        expect(anthropicMock.create).toHaveBeenLastCalledWith({
            model: 'custom-model',
            max_tokens: 1024,
            system: '',
            messages: [{ role: 'user', content: 'Look this up' }],
            tools: [{
                name: 'lookup',
                description: 'Find a record',
                input_schema: {
                    type: 'object',
                    properties: { query: { type: 'string' } },
                    required: ['query']
                }
            }]
        });
    });

    it('executes tool calls and sends their results in a follow-up request', async () => {
        const call = vi.fn(async () => ({
            content: [{ type: 'text' as const, text: 'Found it' }]
        }));
        const tool = new Tool('lookup', '', { type: 'object' }, call);
        client = new AnthropicClient({
            apiKey: 'test-api-key'
        });
        anthropicMock.create
            .mockResolvedValueOnce({
                content: [
                    { type: 'text', text: 'Searching' },
                    { type: 'tool_use', id: 'tool-call-1', name: 'lookup', input: { query: 'hello' } }
                ]
            })
            .mockResolvedValueOnce({
                content: [{ type: 'text', text: 'The result is here' }]
            });

        const response = await client.respond([
            { role: 'user', content: 'Find hello' }
        ], { model: 'custom-model', tools: [tool] });

        expect(response).toEqual({
            message: { role: 'assistant', content: 'The result is here' }
        });

        expect(call).toHaveBeenCalledWith({ query: 'hello' });
        expect(anthropicMock.create).toHaveBeenCalledTimes(2);
        expect(anthropicMock.create.mock.calls[1][0]).toMatchObject({
            messages: [
                { role: 'user', content: 'Find hello' },
                {
                    role: 'assistant',
                    content: [
                        { type: 'text', text: 'Searching' },
                        { type: 'tool_use', id: 'tool-call-1', name: 'lookup', input: { query: 'hello' } }
                    ]
                },
                {
                    role: 'user',
                    content: [{
                        type: 'tool_result',
                        tool_use_id: 'tool-call-1',
                        content: '[object Object]'
                    }]
                }
            ]
        });
    });

    it('reports tool errors in the follow-up tool result', async () => {
        const error = new Error('Lookup failed');
        const tool = new Tool('lookup', '', { type: 'object' }, async () => { throw error; });
        client = new AnthropicClient({
            apiKey: 'test-api-key'
        });
        anthropicMock.create
            .mockResolvedValueOnce({
                content: [{ type: 'tool_use', id: 'tool-call-1', name: 'lookup', input: {} }]
            })
            .mockResolvedValueOnce({
                content: [{ type: 'text', text: 'Could not find it' }]
            });

        await client.respond([{ role: 'user', content: 'Look it up' }], { tools: [tool] });

        expect(anthropicMock.create.mock.calls[1][0]).toMatchObject({
            messages: expect.arrayContaining([
                {
                    role: 'user',
                    content: [{
                        type: 'tool_result',
                        tool_use_id: 'tool-call-1',
                        is_error: true,
                        content: 'Error Lookup failed'
                    }]
                }
            ])
        });
    });

    it('propagates errors from the Anthropic SDK', async () => {
        anthropicMock.create.mockRejectedValue(new Error('Request failed'));

        await expect(client.respond([{ role: 'user', content: 'Hello' }]))
            .rejects.toThrow('Request failed');
    });
});