// eslint-disable-next-line simple-import-sort/imports
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import createMock from './ollama.mock';

import OllamaClient from '@/clients/ollama';
import { Message, Tool } from '@/lib';

const ollamaMock = createMock();

describe('OllamaClient', () => {
    let client: OllamaClient;

    beforeEach(() => {
        vi.resetAllMocks();
        client = new OllamaClient({
            apiKey: 'test-api-key',
            baseURL: 'https://api.example.test'
        });
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('configures the Ollama SDK with the provided host and credentials', () => {
        expect(ollamaMock.constructor).toHaveBeenCalledWith({
            host: 'https://api.example.test',
            headers: { Authorization: 'Bearer test-api-key' }
        });
    });

    it('uses the default host when no base URL is provided', () => {
        new OllamaClient({ apiKey: 'test-api-key' });

        expect(ollamaMock.constructor).toHaveBeenLastCalledWith({
            host: 'https://ollama.com',
            headers: { Authorization: 'Bearer test-api-key' }
        });
    });

    it('returns models from the SDK', async () => {
        const modelsData = { models: [{ name: 'model-a' }] };
        ollamaMock.models.mockResolvedValue(modelsData);

        const models = await client.getModels();

        expect(models).toBe(modelsData);
        expect(ollamaMock.models).toHaveBeenCalledOnce();
    });

    it('converts messages and uses the configured model', async () => {
        ollamaMock.chat.mockResolvedValue({
            message: { role: 'assistant', content: 'Done' }
        });
        const messages: Message[] = [
            { role: 'system', content: 'System instruction' },
            { role: 'user', content: [
                { type: 'text', content: 'Describe this' },
                { type: 'image', url: 'https://example.test/image.png' },
                { type: 'image', url: 'https://example.test/another.png' },
                { type: 'file', name: 'report.pdf', url: 'https://example.test/report.pdf' }
            ] },
            { role: 'assistant', content: 'Previous response' }
        ];

        const response = await client.respond(messages, { model: 'test-model' });

        expect(response).toEqual({
            message: { role: 'assistant', content: 'Done' }
        });
        expect(ollamaMock.chat).toHaveBeenCalledWith({
            model: 'test-model',
            messages: [
                { role: 'system', content: 'System instruction', images: undefined },
                {
                    role: 'user',
                    content: 'Describe this',
                    images: [
                        'https://example.test/image.png',
                        'https://example.test/another.png'
                    ]
                },
                { role: 'assistant', content: 'Previous response', images: undefined }
            ],
            tools: []
        });
    });

    it('uses a requested model and converts tool definitions', async () => {
        ollamaMock.chat.mockResolvedValue({
            message: { role: 'assistant', content: 'Done' }
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
        client = new OllamaClient({
            apiKey: 'test-api-key',
            baseURL: 'https://api.example.test'
        });

        await client.respond([
            { role: 'user', content: 'Look this up' }
        ], {
            model: 'test-model', tools: [tool]
        });

        expect(ollamaMock.chat).toHaveBeenCalledWith({
            model: 'test-model',
            messages: [{ role: 'user', content: 'Look this up', images: undefined }],
            tools: [{
                type: 'function',
                function: {
                    name: 'lookup',
                    description: 'Find a record',
                    parameters: {
                        type: 'object',
                        properties: { query: { type: 'string' } },
                        required: ['query']
                    }
                }
            }]
        });
    });

    it('executes tool calls and sends their results in a follow-up request', async () => {
        const call = vi.fn(async () => ({
            content: [{ type: 'text' as const, text: 'Found it' }]
        }));
        const tool = new Tool('lookup', '', { type: 'object' }, call);
        client = new OllamaClient({
            apiKey: 'test-api-key'
        });
        ollamaMock.chat
            .mockResolvedValueOnce({
                message: {
                    role: 'assistant',
                    content: '',
                    tool_calls: [{
                        function: { name: 'lookup', arguments: { query: 'hello' } }
                    }]
                }
            })
            .mockResolvedValueOnce({
                message: { role: 'assistant', content: 'The result is here' }
            });

        const response = await client.respond([
            { role: 'user', content: 'Find hello' }
        ], { tools: [tool] });

        expect(response).toEqual({
            message: { role: 'assistant', content: 'The result is here' }
        });

        expect(call).toHaveBeenCalledWith({ query: 'hello' });
        expect(ollamaMock.chat).toHaveBeenCalledTimes(2);
        expect(ollamaMock.chat.mock.calls[1][0]).toMatchObject({
            messages: [
                { role: 'user', content: 'Find hello', images: undefined },
                {
                    role: 'assistant',
                    content: '',
                    tool_calls: [{
                        function: { name: 'lookup', arguments: { query: 'hello' } }
                    }]
                },
                { role: 'tool', tool_name: 'lookup', content: '[object Object]' }
            ]
        });
    });

    it('reports tool errors in the follow-up tool message', async () => {
        const error = new Error('Lookup failed');
        const tool = new Tool('lookup', '', { type: 'object' }, async () => { throw error; });
        client = new OllamaClient({
            apiKey: 'test-api-key'
        });
        ollamaMock.chat
            .mockResolvedValueOnce({
                message: {
                    role: 'assistant',
                    content: '',
                    tool_calls: [{
                        function: { name: 'lookup', arguments: {} }
                    }]
                }
            })
            .mockResolvedValueOnce({
                message: { role: 'assistant', content: 'Could not find it' }
            });

        await client.respond([
            { role: 'user', content: 'Look it up' }
        ], { tools: [tool] });

        expect(ollamaMock.chat.mock.calls[1][0].messages).toContainEqual({
            role: 'tool',
            tool_name: 'lookup',
            content: 'Error Lookup failed'
        });
    });

    it('propagates errors from the Ollama SDK', async () => {
        ollamaMock.chat.mockRejectedValue(new Error('Request failed'));

        await expect(client.respond([{ role: 'user', content: 'Hello' }]))
            .rejects.toThrow('Request failed');
    });
});