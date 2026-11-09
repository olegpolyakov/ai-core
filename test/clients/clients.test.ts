// eslint-disable-next-line simple-import-sort/imports
import { beforeEach, describe, expect, it, vi } from 'vitest';

import createAnthropicMock from './anthropic.mock';
import createOllamaMock from './ollama.mock';
import createOpenAiMock from './openai.mock';

import {
    AnthropicClient,
    OllamaClient,
    OpenAIClient
} from '@/clients';
import { AIClient, Message, Tool } from '@/lib';

const anthropicMock = createAnthropicMock();
const ollamaMock = createOllamaMock();
const openaiMock = createOpenAiMock();

const adapters = [
    {
        name: 'Anthropic',
        models: anthropicMock.models,
        request: anthropicMock.create,
        createClient: (tools?: Tool[]): AIClient => new AnthropicClient({
            apiKey: 'test-key',
            model: 'configured-model',
            tools
        }),
        setFinalResponse: () => anthropicMock.create.mockResolvedValue({
            content: [{ type: 'text', text: 'Contract response' }]
        }),
        setToolResponses: () => anthropicMock.create
            .mockResolvedValueOnce({
                content: [{
                    type: 'tool_use',
                    id: 'tool-call',
                    name: 'lookup',
                    input: { query: 'example' }
                }]
            })
            .mockResolvedValueOnce({
                content: [{ type: 'text', text: 'Tool response' }]
            }),
        setFailure: () => anthropicMock.create.mockRejectedValue(new Error('SDK failure'))
    },
    {
        name: 'Ollama',
        models: ollamaMock.models,
        request: ollamaMock.chat,
        createClient: (tools?: Tool[]): AIClient => new OllamaClient({
            apiKey: 'test-key',
            model: 'configured-model',
            tools
        }),
        setFinalResponse: () => ollamaMock.chat.mockResolvedValue({
            message: { role: 'assistant', content: 'Contract response' }
        }),
        setToolResponses: () => ollamaMock.chat
            .mockResolvedValueOnce({
                message: {
                    role: 'assistant',
                    content: '',
                    tool_calls: [{
                        function: {
                            name: 'lookup',
                            arguments: { query: 'example' }
                        }
                    }]
                }
            })
            .mockResolvedValueOnce({
                message: { role: 'assistant', content: 'Tool response' }
            }),
        setFailure: () => ollamaMock.chat.mockRejectedValue(new Error('SDK failure'))
    },
    {
        name: 'OpenAI',
        models: openaiMock.models,
        request: openaiMock.create,
        createClient: (tools?: Tool[]): AIClient => new OpenAIClient({
            apiKey: 'test-key',
            model: 'configured-model',
            tools
        }),
        setFinalResponse: () => openaiMock.create.mockResolvedValue({
            output: [{ type: 'message' }],
            output_text: 'Contract response'
        }),
        setToolResponses: () => openaiMock.create
            .mockResolvedValueOnce({
                output: [{
                    type: 'function_call',
                    call_id: 'tool-call',
                    name: 'lookup',
                    arguments: '{"query":"example"}'
                }],
                output_text: ''
            })
            .mockResolvedValueOnce({
                output: [{ type: 'message' }],
                output_text: 'Tool response'
            }),
        setFailure: () => openaiMock.create.mockRejectedValue(new Error('SDK failure'))
    }
];

const messages: Message[] = [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: 'Answer this request.' }
];

describe.each(adapters)('$name client', adapter => {
    beforeEach(() => {
        vi.resetAllMocks();
    });

    it('provides getModels and resolves successfully', async () => {
        const result = { models: [{ id: 'model-a' }] };
        adapter.models.mockResolvedValue(adapter.name === 'OpenAI'
            ? { data: result }
            : result);
        const client = adapter.createClient();

        const models = await client.getModels();

        expect(models).toBeDefined();
        expect(adapter.models).toHaveBeenCalledOnce();
    });

    it('accepts shared messages and model options and returns the common response shape', async () => {
        adapter.setFinalResponse();
        const client = adapter.createClient();

        const response = await client.respond(messages, { model: 'contract-model' });

        expect(response).toEqual({
            message: { role: 'assistant', content: 'Contract response' }
        });
        expect(adapter.request).toHaveBeenCalledOnce();
        expect(adapter.request.mock.calls[0][0]).toMatchObject({
            model: 'contract-model'
        });
    });

    it('executes configured tools and returns the same response shape after the tool call', async () => {
        adapter.setToolResponses();
        const toolCall = vi.fn(async () => ({
            content: [{ type: 'text' as const, text: 'Tool result' }]
        }));
        const tool = new Tool(
            'lookup',
            'Look up a value',
            { type: 'object', properties: {}, required: [] },
            toolCall
        );
        const client = adapter.createClient([tool]);

        const response = await client.respond(messages, { model: 'contract-model' });

        expect(response).toEqual({
            message: { role: 'assistant', content: 'Tool response' }
        });
        expect(toolCall).toHaveBeenCalledWith({ query: 'example' });
        expect(adapter.request).toHaveBeenCalledTimes(2);
    });

    it('propagates SDK request errors', async () => {
        adapter.setFailure();
        const client = adapter.createClient();

        await expect(client.respond(messages)).rejects.toThrow('SDK failure');
    });
});