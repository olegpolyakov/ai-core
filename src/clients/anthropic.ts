import Anthropic from '@anthropic-ai/sdk';
import type { DocumentBlockParam, ImageBlockParam, TextBlockParam } from '@anthropic-ai/sdk/resources';

import {
    AIClient,
    ClientOptions,
    getErrorMessage,
    Message,
    Response,
    Tool
} from '../lib';

export default class AnthropicClient extends AIClient {
    private _client: Anthropic;

    constructor({
        apiKey,
        baseURL
    }: ClientOptions) {
        super();
        this._client = new Anthropic({ apiKey, baseURL });
    }

    async getModels() {
        return this._client.models.list();
    }

    async loop(
        model: string,
        messages: Message[],
        tools: Tool[] = []
    ): Promise<Response> {
        const system: string = messages
            .filter(message => message.role === 'system')
            .map(message => message.content)
            .join('\n');

        const history: Anthropic.MessageParam[] = messages
            .filter(message => message.role !== 'system')
            .map(message => ({
                role: message.role,
                content: !Array.isArray(message.content)
                    ? message.content
                    : message.content.map(item => {
                        switch (item.type) {
                            case 'image': return {
                                type: 'image',
                                source: {
                                    type: 'url',
                                    url: item.url
                                }
                            } as ImageBlockParam;

                            case 'file': return {
                                type: 'document',
                                source: {
                                    type: 'url',
                                    url: item.url
                                }
                            } as DocumentBlockParam;
                        
                            default: return {
                                type: 'text',
                                text: item.content
                            } as TextBlockParam;
                        }
                    })
            }));

        const anthropicTools: Anthropic.Tool[] = tools.map(t => ({
            name: t.name,
            description: t.description,
            input_schema: {
                ...t.inputSchema,
                type: 'object'
            }
        }));

        return this._loop(
            model,
            system,
            history,
            anthropicTools
        );
    }

    private async _loop(
        model: string,
        system: string,
        messages: Anthropic.MessageParam[],
        tools: Anthropic.Tool[]
    ): Promise<Response> {
        const response = await this._client.messages.create({
            model,
            system,
            messages,
            tools,
            max_tokens: 1024,
        });

        const toolCalls = response.content.filter(
            (block): block is Anthropic.ToolUseBlock => block.type === 'tool_use'
        );

        if (toolCalls.length === 0) return {
            message: {
                role: 'assistant',
                content: response.content
                    .filter((block): block is Anthropic.TextBlock => block.type === 'text')
                    .map(block => block.text)
                    .join('\n')
            }
        };

        messages.push({
            role: 'assistant',
            content: response.content
        });

        const results: Anthropic.ToolResultBlockParam[] = [];

        for (const call of toolCalls) {
            try {
                const result = await this.call(call.name, call.input);

                results.push({
                    type: 'tool_result',
                    tool_use_id: call.id,
                    content: String(result)
                });
            } catch (error) {
                console.error(error);
                results.push({
                    type: 'tool_result',
                    tool_use_id: call.id,
                    is_error: true,
                    content: getErrorMessage(error)
                });
            }
        }

        messages.push({
            role: 'user',
            content: results
        });

        return this._loop(model, system, messages, tools);
    }
}