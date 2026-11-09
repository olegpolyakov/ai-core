import Anthropic from '@anthropic-ai/sdk';
import type { DocumentBlockParam, ImageBlockParam, TextBlockParam } from '@anthropic-ai/sdk/resources';

import {
    AIClient,
    ClientOptions,
    getErrorMessage,
    Message,
    Response
} from '../lib';

export default class AnthropicClient extends AIClient {
    private _client: Anthropic;

    constructor({
        apiKey,
        baseURL,
        tools,
        model
    }: ClientOptions) {
        super(model, tools);
        this._client = new Anthropic({ apiKey, baseURL });
    }

    async getModels() {
        return this._client.models.list();
    }

    async respond(messages: Message[], { model = 'claude-3-5-haiku-20241022' } = {}): Promise<Response> {
        return this.loop(messages, model);
    }

    async loop(messages: Message[], model: string): Promise<Response> {
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

        const tools: Anthropic.Tool[] = Object.values(this._tools).map(({ definition }) => ({
            name: definition.name,
            description: definition.description,
            input_schema: {
                ...definition.parameters,
                type: 'object'
            }
        }));

        return this._loop(
            model,
            system,
            history,
            tools
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
            max_tokens: 1024,
            system,
            messages,
            tools: tools.length > 0 ? tools : undefined
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