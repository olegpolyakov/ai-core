import {
    Message as OllamaMessage,
    Ollama,
    Tool as OllamaTool
} from 'ollama';

import {
    AIClient,
    ClientOptions,
    getErrorMessage,
    Message,
    Response
} from '../lib';

export default class OllamaClient extends AIClient {
    private _client: Ollama;

    constructor({
        apiKey,
        baseURL = 'https://ollama.com',
        tools,
        model = 'gemma4:31b-cloud'
    }: ClientOptions) {
        super(model, tools);
        this._client = new Ollama({
            host: baseURL,
            headers: {
                Authorization: `Bearer ${apiKey}`
            }
        });
    }

    async getModels() {
        return this._client.list();
    }

    async respond(messages: Message[], { model = this._model } = {}): Promise<Response> {
        return this.loop(messages, model);
    }

    async loop(messages: Message[], model: string): Promise<Response> {
        return this._loop(
            model,
            messages.map(message => ({
                role: message.role,
                content: Array.isArray(message.content)
                    ? message.content.find(item => item.type === 'text')?.content ?? ''
                    : message.content,
                images: Array.isArray(message.content)
                    ? message.content
                        .filter(item => item.type === 'image')
                        .map(item => item.url)
                    : undefined
            })) as OllamaMessage[],
            Object.values(this._tools).map(({ definition }) => ({
                type: 'function' as const,
                function: {
                    name: definition.name,
                    description: definition.description,
                    parameters: definition.parameters
                }
            })) as OllamaTool[]
        );
    }

    async _loop(
        model: string,
        messages: OllamaMessage[],
        tools: OllamaTool[]
    ): Promise<Response> {
        const response = await this._client.chat({
            model,
            messages,
            tools
        });

        const toolCalls = response.message.tool_calls ?? [];

        if (toolCalls.length === 0) {
            return {
                message: {
                    role: 'assistant',
                    content: response.message.content
                }
            };
        }

        messages.push(response.message);

        for (const call of toolCalls) {
            try {
                const result = await this.call(
                    call.function.name,
                    call.function.arguments
                );

                messages.push({
                    role: 'tool',
                    tool_name: call.function.name,
                    content: String(result)
                });
            } catch (error) {
                console.error(error);
                messages.push({
                    role: 'tool',
                    tool_name: call.function.name,
                    content: getErrorMessage(error)
                });
            }
        }

        return this._loop(model, messages, tools);
    }
}