import type { JsonSchema } from '@olegpolyakov/core/utils/schema';

import { Tool } from './tool';
import type { Message, Response, ToolDefinition } from './types';

export type ClientOptions = {
    apiKey: string;
    baseURL?: string;
};

export type RespondOptions = {
    model?: string;
    format?: string | JsonSchema;
    stream?: boolean;
    tools?: Tool[]
};

export abstract class AIClient {
    protected _model: string = '';
    protected _tools: Record<string, Tool> = {};
    
    public get tools(): ToolDefinition[] {
        return Object.values(this._tools).map(tool => tool.definition);
    }

    // TODO Make a universal return type for models
    public abstract getModels(): Promise<unknown>;

    public respond(
        messages: Message[],
        {
            model = this._model,
            tools
        }: RespondOptions = {}
    ): Promise<Response> {
        if (tools) {
            this._tools = Object.fromEntries(tools.map(t => [t.name, t]));
        }

        return this.loop(model, messages, tools);
    }

    protected abstract loop(
        model: string,
        messages: Message[],
        tools?: Tool[]
    ): Promise<Response>;

    protected async call(name: string, args: unknown): Promise<unknown> {
        const tool = this._tools[name];

        if (!tool) throw new Error('Unknown tool');

        return tool.call(args);
    }
}