import type { JsonSchema } from '@olegpolyakov/core/utils/schema';

import { Tool } from './tool';
import type { Message, Response, ToolDefinition } from './types';

export type ClientOptions = {
    apiKey: string;
    baseURL?: string;
    tools?: Tool[];
    model?: string;
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

    constructor(model: string = '', tools: Tool[] = []) {
        this._model = model;
        this._tools = tools.reduce((acc, tool) => {
            acc[tool.definition.name] = tool;
            return acc;
        }, {} as Record<string, Tool>);
    }
    
    public get tools(): ToolDefinition[] {
        return Object.values(this._tools).map(tool => tool.definition);
    }

    public set tools(tools: Tool[]) {
        this._tools = Object.fromEntries(tools.map(t => [t.name, t]));
    }

    // TODO Make a universal return type for models
    public abstract getModels(): Promise<unknown>;

    abstract respond(
        messages: Message[],
        options?: RespondOptions
    ): Promise<Response>;

    protected abstract loop(
        messages: Message[],
        model: string,
        tools?: Tool[]
    ): Promise<Response>;

    protected async call(name: string, args: unknown): Promise<unknown> {
        const tool = this._tools[name];

        if (!tool) throw new Error('Unknown tool');

        return tool.call(args);
    }
}