import type { CallToolResult } from '@modelcontextprotocol/server';

import type { JsonSchema } from '@olegpolyakov/core/utils/schema';

import type { ToolDefinition } from './types';

export class Tool {
    constructor(
        public name: string,
        public description: string = '',
        public inputSchema: JsonSchema,
        protected fn: (args: unknown) => Promise<CallToolResult>
    ) {}

    get definition(): ToolDefinition {
        return {
            name: this.name,
            description: this.description,
            parameters: this.inputSchema
        };
    }

    call(args: unknown): Promise<CallToolResult> {
        return this.fn(args);
    }
}