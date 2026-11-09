import OpenAI from 'openai';
import { toResponseInputItems } from 'openai/lib/responses/ResponseInputItems';
import type { ResponseInputFile, ResponseInputImage, ResponseInputText } from 'openai/resources/responses/responses';

import {
    AIClient,
    ClientOptions,
    getErrorMessage,
    Message,
    RespondOptions,
    Response, 
    Tool
} from '../lib';

export default class OpenAIClient extends AIClient {
    private _client: OpenAI;

    constructor({
        apiKey,
        baseURL,
        tools,
        model = ''
    }: ClientOptions) {
        super(model, tools);
        this._client = new OpenAI({ apiKey, baseURL });
    }

    async getModels() {
        const res = await this._client.models.list();

        return res.data;
    }

    async respond(messages: Message[], { model = this._model, tools }: RespondOptions = {}): Promise<Response> {
        return this.loop(messages, model, tools);
    }

    async loop(
        messages: Message[],
        model: string,
        tools: Tool[] = []
    ): Promise<Response> {
        const input = messages.map(message => ({
            role: message.role,
            content: !Array.isArray(message.content)
                ? message.content
                : message.content.map(item => {
                    switch (item.type) {
                        case 'image': return {
                            type: 'input_image',
                            image_url: item.url
                        } as ResponseInputImage;

                        case 'file': return {
                            type: 'input_file',
                            filename: item.name,
                            file_url: item.url,
                            file_data: item.content
                        } as ResponseInputFile;

                        default: return {
                            type: 'input_text',
                            text: item.content
                        } as ResponseInputText;
                    }
                })
        }));

        this.tools = tools;

        const _tools = tools.map(({ definition }) => ({
            type: 'function' as const,
            name: definition.name,
            description: definition.description,
            parameters: toStrictSchema(definition.parameters),
            strict: true
        }) as OpenAI.Responses.Tool);
    
        return this._loop(model, input, _tools);
    }

    private async _loop(
        model: string,
        input: OpenAI.Responses.ResponseInputItem[],
        tools: OpenAI.Responses.Tool[] = []
    ): Promise<Response> {
        const response = await this._client.responses.create({
            model,
            input,
            tools
        });

        if (response.output.length === 0) throw new Error(
            'The model returned no output'
        );

        const functionCalls = response.output.filter(
            item => item.type === 'function_call'
        );

        if (functionCalls.length === 0) return {
            message: {
                content: response.output_text,
                role: 'assistant'
            }
        };

        input.push(...toResponseInputItems(response.output));

        for (const item of functionCalls) {
            try {
                const tool = this.tools.find(
                    tool => tool.name === item.name
                );

                if (!tool) throw new Error('No tool found');

                const args = stripAddedNulls(JSON.parse(item.arguments), tool.parameters);

                const result = await this.call(tool.name, args);

                input.push({
                    type: 'function_call_output',
                    call_id: item.call_id,
                    output: JSON.stringify(result)
                });
            } catch (error) {
                console.error(error);
                input.push({
                    type: 'function_call_output',
                    call_id: item.call_id,
                    output: getErrorMessage(error)
                });
            }
        }

        return this._loop(model, input, tools);
    }
}

/**
 * Converts a standard JSON schema to OpenAI strict mode:
 * every property is required, and the properties that were optional become nullable.
 */
export function toStrictSchema(schema: Schema): Schema {
    if (!schema || typeof schema !== 'object') return schema;

    const result: Schema = { ...schema };

    if (result.anyOf) {
        result.anyOf = result.anyOf.map(toStrictSchema);
    }

    if (result.items) {
        result.items = toStrictSchema(result.items);
    }

    if (result.type === 'object') {
        const required = new Set<string>(schema.required ?? []);
        const properties: Schema = {};

        for (const [key, value] of Object.entries(schema.properties ?? {})) {
            const strict = toStrictSchema(value as Schema);
            properties[key] = required.has(key) ? strict : makeNullable(strict);
        }

        result.properties = properties;
        result.required = Object.keys(properties);
        result.additionalProperties = false;
    }

    return result;
}

type Schema = Record<string, any>;

function makeNullable(schema: Schema): Schema {
    if (Array.isArray(schema.type)) {
        return schema.type.includes('null')
            ? schema
            : { ...schema, type: [...schema.type, 'null'] };
    }

    if (typeof schema.type === 'string') {
        return {
            ...schema,
            type: [schema.type, 'null'],
            // With an enum, null must also be listed as a value
            ...(schema.enum && { enum: [...schema.enum, null] })
        };
    }

    return { anyOf: [schema, { type: 'null' }] };
}

/**
 * Removes the nulls that toStrictSchema added.
 * A null is removed only if the property was optional in the original schema.
 * A property that was already required and nullable keeps its null.
 */
export function stripAddedNulls(value: unknown, schema?: Schema): unknown {
    if (Array.isArray(value)) {
        return schema?.items
            ? value.map(item => stripAddedNulls(item, schema.items))
            : value;
    }

    if (value && typeof value === 'object') {
        const required = new Set<string>(schema?.required ?? []);
        const result: Record<string, unknown> = {};

        for (const [key, v] of Object.entries(value)) {
            if (v === null && !required.has(key)) continue;
            result[key] = stripAddedNulls(v, schema?.properties?.[key]);
        }

        return result;
    }

    return value;
}