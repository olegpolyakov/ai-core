import type { JsonSchema } from '@olegpolyakov/core/utils/schema';

export interface Message {
    role: MessageRole;
    content: string | MessageContent[];
}

export type MessageRole = 'system' | 'user' | 'assistant';

export type MessageContentType = 'text' | 'image' | 'document';

export type MessageContent = TextContent | ImageContent | FileContent;

export type TextContent = {
    type: 'text';
    content: string;
};

export type ImageContent = {
    type: 'image';
    url: string;
};

export type FileContent = {
    type: 'file';
    name?: string;
} & ({
    content?: never;
    url: string;
} | {
    content: string;
    url?: never;
});

export interface ToolDefinition {
    name: string;
    description?: string;
    parameters: JsonSchema;
}

export interface Response {
    message: ResponseMessage;
}

export type ResponseMessage = {
    role: 'assistant';
    content: string;
};