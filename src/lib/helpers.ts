import type { Message } from './types';

export function withUserId(userId: string, messages: Message[]): Message[] {
    const conversation = messages.map(message => ({ ...message }));
    const systemMessage = conversation.find(message => message.role === 'system');

    if (systemMessage) {
        systemMessage.content = `User ID: ${userId}\n${systemMessage.content}`;
    } else {
        conversation.unshift({
            role: 'system',
            content: `User ID: ${userId}`
        });
    }

    return conversation;
}

export function getErrorMessage(error: unknown): string {
    if (error instanceof Error) {
        return `${error.name} ${error.message}`;
    }

    return String(error);
}