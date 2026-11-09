import { vi } from 'vitest';

const mocks = vi.hoisted(() => ({
    constructor: vi.fn(),
    models: vi.fn(),
    chat: vi.fn()
}));

vi.mock('ollama', () => ({
    Ollama: class {
        list = mocks.models;
        chat = mocks.chat;

        constructor(options: unknown) {
            mocks.constructor(options);
        }
    }
}));

export default () => mocks;