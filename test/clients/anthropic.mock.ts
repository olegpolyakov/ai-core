import { vi } from 'vitest';

const mocks = vi.hoisted(() => ({
    constructor: vi.fn(),
    models: vi.fn(),
    create: vi.fn()
}));

vi.mock('@anthropic-ai/sdk', () => ({
    default: class {
        models = { list: mocks.models };
        messages = { create: mocks.create };

        constructor(options: unknown) {
            mocks.constructor(options);
        }
    }
}));

export default () => mocks;