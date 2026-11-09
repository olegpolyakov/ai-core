import { vi } from 'vitest';

const mocks = vi.hoisted(() => ({
    constructor: vi.fn(),
    models: vi.fn(),
    create: vi.fn()
}));

vi.mock('openai', () => ({
    default: class {
        models = { list: mocks.models };
        responses = { create: mocks.create };

        constructor(options: unknown) {
            mocks.constructor(options);
        }
    }
}));

export default () => mocks;