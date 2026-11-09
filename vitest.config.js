import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        globalSetup: './test/setup.ts',
        silent: 'passed-only'
    },
    resolve: {
        alias: {
            '@': './src'
        }
    }
});