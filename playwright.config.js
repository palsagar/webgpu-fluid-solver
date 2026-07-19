import { defineConfig } from '@playwright/test';

// WebGPU needs a real GPU-backed Chromium. Headless shell has no WebGPU, so
// these run headed by default; CI must supply a display (e.g. xvfb-run).
export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  use: {
    // Dedicated port: 8000 is the dev default and may hold an unrelated server
    baseURL: 'http://127.0.0.1:8321',
    headless: false,
    launchOptions: {
      args: ['--enable-unsafe-webgpu'],
    },
  },
  webServer: {
    command: 'uv run uvicorn server:app --port 8321',
    url: 'http://127.0.0.1:8321/api/health',
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
