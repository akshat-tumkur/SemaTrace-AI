import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', 'SEMATRACE_API_PROXY_TARGET')
  return {
    plugins: [react()],
    // Match the documented native beta launcher. Docker uses nginx separately.
    server: { proxy: { '/api': env.SEMATRACE_API_PROXY_TARGET || 'http://127.0.0.1:8010' } },
  }
})
