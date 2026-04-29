import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Base path: served under /NiivueRL/ on GitHub Pages, '/' for local dev.
// Override with VITE_BASE if you fork or rename the repo.
const base = process.env.VITE_BASE ?? (process.env.GITHUB_ACTIONS ? '/NiivueRL/' : '/')

export default defineConfig({
  base,
  plugins: [
    react(),
    {
      name: 'configure-response-headers',
      configureServer(server) {
        server.middlewares.use((_req, res, next) => {
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
          res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
          next()
        })
      },
    },
  ],
})
