import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Relative base so the build works from any path (S3 + CloudFront root or a sub-folder).
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { chunkSizeWarningLimit: 1000 },
})
