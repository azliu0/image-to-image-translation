import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig(({ command, mode }) => {
  return {
    plugins: [react()],
    assetsInclude: ['**/*.md'],
    server: {
      port: 5173,
      host: true,
      proxy: {
        "/api": {
          target: "https://image-to-image-translation-server.onrender.com:3000",
          changeOrigin: true,
        },
      },
    },
  };
});