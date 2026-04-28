import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // 加载环境变量，process.env 在 vite.config.ts 中默认不包含自定义变量
  // 设置第三个参数为空字符串 '' 来加载所有环境变量，或者保持 'VITE_' 前缀限制
  const env = loadEnv(mode, process.cwd(), '')
  
  return {
    plugins: [vue()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, 'src')
      }
    },
    server: {
      host: '0.0.0.0', // 允许外部访问
      port: 5173,      // 明确指定端口
      proxy: {
        '/api': {
          // 优先使用环境变量中的后端地址，否则回退到 localhost
          target: env.BACKEND_URL || 'http://localhost:8000',
          changeOrigin: true
        },
        '/health': {
          target: env.BACKEND_URL || 'http://localhost:8000',
          changeOrigin: true
        }
      }
    }
  }
})
