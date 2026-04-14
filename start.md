# start

## 1. 启动 Ollama
```bash
ollama serve
```
如果已经在运行，可跳过。

## 2. 拉取模型
```bash
ollama pull qwen2.5-coder:14b
ollama pull qwen2.5:7b-instruct
```
Embedding 使用本地 CPU 模型 `BAAI/bge-small-zh-v1.5`，不需要通过 Ollama 拉取。

## 3. 安装后端依赖
```bash
cd /root/Velo
/root/Velo/.venv/bin/pip install -r backend/requirements.txt
```

## 4. 启动后端
```bash
cd /root/Velo/backend
DATA_DIR=/root/Velo/backend/data \
POSTGRES_SERVER=sqlite \
LLM_PROVIDER=ollama \
LLM_BASE_URL=http://localhost:11434/v1 \
COMPLETION_MODEL=qwen2.5-coder:14b \
CHAT_MODEL=qwen2.5:7b-instruct \
EMBEDDING_PROVIDER=huggingface \
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5 \
OPENAI_API_KEY=ollama \
/root/Velo/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 5. 启动前端
```bash
cd /root/Velo/frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

## 6. 打开页面
浏览器访问：
```text
http://127.0.0.1:5173
```

## 7. 最小验证
- 前端能打开页面
- 后端接口可访问：`http://127.0.0.1:8000/`
- 补全走 `qwen2.5-coder:14b`
- RAG / 聊天走 `qwen2.5:7b-instruct`
- embedding 走 CPU 的 `BAAI/bge-small-zh-v1.5`
