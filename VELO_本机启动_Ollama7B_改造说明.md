# Velo 本机启动改造记录（Ollama 7B 方案）

## 1. 目标
本次改造的目标是：

1. 放弃把 Docker / vLLM 作为默认本机开发路径。
2. 把原先依赖 **14B vLLM 模型** 的本地调用方式，切换为 **Ollama + 7B 模型**。
3. 补齐一套可直接在宿主机运行的启动方案。
4. 记录所有需要调整的代码与配置，方便后续排查和继续迭代。

---

## 2. 为什么最终选择 Ollama，而不是继续用 vLLM

### vLLM 的优点
- 在高性能 GPU 环境下，吞吐和大模型推理能力通常更强。
- 对 OpenAI 兼容接口支持较成熟。
- 如果已经有现成 GPU 服务器和量化模型，性能表现会更好。

### vLLM 的问题（针对本项目当前场景）
- 当前仓库里的 vLLM 路线强绑定 Docker Compose。
- 原配置直接指向 14B GPTQ 模型，硬件门槛更高。
- 需要额外准备模型文件、GPU/CUDA/NCCL 环境，纯本机启动成本高。
- 对普通开发机来说，启动成功率和维护成本都不理想。

### Ollama 的优点
- 本机安装和拉模型都更简单。
- 更适合作为默认开发环境。
- 对 7B 模型支持比较友好，资源需求明显低于 14B。
- 能提供本机 OpenAI 兼容接口，便于当前项目最小改造接入。

### 最终选择
**默认采用 Ollama。**

原因不是它绝对更快，而是它在“部署方便性 + 本机开发成功率 + 足够可用的响应速度”这个综合维度上，更适合 Velo 当前诉求。

---

## 3. 综合评估：vLLM vs Ollama

| 维度 | vLLM | Ollama |
|---|---|---|
| 本机安装难度 | 高 | 低 |
| GPU 依赖强度 | 高 | 中/低 |
| 对 14B/大模型支持 | 更强 | 一般 |
| 对 7B 本机开发适配 | 一般 | 更适合 |
| 项目当前接入成本 | 较高 | 较低 |
| 启动成功率 | 较低 | 较高 |
| 开发期综合便利性 | 一般 | 更好 |

### 结论
- 如果目标是 **本机开发优先、尽快跑起来**：选 **Ollama**。
- 如果未来目标是 **高吞吐 GPU 推理平台**：可以再考虑回到 **vLLM**。

---

## 4. 本次代码修改点

### 4.1 后端统一 AI 配置
修改文件：
- `backend/app/core/config.py`

变更内容：
- 新增了本机 Ollama 优先的统一配置：
  - `LLM_PROVIDER`
  - `LLM_BASE_URL`
  - `LLM_MODEL`
  - `EMBEDDING_PROVIDER`
  - `EMBEDDING_BASE_URL`
  - `EMBEDDING_MODEL`
- 保留了 `VLLM_API_URL`，但不再作为默认主路径。
- 将 `OPENAI_API_KEY` 默认值改成 `ollama`，用于本机 OpenAI 兼容客户端占位。
- 新增：
  - `chat_api_base`
  - `completion_api_base`
  - `embedding_api_base`

### 4.2 补全接口从 vLLM 14B 改为统一读取本地模型配置
修改文件：
- `backend/app/services/llm_service.py`

变更内容：
- 去掉原先写死的 14B 模型名。
- 改为从 `settings.LLM_MODEL` 读取模型。
- 改为从 `settings.completion_api_base` 读取本机服务地址。
- 保留原有前后文裁剪和结果后处理逻辑。
- 默认走 Ollama 提供的 OpenAI 兼容接口。

### 4.3 聊天 / RAG 默认切换到本机 Ollama
修改文件：
- `backend/app/services/agent_service.py`

变更内容：
- 原先聊天模型写死为 `gpt-3.5-turbo`，现改为 `settings.LLM_MODEL`。
- Embedding 模型改为 `settings.EMBEDDING_MODEL`，并走本机 `EMBEDDING_BASE_URL`。
- 去掉“没有 OpenAI key 就直接跳过索引”的逻辑，以兼容本机 Ollama 模式。
- 保留原有 Chroma + LangChain + Redis 的整体结构，降低改造风险。

### 4.4 前端统一走相对路径 API，不再写死 localhost
修改文件：
- `frontend/src/stores/editorStore.ts`

变更内容：
- 删除 `axios.create({ baseURL: 'http://localhost:8000/api/v1' })`
- 改为复用 `frontend/src/api/request.ts` 提供的统一请求客户端。
- 所有文档相关请求统一走 `/api/v1` 相对路径，由 Vite proxy 转发。

这样做的好处是：
- 本机跑前端时不再需要额外改地址。
- 后续如果换后端地址，只需要改 Vite proxy 或环境变量，不用动业务代码。

### 4.5 本机启动文档更新
修改文件：
- `启动.md`

变更内容：
- 改成以 **Ollama 7B 本机启动** 为主线。
- 补齐了：
  - Ollama 安装/启动
  - 模型拉取
  - 环境变量设置
  - backend 启动命令
  - frontend 启动命令
  - Redis 降级说明
  - RAG/embedding 说明

---

## 5. 从 14B 改成 7B 的关键变化

### 原本
- 补全路径中硬编码模型：
  - `Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4`
- 依赖 vLLM 容器与模型目录挂载
- 适合 GPU 更强、模型资源已准备好的环境

### 现在
- 默认模型拆分为：
  - 补全：`qwen2.5-coder:14b`
  - RAG / 聊天：`qwen2.5:7b-instruct`
  - Embedding：`BAAI/bge-small-zh-v1.5`（CPU）
- 部署方式改为：
  - `ollama pull`
  - `ollama serve`
  - 本地 CPU embedding
- 更适合 3090 这类单卡机器做“补全效果优先 + RAG 可用”的均衡方案

### 代价
- 7B 模型整体能力通常弱于 14B。
- 某些长上下文和复杂生成任务的质量可能下降。

### 收益
- 更容易部署。
- 更容易跑起来。
- 开发调试成本更低。

---

## 6. 当前默认建议环境变量

```bash
export POSTGRES_SERVER=sqlite
export DATA_DIR=/root/Velo/backend/data
export REDIS_HOST=localhost
export LLM_PROVIDER=ollama
export LLM_BASE_URL=http://localhost:11434/v1
export COMPLETION_MODEL=qwen2.5-coder:14b
export CHAT_MODEL=qwen2.5:7b-instruct
export EMBEDDING_PROVIDER=huggingface
export EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
export OPENAI_API_KEY=ollama
```

---

## 7. 验证建议

### 基础验证
1. `ollama --version`
2. `ollama list`
3. 后端是否能启动
4. 前端是否能启动
5. 页面是否能打开并正常请求后端

### AI 验证
1. `/api/v1/completion` 是否能返回补全文本
2. 聊天接口是否能正常回复
3. 如启用 RAG，embedding 是否可用
4. 索引一个文档后是否能检索到来源

---

## 8. 当前仍需注意的问题

1. **本机当前未安装 Ollama**
   - 我在实际验证时执行了 `ollama --version`，结果是命令不存在。
   - 所以代码和文档已经切到 Ollama 方案，但宿主机还需要你先安装 Ollama。

2. **模型标签可能因 Ollama 版本而略有差异**
   - 如果 `qwen2.5:7b-instruct` 拉取失败，需要换成你本机可用的 7B 指令模型，并同步修改 `CHAT_MODEL`。

3. **RAG 使用本地 embedding 模型后，如之前已有旧 embedding 索引，可能需要重建索引**
   - `BAAI/bge-small-zh-v1.5` 通过本地 CPU 加载，不走 Ollama。
   - 如果之前已有旧 embedding 索引，建议重建，否则向量空间可能不兼容。

4. **Docker Compose 仍然保留着旧 vLLM 路线信息**
   - 当前这次改造重点是“纯本机启动”。
   - 如果后续你希望，我可以继续把 `docker-compose.yml` 也一起改成 Ollama 版，或者明确标注成旧方案。

---

## 9. 下一步建议
如果继续往下做，建议按这个顺序：

1. 先在宿主机安装 Ollama。
2. 拉取 `qwen2.5-coder:14b` 和 `qwen2.5:7b-instruct`。
3. 安装 Python 依赖，确保可本地加载 `BAAI/bge-small-zh-v1.5`。
4. 启动后端并测试 `/completion`。
5. 启动前端验证页面交互。
6. 如需 RAG，重建向量索引后再验证问答。

---

## 10. 一句话总结
这次改造的核心不是单纯把 14B 改成 7B，而是把 Velo 的本机开发默认路径从“依赖 Docker + vLLM + 大模型资源”切换为“基于 Ollama 的轻量本机开发方案”。
