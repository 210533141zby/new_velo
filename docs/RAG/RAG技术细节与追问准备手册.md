# RAG技术细节与追问准备手册

## 1. 文档定位

这份文档是写给你自己的。

目标不是“概括介绍”，而是让你在老师连续追问 20 到 30 分钟时，依然能把下面这些问题答稳：

1. 当前系统到底怎么跑。
2. 每个模块具体负责什么。
3. 关键公式、阈值、策略是不是和代码一致。
4. 哪些说法是旧文档里的历史口径，不能再当成当前事实。
5. 当前系统的真实优点、真实不足和真实实验结果分别是什么。

这份文档严格遵循一个原则：

> 以当前工作区代码为准。  
> 旧文档、旧测试、旧表述如果和运行时代码冲突，以代码行为为准。

---

## 2. 阅读范围与依据

这份手册基于我对当前目录下 RAG 相关代码和材料的交叉阅读整理，重点覆盖了以下内容：

### 2.1 后端核心源码

1. `backend/app/main.py`
2. `backend/app/api/rag.py`
3. `backend/app/api/content.py`
4. `backend/app/core/config.py`
5. `backend/app/models.py`
6. `backend/app/schemas.py`
7. `backend/app/services/content/content_service.py`
8. `backend/app/services/model_factory/__init__.py`
9. `backend/app/services/rag/*.py`
10. `backend/app/cache.py`
11. `backend/app/logger.py`

### 2.2 前端交互链路

1. `frontend/src/api/chat.ts`
2. `frontend/src/api/documents.ts`
3. `frontend/src/components/Copilot/ChatPanel.vue`
4. `frontend/src/stores/editorStore.ts`
5. `frontend/src/types/chat.ts`

### 2.3 测试与探针

1. `backend/tests/test_hybrid_retrieval.py`
2. `backend/tests/test_rag_pipeline_execution.py`
3. `backend/tests/test_rag_pipeline_routing.py`
4. `backend/tests/test_unified_evidence_scorer.py`
5. `backend/scripts/run_rag_probes.py`

### 2.4 现有文档与实验材料

1. `docs/RAG/*.md`
2. `experiments/mainstream_rag_benchmark.py`
3. `experiments/ablation_study.py`
4. `experiments/主流RAG检索对比实验/*`
5. `experiments/消融实验_模块贡献分析/*`
6. `llamaindex_rag_eval/*.py`
7. `CRUD-RAG/*.py`

### 2.5 当前数据快照

我还额外核对了：

1. `backend/data/wiki.db` 当前有效文档
2. `backend/data/chroma_db` 当前向量集合名和 chunk 数量
3. 当前评测输出 JSON / CSV 摘要
4. 当前单测运行结果

所以这份手册不是只根据旧文档改写，而是“代码 + 评测 + 数据快照”三方交叉后的结果。

---

## 3. 当前运行面快照

以下内容是基于当前工作区的运行时事实，不是泛泛而谈。

### 3.1 基础信息

| 项目 | 当前值 |
| --- | --- |
| 后端框架 | FastAPI |
| 问答接口 | `/api/v1/agent/chat` |
| 健康检查 | `/health` |
| 主服务入口 | `app.services.rag.rag_service.RagService.rag_qa` |
| 数据库 | SQLite |
| 数据库文件 | `backend/data/wiki.db` |
| 向量库 | Chroma |
| 向量目录 | `backend/data/chroma_db` |
| 缓存 | Redis 优先，失败降级内存缓存 |
| Chat 模型 | `qwen2.5:7b-instruct` |
| Embedding 模型 | `nomic-embed-text:latest` |
| Rerank 模型 | `BAAI/bge-reranker-v2-m3` |

### 3.2 当前知识库内容

当前数据库中 `is_active=1` 的有效文档只有 4 篇：

| doc_id | 标题 | 作用类型 |
| --- | --- | --- |
| 33 | 量子计算与Shor算法 | 事实问答、编号/年份、礼貌问法 |
| 34 | 线粒体DNA的母系遗传机制 | 生物学事实问答 |
| 35 | 北宋熙宁变法中的经济政策 | 列举、多信息、政策定义 |
| 38 | 雾潮镇的档案馆 | 长文档、概况、关系、原因、地点、多角色 |

### 3.3 当前向量库快照

当前 Chroma collection 名称是：

```text
velo_ollama_nomic-embed-text_latest
```

当前 collection 总 chunk 数量是：

```text
7
```

各文档 chunk 分布如下：

```text
33 -> 1
34 -> 1
35 -> 1
38 -> 4
```

这个细节很有用，因为老师如果问“为什么 long document 更适合展示你的系统能力”，你可以直接回答：

1. 短文档只切 1 个 chunk，更像原子事实检索。
2. 长文档切成多个 chunk，才会暴露 chunk 召回、候选融合、全文综合、关系问题、多信息问题这些真正的 RAG 难点。

---

## 4. 一句话总览

当前系统的真实主链路可以概括成：

```text
文档写入 SQLite
-> 后台切块并写入 Chroma
-> 用户提问进入 RagService
-> QueryIntentBuilder 生成意图画像
-> 向量召回 + 文档级 BM25
-> adaptive hybrid 候选构造
-> cross-encoder rerank
-> UnifiedEvidenceScorer 统一证据判定
-> AnswerModeRouter 决定回答模式
-> GeneratorFactory 执行抽取/结构化/生成
-> 返回 response + sources
```

这套设计的关键词不是“生成能力”，而是：

1. 查询画像
2. 混合检索
3. 统一证据评分
4. 显式回答路由

---

## 5. 模块地图

下面这张表一定要熟，因为老师追问时你随时会用到。

| 文件 | 模块职责 | 你怎么讲 |
| --- | --- | --- |
| `backend/app/api/rag.py` | RAG HTTP 接口入口 | 前端问题进入后端的第一站 |
| `backend/app/services/rag/rag_service.py` | 主编排器 | 真正的主链路控制中枢 |
| `backend/app/services/rag/query_intent_builder.py` | query 理解与画像 | 先决定怎么搜、怎么防御 |
| `backend/app/services/rag/hybrid_search.py` | BM25、RRF、自适应融合、候选池 | 你的核心检索改进区 |
| `backend/app/services/rag/rerank_service.py` | cross-encoder 精排 | 候选深度判断 |
| `backend/app/services/rag/evidence_scorer.py` | 统一证据评分 | 决定文档能不能用 |
| `backend/app/services/rag/evidence_judge.py` | 选择性 Judge | 高风险 query 的语义复核 |
| `backend/app/services/rag/answer_mode_router.py` | 回答模式路由 | 决定拒答还是怎么答 |
| `backend/app/services/rag/answer_generators.py` | 回答生成与降级 | 具体输出答案 |
| `backend/app/services/rag/vector_index_service.py` | 文档切块、embedding、Chroma 写入 | 入库链路 |
| `backend/app/services/model_factory/__init__.py` | Chat/Judge 模型工厂 | 模型实例管理 |
| `backend/app/api/content.py` | 文档 CRUD 接口 | 文档写入触发索引 |
| `frontend/src/components/Copilot/ChatPanel.vue` | 前端问答 UI | 用户如何看到答案和引用 |

---

## 6. 数据层和存储层

### 6.1 SQLite 负责什么

SQLite 存的是文档原文和基础元数据，核心表包括：

1. `documents`
2. `folders`
3. `system_logs`

其中和 RAG 最相关的是 `documents`：

1. `id`
2. `title`
3. `content`
4. `summary`
5. `tags`
6. `is_active`
7. `folder_id`

你要注意两个追问点：

1. `summary` 和 `tags` 字段在模型里预留了，但当前 RAG 主链路并没有自动生成它们，也没有在检索中使用它们。
2. `folder_id` 只参与前端组织文档，不参与当前检索过滤；当前 RAG 默认会对所有 `is_active=True` 的文档做全局检索。

### 6.2 Chroma 负责什么

Chroma 只负责存 chunk 级向量和 chunk 元数据。

每个 chunk 当前至少会带这几个 metadata：

1. `source`
2. `doc_id`
3. hybrid 阶段追加的 `vector_score`
4. `bm25_score`
5. `rrf_score`
6. `adaptive_score`
7. `coverage_score`
8. `identifier_overlap`
9. `candidate_source`

### 6.3 Redis / 内存缓存负责什么

当前缓存不是单点依赖。

系统会：

1. 优先连接 Redis
2. 如果 Redis 不可用，自动降级到进程内存缓存

当前缓存两类数据：

1. 文档列表缓存 `documents_list`，TTL 为 300 秒
2. RAG 问答缓存 `rag:response:v23:<md5(query)>`，TTL 为 3600 秒

这两个 TTL 你最好记住，因为老师如果问“缓存有没有设计”，你可以直接报出具体行为。

---

## 7. 文档入库与索引构建

## 7.1 文档写入时发生了什么

文档增删改入口在 `backend/app/api/content.py`。

当前流程是：

1. `POST /documents/` 创建文档
2. `PUT /documents/{doc_id}` 更新文档
3. `DELETE /documents/{doc_id}` 软删除文档

创建和更新后，只要文档有内容，就会通过 `BackgroundTasks` 异步触发：

1. `index_document_chunks(doc_id, title, content)`
2. 或 `delete_document_chunks(doc_id)`

这里的关键词是：

> 后台任务、最终一致性

也就是说，接口返回成功并不代表向量索引已经同步完成。

### 7.2 切块策略

`vector_index_service.py` 中的索引逻辑分两步：

1. `MarkdownHeaderTextSplitter`
2. `RecursiveCharacterTextSplitter`

具体参数是：

1. 标题层级：`# / ## / ###`
2. `chunk_size = 1000`
3. `chunk_overlap = 200`

如果文档没有被 Markdown splitter 切出结果，才会直接按整篇 fallback 切块。

### 7.3 轻量语义锚点注入

这是当前代码里比较值得讲的一个点。

`vector_index_service.py` 会在入库前做一件事：

1. 从标题中抽词，权重大
2. 从首段中用 `jieba.posseg` 抽名词，权重稍低
3. 去掉一些核心停用词，比如“文档、资料、介绍、内容、背景、历史、概况、总结”等
4. 取出现频次和权重最高的 1 到 3 个实体
5. 生成类似：

```text
【核心主题：实体1、实体2】
```

然后把它注入到每个 chunk 的开头。

### 7.4 这么做的意义

你要会把它讲清楚：

1. 不是为了让 chunk 更长。
2. 不是为了让大模型直接看到花哨标签。
3. 而是为了在 embedding 阶段补一个轻量主题锚点，降低“文档只是顺带提到某个词，但主题其实不是它”的误召回。

### 7.5 文档删除如何处理

删除并不是直接删数据库行，而是：

1. SQLite 中 `is_active=False`
2. Chroma 中按 `where={'doc_id': doc_id}` 删除对应 chunk

这样数据库保留审计历史，检索面则把它移出。

---

## 8. 应用启动时的 RAG 行为

`backend/app/main.py` 的启动阶段和 RAG 关系很大。

### 8.1 生命周期里会做什么

应用启动时会做：

1. 初始化 Redis
2. 初始化数据库
3. 尝试执行 `RagService(session).ensure_bootstrap_index()`
4. 如果 `AI_WARMUP_ON_STARTUP=True`，会预热聊天模型
5. 还会预热 rerank 模型

### 8.2 `ensure_bootstrap_index()` 的逻辑

`RagService.ensure_bootstrap_index()` 的逻辑是：

1. 先查所有有效文档
2. 再查当前 collection 的 chunk 数量
3. 如果 collection 为空，就逐篇文档重建索引
4. 如果 collection 已有内容，就不重建

这意味着：

1. 新环境第一次启动时会自动建索引
2. 不是每次启动都重新嵌入全部文档

---

## 9. HTTP 接口到主链路的真实入口

### 9.1 接口签名

`backend/app/api/rag.py` 中：

```python
@rag_router.post('/chat', response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest, rag_service: RagService = Depends(...)):
    user_query = request.messages[-1].content
    result = await rag_service.rag_qa(user_query)
```

这个片段非常值得记。

### 9.2 这段代码说明了什么

它说明三件重要的事实：

1. 后端虽然接收的是 `messages` 列表，但当前只使用最后一条消息内容。
2. `ChatRequest` 里虽然有 `doc_id` 字段，但当前 RAG 接口没有使用它。
3. 这意味着当前系统从产品角度更准确地说是“单轮知识库问答”，不是完整多轮记忆型 RAG 会话。

如果老师问你“是不是支持多轮对话上下文”，你不能直接说“支持”。  
准确回答应该是：

> 当前 schema 预留了多轮消息结构，但运行时代码只读取最后一条用户消息，所以现在真实能力是单轮知识库问答。

---

## 10. RagService 主编排器

`rag_service.py` 是你最需要吃透的文件。

## 10.1 主函数 `rag_qa(query)`

执行顺序如下：

1. 先查 RAG 响应缓存
2. 再判断是不是“你是什么模型”这类系统信息问题
3. 调 `QueryIntentBuilder.build(query)`
4. 记录 intent 日志
5. 调 `_retrieve_candidates(query, intent)`
6. 调 `evidence_scorer.assess_concurrently(candidates, intent)`
7. 只保留 `usable_assessments`
8. 调 `answer_router.route(intent, usable_assessments)`
9. 调 `_execute_with_fallback(...)`
10. 返回 `{'response': ..., 'sources': ...}`
11. 把结果写入缓存

### 10.2 系统信息短路

`prompt_templates.py` 中有一个特殊逻辑：

1. 如果问题是“你是什么模型”“用的什么模型”
2. 则直接返回当前 `CHAT_MODEL`
3. 不走知识库
4. `sources` 为空

这是一个很小但很实用的工程短路。

### 10.3 当前缓存键

缓存键格式是：

```text
rag:response:v23:<md5(query)>
```

`v23` 是人工维护的 cache version。

它的意义是：

1. 当主链路语义发生变化时，可以手动换版本，避免旧缓存污染新逻辑。
2. 但当前代码并没有在文档变更后自动删除这类缓存。

这个问题后面会单独讲。

---

## 11. QueryIntentBuilder：系统先“理解问题”，再“检索问题”

这是当前系统的第一层控制面。

## 11.1 输出对象

最终返回的是一个 `QueryIntent`，包含这些字段：

1. `original_query`
2. `normalized_query`
3. `keyword_query`
4. `intent_type`
5. `retrieval_depth`
6. `defense_profile`
7. `evidence_requirement`
8. `wants_short_answer`
9. `needs_judge`
10. `trace_tags`

### 11.2 Query 规范化

主要包括：

1. 去掉礼貌前缀，比如“请问一下”“我想知道”“帮我看看”
2. 去掉句尾噪声和语气词，比如“吗”“呢”“呀”
3. 压缩空白

### 11.3 `keyword_query` 怎么生成

当前不会做重写式 query expansion，而是做轻量关键词视图。

核心逻辑是：

1. 分词
2. 去停用词
3. 优先保留 identifier
4. 再保留较长的、较有区分度的 token
5. 默认保留最多 8 个词

纠错型 query 是一个特例：

1. 会优先抽取“新闻开头”或“原文”部分
2. 最多保留 12 个词

### 11.4 意图类型

当前定义了 7 种：

1. `LOOKUP`
2. `FACTOID`
3. `RELATION`
4. `SUMMARY`
5. `OVERVIEW`
6. `REASON`
7. `LOCATION`

### 11.5 默认 `retrieval_depth`

默认映射关系如下：

| 意图 | 默认深度 |
| --- | --- |
| LOOKUP | 6 |
| FACTOID | 8 |
| LOCATION | 8 |
| REASON | 10 |
| RELATION | 12 |
| SUMMARY | 12 |
| OVERVIEW | 14 |

如果 query 含 identifier，再加 2。  
如果 token 数大于等于 10，再加 2。  
上限是 16。

### 11.6 但这里有一个很关键的“代码级细节”

虽然 `QueryIntentBuilder` 会生成 `retrieval_depth`，但在当前默认配置下，它对实际召回深度几乎没有影响。

原因是 `rag_service.py` 里写的是：

```python
vector_limit = max(intent.retrieval_depth, settings.RAG_VECTOR_SEARCH_LIMIT)
bm25_limit = max(intent.retrieval_depth, settings.RAG_BM25_SEARCH_LIMIT)
candidate_limit = max(intent.retrieval_depth, settings.RAG_HYBRID_CANDIDATE_LIMIT)
```

而默认配置是：

1. `RAG_VECTOR_SEARCH_LIMIT = 50`
2. `RAG_BM25_SEARCH_LIMIT = 50`
3. `RAG_HYBRID_CANDIDATE_LIMIT = 30`

所以当前默认运行面实际上是：

1. 向量先取 50
2. BM25 先取 50
3. 候选池取 30

`retrieval_depth` 只有在你把全局 limit 配得比它更小的时候，才会真正改变 fan-out。

这个细节非常容易被老师问住。你要明确：

> 代码里已经有按意图给 retrieval_depth 的机制，但在当前默认配置下，它更像一个预留控制量，而不是当前主要起作用的召回上限。

### 11.7 防御画像 `DefenseProfile`

当前分三档：

1. `STRICT`
2. `MODERATE`
3. `LOOSE`

大致规则：

1. `LOOKUP`、属性型 query 往往更严格
2. `SUMMARY`、`OVERVIEW` 更宽松
3. `RELATION + 共同点` 是特例，会放松到 `LOOSE`

### 11.8 证据要求 `EvidenceRequirement`

当前分三档：

1. `ATOMIC_SPAN`
2. `MULTI_SPAN`
3. `FULL_DOCUMENT`

映射规律：

1. `LOOKUP / FACTOID / LOCATION` 多数是 `ATOMIC_SPAN`
2. `RELATION / REASON` 多数是 `MULTI_SPAN`
3. `SUMMARY / OVERVIEW` 多数是 `FULL_DOCUMENT`

### 11.9 Judge 是否启用

`needs_judge` 不是“只要 STRICT 就一定 True”。

当前代码真实逻辑是：

1. `defense_profile` 必须是 `STRICT`
2. `evidence_requirement` 不能是 `MULTI_SPAN`
3. 不能是明显 exact-fact 时间问法
4. 意图必须是 `LOOKUP / LOCATION`，或者是属性型 query

这意味着：

1. 短实体 query 常会触发 Judge
2. 属性型 `X 的 Y` 常会触发 Judge
3. 关系型 `RELATION` 虽然常是 `STRICT`，但因为它们通常是 `MULTI_SPAN`，当前默认不走 Judge

这是当前代码和部分旧文档/旧测试最容易冲突的地方。

### 11.10 具体例子：当前运行时代码的真实输出

我实际跑出来的结果如下：

| query | intent_type | defense_profile | evidence_requirement | needs_judge |
| --- | --- | --- | --- | --- |
| 韩国与新加坡的关系 | RELATION | STRICT | MULTI_SPAN | False |
| 新加坡的档案馆 | FACTOID | STRICT | ATOMIC_SPAN | True |
| 沈阳 | LOOKUP | STRICT | ATOMIC_SPAN | True |
| 档案馆建于哪一年 | FACTOID | MODERATE | ATOMIC_SPAN | False |
| 为什么研究者重视编号为A-17-204的蓝布账册 | REASON | MODERATE | MULTI_SPAN | False |

如果老师问你 relation 为什么不走 Judge，你就按这张表回答，不要沿用旧说法。

---

## 12. 候选召回：不是只有向量检索

## 12.1 向量召回

向量召回来自 Chroma：

```python
self.vector_store.similarity_search_with_relevance_scores(retrieval_query, vector_limit)
```

Embedding 默认通过 `OpenAIEmbeddings` 调 Ollama OpenAI-compatible 接口。

也就是说：

1. 不是直接把本地模型手写对接
2. 而是通过 OpenAI 协议兼容层接入 Ollama

### 12.2 BM25 召回

BM25 是文档级，不是 chunk 级。

`hybrid_search.py` 会把每篇文档构造成 `IndexedDocument`：

1. `doc_id`
2. `title`
3. `content_preview`
4. `bm25_text`
5. `tokens`
6. `identifier_tokens`

其中 `bm25_text` 是：

```text
title + title + content
```

标题会重复一次，以提高标题命中权重。

### 12.3 为什么 BM25 用文档级而不是 chunk 级

因为当前设计里：

1. 向量侧已经负责 chunk 级召回，适合局部语义。
2. BM25 更适合用标题和整篇文本做词法锚定，尤其适合短标题、专有名词、编号、年份。

所以系统不是两路都在做同一件事，而是：

1. dense 看语义局部命中
2. BM25 看全文词法命中

---

## 13. Hybrid 检索：你的核心算法区

这部分一定要讲成“为什么这样设计”，而不是只背公式。

## 13.1 先 collapse 向量结果

一个文档可能对应多个 chunk。

`rag_service._collapse_scored_matches()` 会先按 source 聚合：

1. 对每个文档只保留最有代表性的 chunk
2. 这个“代表性”不仅看向量分数
3. 还会额外给 identifier 命中和 token 命中加分

具体做法：

1. `identifier_hits * 10.0`
2. `token_hits * 0.1`

这个设计的意义是：

1. 如果 query 包含编号，编号命中的 chunk 更可能被保留下来
2. 不会让同一篇文档的多个弱相关 chunk 挤占候选池

## 13.2 Query Profile

`hybrid_search.compute_query_profile()` 当前核心逻辑是：

1. 初始 `lexical_weight = 0.36`
2. 如果 query 含 identifier，`+0.18`
3. 如果 token 数 `<=4`，`+0.14`
4. 如果 token 数 `>=10`，`-0.08`
5. 如果存在显著高 IDF 稀有词，`+0.10`
6. 最后裁剪到 `[0.24, 0.78]`
7. `dense_weight = 1 - lexical_weight`

你解释时可以这么说：

1. 短 query 和编号 query 更依赖词法
2. 长 query 和抽象 query 更依赖语义
3. 这不是训练型学习排序，而是基于 query 特征的可解释权重调整

## 13.3 自适应融合分数

当前 `adaptive_score` 的代码公式是：

```text
adaptive_score
= dense_weight * normalized_dense
+ lexical_weight * normalized_bm25
+ 0.26 * normalized_rrf
+ 0.08 * coverage
+ 0.03 * identifier_overlap
```

这里每项的意义：

1. `normalized_dense`：语义相关
2. `normalized_bm25`：词法相关
3. `normalized_rrf`：双路排序稳定共识
4. `coverage`：query 关键词在候选里覆盖得够不够
5. `identifier_overlap`：编号/版本/数字是否一致

### 13.4 为什么还有 RRF

因为 dense 和 BM25 的量纲不同，RRF 提供了一种对排序位置更稳定的融合信号。

你可以这样答：

> dense 和 BM25 解决的是两个维度的问题，RRF 解决的是“这篇文档在两路里是不是都排得靠前”的问题，相当于给多视图一致性加一个稳定先验。

### 13.5 `coverage` 的作用

`coverage_ratio(query_tokens, candidate_tokens)` 表示 query 词项覆盖率。

它要解决的是：

1. 文档主题看起来差不多
2. 但问题里真正关键的词没有被覆盖
3. 这种情况下 dense 可能会过于乐观

### 13.6 `identifier_overlap` 的作用

主要面向：

1. 课程章节
2. 年份
3. 编号
4. 型号

但你要注意：

1. 公开消融实验中，identifier 约束没有显示稳定正收益
2. 所以你不能把它讲成“主要提升来源”
3. 更准确的说法是：它是一个工程上合理、但仍需继续验证的辅助信号

## 13.7 候选池构造

候选池不是直接取 adaptive 排序 Top-K。

当前逻辑是：

1. 先取 adaptive 排名前半偏多的一部分
2. 再补 RRF 排名高位
3. 再补 adaptive 剩余部分
4. 再补原向量列表
5. 再补 BM25 列表

目标是：

1. 不过早丢掉互补候选
2. 既保留 adaptive 强者，也保留双路共识和极端 lexical 命中

这正是公开消融里收益很大的模块之一。

### 13.8 一个很容易被忽略的细节：`hybrid_lexical_fallback`

如果一个文档：

1. 向量命中了某个 chunk
2. BM25 也命中了这篇文档
3. 但命中的那个 chunk 对 query 的 token 覆盖率太低

系统会把 chunk 候选替换成一份 synthetic doc，也就是基于文档预览构造的候选，并标记：

```text
candidate_source = hybrid_lexical_fallback
```

这个逻辑的价值是：

1. 避免向量误命中的局部 chunk 把整篇文档拖偏
2. 给 rerank 和后续评分一个更符合词法线索的候选视图

---

## 14. Hybrid 词法索引的刷新机制

这个点是当前代码里非常值得提前准备的“老师追问坑”。

## 14.1 设计意图

`hybrid_search.py` 内部维护了一个进程级 `HybridLexicalIndex` 缓存，并用：

1. `_hybrid_index`
2. `_hybrid_index_signature`
3. `_hybrid_index_needs_refresh`

来判断是否需要重建。

### 14.2 但当前代码存在一个真实风险

`invalidate_hybrid_index()` 只在两个地方被调用：

1. `RagService.index_document()`
2. `RagService.delete_document_index()`

可问题在于，文档 CRUD API 并没有调用这两个方法，而是直接在后台任务里调用：

1. `index_document_chunks`
2. `delete_document_chunks`

结果就是：

1. SQLite 和 Chroma 会更新
2. 但进程内的 BM25 词法索引不一定会同步失效
3. 之后一段时间里，向量检索和词法检索可能看到的不是同一版知识库

这是当前代码层面客观存在的一个一致性风险。

如果老师问“索引有没有完全同步”，你最稳的答法是：

> 当前向量索引更新是有的，但 BM25 词法索引的进程内刷新还有进一步完善空间，这是我已经识别到的工程风险点。

---

## 15. Rerank：精排是怎么接进来的

## 15.1 模型加载

`rerank_service.py` 当前会：

1. 自动判断 `cuda` 还是 `cpu`
2. 根据配置选择模型
3. 优先使用本地已有模型文件
4. 否则从 HuggingFace cache 目录加载

### 15.2 输入格式

每个候选的 rerank 输入是：

```text
标题：<title>
内容：<preview>
```

而不是只给正文。

### 15.3 分数归一化

当前不是直接相信 raw logits，而是对一批候选做 min-max 归一化。

这么做的原因是：

1. 不同 batch 的 raw logits 不可直接横向比较
2. 当前 pipeline 更关心“这一批候选里谁更相关”

### 15.4 但你还要知道一个事实

`rerank_service.py` 里还保留了旧函数：

1. `build_retrieved_match()`
2. `rank_retrieved_matches()`

它们当前不在主链路里被调用。

所以如果老师问“最终 usable 是不是由 rerank_service 直接判断的”，你要答：

> 不是。当前 rerank 只负责提供精排信号，真正的 usable 判定已经收口到 `UnifiedEvidenceScorer`。

### 15.5 `RERANK_MIN_SCORE` 的现状

`RERANK_MIN_SCORE=0.56` 这个配置当前只出现在旧的 `build_retrieved_match()` 逻辑里。

由于这套旧匹配逻辑不在主链路中使用，所以：

> `RERANK_MIN_SCORE` 当前不是主链路里真正起决定作用的门槛。

这也是一个很容易被问住的点。

---

## 16. UnifiedEvidenceScorer：当前最关键的“可答性判定器”

这是第二层控制面，也是当前系统最重要的质量闸门。

## 16.1 为什么要有它

只靠检索和 rerank 还不够，因为：

1. 检索到相关文档不代表文档能直接回答问题
2. rerank 高不代表主题没有错配
3. 多类型 query 对“好证据”的定义不同

所以当前系统把这些信号都统一收口到一个 `EvidenceAssessment`。

### 16.2 Snapshot 里会构造什么

对每个候选，评分器会先生成一个 `CandidateSnapshot`，里面关键字段有：

1. `title`
2. `content`
3. `adaptive_score`
4. `query_tokens`
5. `title_tokens`
6. `content_tokens`
7. `topic_alignment`
8. `title_alignment`
9. `base_relevance`

### 16.3 `base_relevance` 的公式

当前只有两项：

```text
base_relevance
= 0.50 * adaptive_score
+ 0.50 * rerank_score
```

也就是说：

1. hybrid 粗排先验保留下来了
2. rerank 精排信号也被正式纳入
3. 不再用旧式 scattered rules 直接拍板

### 16.4 `topic_alignment` 的作用

它本质上是 query token 与标题/内容 token 的覆盖比。

作用是：

1. 防止文档只是顺带提到一个实体
2. 防止 rerank 因局部语义相近而放过主题错位文档

### 16.5 Judge 是怎么选的

即使 `needs_judge=True`，也不是对所有候选都跑。

当前规则：

1. 按 `base_relevance` 排序
2. 只选前面最可能的候选
3. 最多 `MAX_JUDGE_CANDIDATES = 3`
4. 候选 base relevance 太低就不送 Judge

这说明系统是在“用 Judge 做高价值复核”，不是盲目全量调用。

### 16.6 Judge 并发和超时

当前实现：

1. `asyncio.gather(..., return_exceptions=True)`
2. 每个 Judge 调用都包 `asyncio.wait_for(..., timeout=6.0)`

如果超时：

1. 该候选记为 `timed_out=True`
2. `usable=False`
3. flag 会带 `LLM_JUDGE_TIMEOUT`

### 16.7 当前权重

当前贡献项权重如下：

#### 基础相关性

| 信号 | 权重 |
| --- | --- |
| adaptive_signal | 0.50 |
| rerank_signal | 0.50 |

#### `topic_signal` 权重

| profile | 权重 |
| --- | --- |
| STRICT | 0.18 |
| MODERATE | 0.14 |
| LOOSE | 0.10 |

#### `judge_signal` 权重

| profile | 权重 |
| --- | --- |
| STRICT | 0.24 |
| MODERATE | 0.18 |
| LOOSE | 0.00 |

### 16.8 当前门槛

#### `MIN_BASE_RELEVANCE`

| profile | 值 |
| --- | --- |
| STRICT | 0.48 |
| MODERATE | 0.42 |
| LOOSE | 0.36 |

#### `MIN_TOPIC_ALIGNMENT`

| profile | 值 |
| --- | --- |
| STRICT | 0.20 |
| MODERATE | 0.08 |
| LOOSE | 0.05 |

#### `MIN_FINAL_SCORE`

| profile | 值 |
| --- | --- |
| STRICT | 0.58 |
| MODERATE | 0.45 |
| LOOSE | 0.46 |

### 16.9 `usable` 的真实判定逻辑

一个候选会被判不可用，主要有几种情况：

1. `topic_alignment` 太低
2. `final_score` 太低
3. Judge 被调用且未通过
4. 在 `STRICT + ATOMIC_SPAN` 场景下，没有 `direct_evidence`

### 16.10 `direct_evidence` 和 `supports_extractive`

这是两个非常容易被老师问的概念。

#### `direct_evidence`

意思是：

> 文档里有没有足够直接的证据支持回答，而不是只有背景信息。

#### `supports_extractive`

意思是：

> 这个问题适不适合用抽句方式给出简洁答案。

所以二者不是一回事：

1. 一个候选可以 usable，但不适合抽取
2. 一个候选也可以 topic 对，但 direct evidence 不够，最终被拒掉

### 16.11 `trace_summary`

评分器还会输出结构化 trace 日志，包含：

1. `doc_id`
2. `title`
3. `adaptive_score`
4. `base_relevance`
5. `topic_match`
6. `judge_latency_ms`
7. `flags`
8. `usable`
9. `final_score`
10. `judge_status`

这是典型的工程型可观测设计。

### 16.12 一个必须提前知道的风险：Judge 异常时是“宽松 fallback”

`evidence_judge.py` 当前如果模型调用异常，会返回：

1. `core_topic_match=True`
2. `contains_direct_evidence=True`
3. `answerable=True`
4. `reason='judge_fallback'`

这意味着：

1. Judge 不是 fail-closed
2. 而是 availability-first 的 fail-open fallback

你可以把这个设计解释成：

> 当前实现更偏向保证系统可用，而不是把 Judge 失败直接当成硬拒答；但这确实带来一定的误放风险，是后续可继续收紧的地方。

---

## 17. AnswerModeRouter：回答方式不是固定的

这是第三层控制面。

## 17.1 四种模式

1. `NO_CONTEXT`
2. `EXTRACTIVE`
3. `STRUCTURED`
4. `GENERATIVE`

### 17.2 路由原则

#### 直接 `NO_CONTEXT`

没有 usable evidence 就拒答。

#### 直接 `GENERATIVE`

以下情况会直接 generative：

1. `SUMMARY`
2. `OVERVIEW`
3. `FULL_DOCUMENT`
4. 多文档 `RELATION / REASON`
5. 多信息 `FACTOID`

#### `STRUCTURED`

主要是：

1. `REASON`
2. 想要短答，但又没有足够稳定的 extractive 证据

#### `EXTRACTIVE`

需要同时满足：

1. `wants_short_answer=True`
2. `ATOMIC_SPAN`
3. 主候选 `direct_evidence=True`
4. 主候选 `supports_extractive=True`
5. 主候选和第二名有足够差距

### 17.3 `PRIMARY_EVIDENCE_GAP`

当前阈值：

```text
0.12
```

意思是：

> 如果第一名只比第二名高一点点，系统不会轻易做 extractive 单句回答，因为主证据不够“独占”。

---

## 18. Answer Generators：最后怎么生成答案

## 18.1 GeneratorFactory 的职责

它负责三件事：

1. 选具体生成器
2. 统一构造 sources
3. 在失败时优雅降级

当前降级链是：

```text
EXTRACTIVE -> STRUCTURED -> GENERATIVE -> NO_CONTEXT
```

### 18.2 ExtractiveGenerator

核心策略：

1. 优先用 Judge 产出的 `evidence_quote`
2. 如果没有，就在 chunk 内做窗口选择
3. 再做句子级打分
4. 必要时尝试拼接 top-2 句
5. 如果抽不稳，就返回 `None`，交给上层降级

#### 你要知道它为什么不是纯规则抽句

因为当前设计不是想靠复杂规则把所有答案都抽出来，而是：

1. 能抽准就抽
2. 抽不准就交给结构化生成

这是一种工程上更稳妥的策略。

#### 当前一个真实现象

单测里已经暴露出一个问题：

对礼貌问法“请问一下，Shor主要解决什么问题吗？”，  
当前抽取结果会把句首“该算法能在多项式时间内”裁掉，只剩下：

```text
解决大整数质因数分解问题，而经典计算机对此问题需要指数时间。
```

这说明 `_trim_to_relevant_start()` 当前存在“裁剪过猛”的现象。

这是个很好的“当前不足”例子。

### 18.3 StructuredGenerator

它适合：

1. 直接抽句不稳
2. 但问题又不需要长篇综述

关键行为：

1. 原子事实题且已有 `answer_brief` 时，直接用它
2. 多信息问题会扩大上下文上限
3. `RELATION / REASON / 多信息 FACTOID` 更倾向用 `full_content`

### 18.4 GenerativeGenerator

它适合：

1. `SUMMARY`
2. `OVERVIEW`
3. 多文档综合
4. 纠错类任务

关键点：

1. `SUMMARY / OVERVIEW` 会切到 `full_content`
2. correction mode 也会用 `full_content`
3. 生成后还会做后处理，过滤元话语、低相关延伸句、无支撑句

### 18.5 生成后不是彻底放任

虽然当前没有额外的生成后审核层，但也不是完全放开。

生成后当前会做：

1. 去除通用前缀
2. 切句
3. 和支持窗口做 support ratio 对齐
4. 删掉元话语句、拒答模板句、低相关扩展句

也就是说，当前稳定性不是靠“再找一个模型审答案”，而是：

1. 前面证据路由更严格
2. 后面生成后做轻量约束清洗

---

## 19. `chunk_text` 和 `full_content` 是怎么同时存在的

这是当前系统设计里很重要、也很容易被老师问到的一个点。

`RagService._build_retrieved_candidates()` 会同时保留：

1. `chunk_text`
2. `full_content`
3. `doc.page_content` 仍然是 chunk 内容

这意味着当前系统不是“一旦命中 chunk 就把全文替换掉”，而是：

1. 评分和抽取优先看 chunk
2. 全文只在确实需要综合时才用

### 19.1 为什么这样设计

因为：

1. chunk 更适合事实级定位
2. 全文更适合概况、综合、多段关系
3. 如果一上来就把所有环节都切到全文，短事实问题会被稀释

### 19.2 当前哪些环节更偏向全文

1. `SUMMARY / OVERVIEW` 的生成
2. correction query
3. 部分 relation/reason 场景的结构化/生成式回答
4. `RELATION` 在 Judge 时优先看全文

---

## 20. 前端链路：用户到底经历了什么

## 20.1 问答发送

`frontend/src/api/chat.ts` 实际发送的是：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<问题>"
    }
  ]
}
```

前端不会传 `doc_id`，也不会传 mode。

### 20.2 用户看到什么

`ChatPanel.vue` 会显示：

1. 回答文本
2. sources 列表
3. 每个 source 的 rank
4. 每个 source 的 confidence

点 source 后，前端会调用：

```ts
store.loadDocument(source.doc_id)
```

也就是说：

> 当前 sources 不是只做展示，而是可点击回跳到原文档。

### 20.3 这套设计的意义

你可以这样讲：

1. 后端把可解释性落在 `sources`
2. 前端把可解释性落成“可追溯原文”

这比只给一个模型答案更像知识库系统，而不是普通聊天机器人。

---

## 21. 当前缓存与一致性问题

这部分是老师很容易追问的工程性问题。

## 21.1 文档列表缓存

`DocumentService` 会缓存文档列表，更新文档时会删掉 `documents_list`。

这块相对完整。

### 21.2 RAG 响应缓存

`RagService` 会按 query 缓存最终回答 1 小时。

但当前文档更新和删除时，并没有对应清理这些 RAG 回答缓存。

这意味着：

1. 文档内容已经变了
2. 但用户同样的问题可能还会命中旧回答缓存
3. 直到缓存自然过期

这是当前真实存在的陈旧风险。

### 21.3 索引更新和查询的时序问题

由于文档索引是后台任务，所以：

1. 用户保存文档
2. 索引任务异步跑
3. 这段时间内马上提问，可能还查到旧索引

这属于典型的最终一致性窗口。

---

## 22. 当前代码和测试、文档之间的几个“口径冲突”

这是你最需要提前知道的部分之一。

## 22.1 冲突一：关系型 query 是否启用 Judge

当前代码：

1. `RELATION` 一般是 `MULTI_SPAN`
2. `MULTI_SPAN` 当前不启用 Judge
3. 所以 relation 问题的 `needs_judge=False`

但测试 `test_builds_relation_intent_for_relation_query` 还期待：

```text
intent.needs_judge == True
```

这说明：

1. 测试口径落后于当前代码
2. 你答辩时必须以代码运行时行为为准

### 22.2 冲突二：抽取器对礼貌问法的裁剪

测试里期望完整句：

```text
该算法能在多项式时间内解决大整数质因数分解问题...
```

当前运行结果却会变成：

```text
解决大整数质因数分解问题...
```

这说明：

1. `ExtractiveGenerator` 当前确实存在句首裁剪偏激进的问题
2. 不是“完全没有问题”的系统

### 22.3 冲突三：旧文档里“Judge 主要覆盖关系型 query”的说法

这在当前代码层面并不精确。

更准确的说法应该是：

> 当前 Judge 主要覆盖 `STRICT + atomic-span` 的高风险 query，例如短实体、地点、属性型问题；关系型、多段综合型问题主要靠统一证据评分和回答路由。

---

## 23. 当前单测运行结果

我实际运行的命令是：

```bash
/root/Velo/.venv/bin/python -m unittest discover -s /root/Velo/backend/tests
```

结果：

1. 共运行 33 个测试
2. 失败 2 个

失败项分别是：

1. `test_extracts_original_sentence_for_polite_factoid_query`
2. `test_builds_relation_intent_for_relation_query`

### 23.1 这两个失败意味着什么

第一个失败说明：

1. 抽取器对礼貌问法仍有边缘截断问题

第二个失败说明：

1. 测试对 relation query 的 Judge 预期已经和当前代码分叉

### 23.2 你怎么对老师说

如果老师问“你们有没有测试”，你不要只说“有”。

你应该说：

> 我们有单测，而且我核对过当前测试现状。大部分核心流程都有覆盖，但也有两个失败用例恰好暴露出当前实现和旧测试预期的分叉：一个是礼貌问法抽取边界，一个是 relation query 是否启用 Judge 的口径变化。这两个点我都已经明确识别出来了。

这样老师会觉得你是在真正维护系统，而不是背答案。

---

## 24. 当前测试和脚本覆盖了什么

### 24.1 `test_hybrid_retrieval.py`

覆盖：

1. identifier query 会提高 lexical weight
2. BM25-only 文档能被 hybrid 候选池补回来

### 24.2 `test_rag_pipeline_execution.py`

覆盖：

1. 抽取式回答
2. 抽取失败后的降级
3. polite query
4. list-style 文档抽取
5. reason query
6. `RagService.rag_qa` 的返回契约

### 24.3 `test_rag_pipeline_routing.py`

覆盖：

1. QueryIntentBuilder 的意图分类
2. AnswerModeRouter 的路由逻辑

### 24.4 `test_unified_evidence_scorer.py`

覆盖：

1. 不 silent drop candidate
2. Judge 并发执行
3. Judge 超时不阻塞整批
4. trace log 结构

### 24.5 `run_rag_probes.py`

这是在线探针脚本，不是单测。

它覆盖一些实际问题，比如：

1. `Shor算法能解决什么问题`
2. `青苗法是什么`
3. `沈阳`
4. `韩国与新加坡的关系`
5. `新加坡的风景`

这些 probe 的价值是：

1. 验证端到端结果
2. 验证拒答边界
3. 验证 sources 标题

---

## 25. 公开 benchmark 结果：你能拿得出手的量化依据

## 25.1 对比对象

公开 benchmark 里对比的是三条链路：

1. 主流方案 A：Dense + Rerank
2. 主流方案 B：Hybrid + RRF + Rerank
3. 你的方案：Adaptive Hybrid + Rerank

### 25.2 数据集

1. `DuRetrieval`
2. `T2Retrieval`

每个数据集当前正式评测 query 数都是 1200。

### 25.3 关键结果

#### DuRetrieval

| 方案 | Hit@1 | nDCG@10 | 平均时延(ms) |
| --- | --- | --- | --- |
| 主流 A | 0.2333 | 0.1311 | 188.55 |
| 主流 B | 0.8408 | 0.6616 | 289.44 |
| 你的方案 | 0.8558 | 0.6858 | 373.12 |

#### T2Retrieval

| 方案 | Hit@1 | nDCG@10 | 平均时延(ms) |
| --- | --- | --- | --- |
| 主流 A | 0.2667 | 0.1499 | 247.81 |
| 主流 B | 0.8450 | 0.6904 | 375.00 |
| 你的方案 | 0.8475 | 0.6984 | 451.72 |

### 25.4 你该怎么解释这组结果

最稳的解释是：

1. 相对单路 dense baseline，混合检索本身提升巨大
2. 在已经较强的主流 hybrid + RRF + rerank 基线上，你的方案仍然有稳定正提升
3. 代价是时延变高，因为候选构造和多信号融合更复杂

千万不要说成“我们比任何主流方法都大幅领先”，那不准确。  
更准确的是：

> 我的改进不是把弱基线刷高，而是在一个已经较强的 hybrid 基线之上继续拿到稳定正收益。

---

## 26. 消融实验：到底是哪几个模块在贡献收益

当前最重要的消融结论是：

1. 自适应权重是主要贡献源之一
2. 混合候选池是最强贡献项之一
3. 覆盖率奖励有小幅正贡献
4. 标识符约束在 DuRetrieval 上没有显示稳定正收益

### 26.1 核心数字

完整方案相对各去模块版本的差值：

| 对比项 | Hit@1 差值 | nDCG@10 差值 |
| --- | --- | --- |
| 相对主流 B | +0.0150 | +0.0242 |
| 去掉自适应权重 | +0.0167 | +0.0300 |
| 去掉覆盖率奖励 | +0.0017 | +0.0038 |
| 去掉标识符约束 | -0.0008 | -0.0017 |
| 去掉混合候选池 | +0.0217 | +0.0535 |

### 26.2 你该怎么说

标准答法：

> 当前主要收益来自查询自适应权重和混合候选池；覆盖率奖励是细化项；标识符约束从工程直觉上合理，但在当前公开消融上还没有表现出稳定正收益。

这样最稳。

---

## 27. 系统级评测：不是只看检索指标

## 27.1 本地知识库 vs LlamaIndex

当前 `llamaindex_rag_eval/outputs/rag_comparison_summary.json` 中：

### internal_rag

1. `faithfulness = 1.0`
2. `answer_relevancy = 0.8112`
3. `context_precision = 1.0`
4. `accuracy = 0.3`
5. `retrieval_hit_rate_at_3 = 1.0`
6. `refusal_f1 = 1.0`

### llamaindex

1. `faithfulness = 0.85`
2. `answer_relevancy = 0.8545`
3. `context_precision = 1.0`
4. `accuracy = 0.2`
5. `retrieval_hit_rate_at_3 = 1.0`
6. `refusal_f1 = 0.9091`

### 27.2 你怎么解读

1. 本地小知识库里，两套系统都能找到正确文档
2. 你的系统在 refusal 边界上更稳
3. 但 accuracy 也并不高，说明回答生成本身仍有改进空间

这是很诚实、很专业的说法。

## 27.3 RGB 数据集

当前 `rgb_rag_comparison_summary.json`：

### internal_rag

1. `faithfulness = 0.92`
2. `answer_relevancy = 0.8113`
3. `context_precision = 0.93`
4. `accuracy = 0.8033`
5. `retrieval_hit_rate_at_3 = 0.8533`

### llamaindex

1. `faithfulness = 0.8517`
2. `answer_relevancy = 0.6429`
3. `context_precision = 0.7333`
4. `accuracy = 0.59`
5. `retrieval_hit_rate_at_3 = 0.6033`

这组结果说明：

1. 你的系统在公开数据上的上下文选择和最终回答都更稳
2. 不是只在本地 demo 上有效

## 27.4 CRUD-RAG

当前 `crud_rag_comparison_summary.json`：

### internal_rag

1. `faithfulness = 0.9201`
2. `answer_relevancy = 0.8494`
3. `context_precision = 0.9653`
4. `context_recall = 0.9333`
5. `noise_robustness = 0.6006`
6. `negative_rejection = 1.0`
7. `information_integration = 0.0`

### llamaindex

1. `faithfulness = 0.9286`
2. `answer_relevancy = 0.7869`
3. `context_precision = 0.8785`
4. `context_recall = 0.8750`
5. `noise_robustness = 0.2834`
6. `negative_rejection = 1.0`
7. `information_integration = 0.0`

### 27.5 你怎么诚实解释

1. 你的系统在噪声鲁棒性上明显更强
2. context precision / recall 也更好
3. 但 `information_integration = 0.0`，说明跨文档整合类任务仍然是薄弱点

这个“承认弱点”的姿态非常重要。

---

## 28. 当前系统的真实优点

你可以总结成下面 8 条：

1. 主链路已经模块化，不再是混杂逻辑堆在一个大函数里。
2. QueryIntentBuilder 把“怎么搜、怎么防御、证据要求是什么”前置了。
3. 检索不是单路 dense，而是 dense + BM25 + 自适应融合。
4. rerank 没有被神化，而是作为证据信号的一部分进入统一评分。
5. 统一证据评分器把 usable 判定结构化了。
6. 回答模式不是固定生成，而是抽取/结构化/生成/拒答分流。
7. 前端引用是可点回原文的，系统可解释性较强。
8. 有公开 benchmark、系统级评测和消融实验支撑，不是单一 demo。

---

## 29. 当前系统的真实不足

这部分你一定要提前会说。

### 29.1 单轮而非多轮

虽然 schema 是 message list，但当前只取最后一条消息。

### 29.2 RAG 缓存失效机制不完整

文档更新后，旧 query 的回答缓存不会立即失效。

### 29.3 BM25 词法索引刷新不够严谨

CRUD API 直接调向量索引函数，没有显式调用 `invalidate_hybrid_index()`。

### 29.4 Judge 失败时偏宽松

模型异常时 `judge_rag_document()` 会 fail-open fallback。

### 29.5 `retrieval_depth` 当前默认不真正控制 fan-out

因为默认 50/50/30 比它大得多。

### 29.6 `RAG_RESULT_LIMIT` 当前未接入主链路

虽然配置存在，但当前 sources 数量更多由生成器决定：

1. extractive 通常 1 条 source
2. structured / generative 最多 5 条 source

### 29.7 `summary` / `tags` 预留未用

模型层有字段，但当前 RAG 不使用它们做检索增强。

### 29.8 Chroma 类有弃用警告

当前还在用 `langchain_community.vectorstores.Chroma`，运行时已有 deprecation warning。

---

## 30. 你在答辩中最容易被问的 25 个问题

下面这 25 个问题，你最好都能开口答。

### 30.1 你这个系统的主入口在哪？

答：HTTP 入口是 `POST /api/v1/agent/chat`，真正的业务主入口是 `RagService.rag_qa`。

### 30.2 你当前是多轮对话吗？

答：不是严格意义上的多轮。接口接收 message 列表，但当前实现只取最后一条用户消息，所以运行时能力是单轮知识库问答。

### 30.3 为什么不是直接“向量检索 + LLM 回答”？

答：因为那样很容易出现误召回、主题错配和证据不足仍然强行生成的问题。我把控制点前移到了 query 画像、统一证据评分和回答模式路由。

### 30.4 为什么要加 BM25？

答：dense 擅长语义近似，但对短标题、专有名词、编号、年份不够稳；BM25 正好补这类词法锚定，所以两者是互补关系。

### 30.5 为什么 BM25 用文档级，不用 chunk 级？

答：因为向量侧已经是 chunk 级召回，BM25 用文档级能更好利用标题和全文词汇做锚定，不和 dense 完全重复。

### 30.6 你最核心的算法改进是什么？

答：是 query 自适应的 hybrid 融合和混合候选池构造。前者根据 query 特征动态调 dense/lexical 权重，后者避免过早丢掉互补候选。

### 30.7 公式能说出来吗？

答：能。`adaptive_score = dense_weight * normalized_dense + lexical_weight * normalized_bm25 + 0.26 * normalized_rrf + 0.08 * coverage + 0.03 * identifier_overlap`。

### 30.8 你这个是 learning-to-rank 吗？

答：不是训练型 learning-to-rank。当前是规则驱动、可解释、可复现的多信号融合。

### 30.9 rerank 在系统里是什么角色？

答：它是候选精排器，不是最终裁决者。最终 usable 判定在 `UnifiedEvidenceScorer`。

### 30.10 什么情况下会触发 Judge？

答：当前主要是 `STRICT + atomic-span` 的高风险 query，例如短实体、地点、属性型问题。relation 这类多段综合型问题当前默认不走 Judge。

### 30.11 为什么 relation 不走 Judge？

答：因为 relation 多数是 `MULTI_SPAN`，Judge 更适合做原子证据复核；多段综合型问题强行让 Judge 逐个卡，容易误杀，所以当前更多靠统一证据评分和回答模式路由。

### 30.12 你的拒答是怎么实现的？

答：不是靠一条 prompt 模板，而是靠 `UnifiedEvidenceScorer` 的 usable 判定和 `AnswerModeRouter` 的 `NO_CONTEXT` 分支。

### 30.13 你怎么保证不是“提到关键词就乱答”？

答：一方面 hybrid 阶段有 coverage 和 identifier 约束；另一方面证据评分里有 `topic_alignment`，高风险场景还会有 Judge 复核。

### 30.14 为什么还要区分 `direct_evidence` 和 `supports_extractive`？

答：因为“文档能回答”不等于“适合直接抽句回答”。有些证据可用，但需要组织后再答。

### 30.15 什么时候走抽取式回答？

答：短答案、原子事实、主证据足够清晰、第一名明显强于第二名、并且候选支持 extractive 时才走。

### 30.16 什么时候走生成式回答？

答：概况类、总结类、需要全文综合的场景，以及多文档关系/原因问题会更倾向生成式。

### 30.17 你现在 demo 知识库很小，能说明什么？

答：小知识库适合做可视化演示和端到端验证；为了证明方法不只在 demo 上有效，我还补了公开 retrieval benchmark、RGB 和 CRUD-RAG 评测。

### 30.18 你的系统最明显的量化提升是什么？

答：在 `DuRetrieval` 上，相对主流 hybrid + RRF + rerank 基线，`Hit@1` 从 `0.8408` 提到 `0.8558`，`nDCG@10` 从 `0.6616` 提到 `0.6858`。

### 30.19 你的系统时延变高了，值得吗？

答：值得，但需要明确场景。我的系统不是单纯追求最低时延，而是追求更高的可答性边界和更稳的候选质量。当前 tradeoff 是更高质量换更高时延。

### 30.20 你最诚实的不足是什么？

答：当前还不是完整多轮 RAG；缓存失效机制还不够完善；BM25 词法索引刷新也还有工程改进空间；另外跨文档深度信息整合仍然是弱项。

### 30.21 为什么不把所有问题都交给 Judge？

答：因为那会显著增加时延，而且 Judge 也不是万能的；很多问题真正需要的是更好的检索、评分和模式路由，而不是再套一层判断模型。

### 30.22 你有没有测试？

答：有。当前后端有 33 个单测，覆盖 hybrid 检索、意图构建、统一证据评分、回答路由和端到端返回契约。我也核对过当前有 2 个失败用例，分别是礼貌问法抽取边界和 relation query 的旧测试预期。

### 30.23 为什么要保留 source 引用？

答：因为这是知识库系统最重要的可解释性出口。后端返回 `title/doc_id/rank/confidence`，前端还能点回原文，不是只给黑盒答案。

### 30.24 你有没有额外的查询重写或生成后审核模块？

答：当前主链路不引入额外的历史增强模块。现在运行面的重点是 QueryIntentBuilder、自适应 hybrid、统一证据评分和回答模式路由。

### 30.25 如果你接下来继续做，会优先做什么？

答：我会先补三件事：  
第一，完善 RAG 响应缓存和 BM25 词法索引的失效机制；  
第二，把 `retrieval_depth` 真正接成有效运行参数；  
第三，继续改进多文档信息整合能力，而不是只强化单文档抽取。

---

## 31. 最后给你一个“稳答模板”

如果老师抛来一个你没完全准备到的问题，你可以先用下面这个模板稳住节奏：

> 当前这部分我先按“代码实际怎么做”来回答。  
> 在我的实现里，这个问题主要落在 `<模块名>` 这一层。  
> 它的输入是 `<输入>`，输出是 `<输出>`，当前关键判断依据是 `<阈值/规则/公式>`。  
> 如果从效果上看，它解决的是 `<问题>`，但当前也还有 `<不足>` 这个边界。

这个模板的好处是：

1. 先把话题拉回你熟悉的代码层
2. 再用输入/输出/规则/边界四步结构回答
3. 即使问题很尖锐，你也不容易乱

---

## 32. 你在答辩时绝对不要讲错的五件事

1. 不要说当前是完整多轮记忆型 RAG，对代码不准确。
2. 不要说 relation query 默认一定会跑 Judge，对代码不准确。
3. 不要说 `retrieval_depth` 当前一定会改变召回上限，在默认配置下不准确。
4. 不要说标识符约束是公开实验里的核心收益来源，不准确。
5. 不要把旧测试或旧文档里的口径直接当成当前运行事实。

---

## 33. 这一套系统，你最后要记住的“标准定义”

如果老师最后问你：“一句话总结你的系统是什么？”

你最稳的回答是：

> 这是一套以本地知识库为基础、以自适应 hybrid 检索为核心、以统一证据评分和显式回答路由为质量保障的 RAG 系统。它的重点不只是把文档找出来，更重要的是判断证据是否足够、决定应该怎么回答，以及在证据不足时稳定拒答。

这句话和当前代码是对齐的。
