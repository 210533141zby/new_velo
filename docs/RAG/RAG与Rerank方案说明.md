# RAG 与 Rerank 方案说明

## 1. 这份文档的定位

这份文档只负责讲当前 backend 实际运行的 RAG 技术方案。

重点回答四件事：

1. 当前主链路到底由哪些模块组成。
2. 检索、融合、精排、评分、路由、生成分别谁负责。
3. 当前模型、阈值和公式是什么。
4. 哪些旧能力已经从代码库移除，不再属于当前系统。

如果你想看：

1. 演进过程和失败闭环，请看 `RAG主链路优化与问题闭环报告.md`
2. 启动、自检、探针和联调，请看 `RAG主链路执行流程与联调手册.md`

## 2. 当前主链路总览

当前实际运行的主链路是：

```text
原始 query
-> query 去噪 / 关键词视图
-> QueryIntentBuilder
-> 单路向量召回
-> 文档级 BM25 补召回
-> adaptive hybrid 融合
-> cross-encoder rerank
-> UnifiedEvidenceScorer
-> AnswerModeRouter
-> Extractive / Structured / Generative
-> sources 排序与返回
```

这条链路有三个关键变化：

1. 旧的 `AgentService + universal defense + 多处 if-else` 已经收口。
2. 当前控制面已经集中到：
   - `QueryIntentBuilder`
   - `UnifiedEvidenceScorer`
   - `AnswerModeRouter`
3. 早期探索过的额外检索增强和生成后审核逻辑，已经不属于当前代码库与运行面。

## 3. 当前模块分工

### 3.1 `rag_service.py`

路径：

- `backend/app/services/rag/rag_service.py`

职责：

1. 命中缓存时直接返回。
2. 系统信息问题直接短路，不走知识库。
3. 调 `QueryIntentBuilder.build()` 生成检索与防御参数。
4. 调向量召回、BM25、hybrid 融合和 rerank。
5. 构造同时包含 `chunk_text` 和 `full_content` 的候选对象，其中 `page_content` 保持为 chunk 文本。
6. 调 `UnifiedEvidenceScorer` 做统一评分。
7. 调 `AnswerModeRouter` 选回答模式。
8. 调 `GeneratorFactory` 执行生成和优雅降级。

### 3.2 `query_intent_builder.py`

路径：

- `backend/app/services/rag/query_intent_builder.py`

职责：

1. query 去噪
2. polite prefix 清洗
3. 句尾语气词清洗
4. `keyword_query` 构造
5. `intent_type` 判定
6. `retrieval_depth` 生成
7. `defense_profile` 生成
8. `evidence_requirement` 生成
9. `needs_judge` 判定

### 3.3 `hybrid_search.py`

路径：

- `backend/app/services/rag/hybrid_search.py`

职责：

1. 文档级 BM25 索引
2. query profile
3. lexical / dense 自适应权重
4. `dense + bm25 + rrf + coverage + identifier` 融合
5. 混合候选池构造

### 3.4 `rerank_service.py`

路径：

- `backend/app/services/rag/rerank_service.py`

职责：

1. 加载 cross-encoder
2. 构造 rerank 输入
3. 返回按批次 min-max 归一化后的 `rerank_score`

注意：

1. `rerank_service.py` 里保留了旧版 `build_retrieved_match / rank_retrieved_matches` 逻辑。
2. 当前默认主链路不再用它来决定 `usable`。
3. 当前实际可用性判断已经转移到 `evidence_scorer.py`。

### 3.5 `evidence_scorer.py`

路径：

- `backend/app/services/rag/evidence_scorer.py`

职责：

1. 计算基础相关性
2. 计算 `topic_alignment`
3. 选择性触发 `LLM Judge`
4. 计算 `direct_evidence`
5. 计算 `supports_extractive`
6. 决定 `usable / reject_reason / flags`
7. 输出结构化 trace 日志

### 3.6 `answer_mode_router.py`

路径：

- `backend/app/services/rag/answer_mode_router.py`

职责：

1. 在 `NO_CONTEXT / EXTRACTIVE / STRUCTURED / GENERATIVE` 之间路由
2. 统一回答模式判断树
3. 固定 sources 选择范围

### 3.7 `answer_generators.py`

路径：

- `backend/app/services/rag/answer_generators.py`

职责：

1. 启发式抽取原文句子
2. 结构化短答
3. 生成式回答
4. 失败后的优雅降级
5. sources 构造

### 3.8 `prompt_templates.py`

路径：

- `backend/app/services/rag/prompt_templates.py`

职责：

1. Judge Prompt
2. Structured Prompt
3. Generative Prompt
4. 系统信息回答
5. No-context 回答

### 3.9 `vector_index_service.py`

路径：

- `backend/app/services/rag/vector_index_service.py`

职责：

1. 文档切块
2. 提取标题和首段中的核心实体，并注入 `【核心主题：...】` 前缀
3. metadata 注入
4. Chroma 写入与删除
5. Embedding 客户端构造

## 4. 当前模型与组件选型

### 4.1 Embedding

配置来源：

- `EMBEDDING_PROVIDER = ollama`
- `EMBEDDING_BASE_URL = http://127.0.0.1:11434/v1`
- `EMBEDDING_MODEL = nomic-embed-text:latest`

当前运行态：

1. Provider: `ollama`
2. SDK 适配类: `OpenAIEmbeddings`
3. 模型名: `nomic-embed-text:latest`
4. 向量维度: `768`

### 4.2 Chat / Judge 模型

配置来源：

- `CHAT_MODEL = qwen2.5:7b-instruct`
- `LLM_PROVIDER = ollama`
- `LLM_BASE_URL = http://127.0.0.1:11434/v1`
- `RAG_JUDGE_MAX_TOKENS = 220`

当前运行态：

1. 普通回答模型：`qwen2.5:7b-instruct`
2. Judge 模型：`qwen2.5:7b-instruct`
3. 普通回答温度：`0.3`
4. Judge 温度：`0.0`
5. 模型 timeout：`120s`
6. `max_retries = 1`

### 4.3 Rerank 模型

配置来源：

- `RERANK_ENABLED = True`
- `RERANK_MODEL = BAAI/bge-reranker-v2-m3`
- `RERANK_CPU_MODEL = BAAI/bge-reranker-base`
- `RERANK_GPU_MODEL = BAAI/bge-reranker-v2-m3`
- `RERANK_DEVICE = auto`
- `RERANK_BATCH_SIZE = 8`
- `RERANK_MAX_LENGTH = 512`
- `RERANK_MAX_INPUT_CHARS = 1400`
- `RERANK_MIN_SCORE = 0.56`

当前运行态：

1. `device = cuda`
2. 实际解析模型：`BAAI/bge-reranker-v2-m3`

## 5. 技术方案的关键部分

### 5.1 检索前

当前默认主路径只做轻量、稳定的 query 处理：

1. 去掉口语前缀
2. 去掉句尾语气词
3. 生成 `normalized_query`
4. 生成 `keyword_query`
5. 判定 `intent_type`
6. 判定 `retrieval_depth`
7. 判定 `defense_profile`
8. 判定 `evidence_requirement`
9. 判定 `needs_judge`

检索前不再存在额外的历史检索增强分支。

### 5.2 检索中

当前不是简单把 dense 和 BM25 相加，而是：

1. 先做 Chroma 向量召回
2. 再做文档级 BM25
3. 再做 query-adaptive 融合
4. 再构造混合候选池

自适应 query profile 规则：

```text
初始 lexical_weight = 0.36
如果 query 含 identifier，+0.18
如果 token 数 <= 4，+0.14
如果 token 数 >= 10，-0.08
如果存在明显高 idf 稀有词，+0.10
最后裁剪到 [0.24, 0.78]
dense_weight = 1 - lexical_weight
```

adaptive score 公式：

```text
adaptive_score
= dense_weight * normalized_dense
+ lexical_weight * normalized_bm25
+ 0.26 * normalized_rrf
+ 0.08 * coverage
+ 0.03 * identifier_overlap
```

RRF 公式：

```text
RRF(doc) = Σ 1 / (60 + rank_i(doc))
```

当前候选池不是单一排序直取 top-k，而是：

1. 先取 `adaptive_rank` 高位
2. 再补 `rrf_rank` 高位
3. 再补 `adaptive_rank` 剩余部分
4. 不够时再补 `vector_doc_ids`
5. 最后补 `bm25_doc_ids`

### 5.3 Rerank

当前 rerank 只提供 cross-encoder 深度相关性分数，不直接决定最终 `usable`。

构造输入：

```text
标题：{title}
内容：{preview[:1400]}
```

输出处理：

1. 先拿 raw cross-encoder 输出
2. 再按当前 query 的 candidate batch 做 min-max 归一化
3. 得到 `rerank_score`

### 5.4 统一评分

当前默认主路径里，真正决定文档是否可用的是 `UnifiedEvidenceScorer`。

基础信号权重：

```text
adaptive_signal = 0.50
rerank_signal = 0.50
```

附加信号：

```text
topic_signal:
STRICT = 0.18
MODERATE = 0.14
LOOSE = 0.10

judge_signal:
STRICT = 0.24
MODERATE = 0.18
LOOSE = 0.0
```

阈值：

```text
MIN_BASE_RELEVANCE:
STRICT = 0.48
MODERATE = 0.42
LOOSE = 0.36

MIN_TOPIC_ALIGNMENT:
STRICT = 0.20
MODERATE = 0.08
LOOSE = 0.05

MIN_FINAL_SCORE:
STRICT = 0.58
MODERATE = 0.45
LOOSE = 0.46

WEAK_EVIDENCE_THRESHOLD = 0.34
MAX_JUDGE_CANDIDATES = 3
```

补充说明：

1. `MIN_BASE_RELEVANCE` 现在只保留为观测 flag，不再单独作为拒答分支。
2. `RELATION` 类型在 Judge 阶段会优先读取 `full_content`，减少跨段关系问题被单 chunk 误杀。
3. 无 Judge 场景下，`direct_evidence` 的兜底条件为 `topic_alignment >= 0.35 and (title_alignment >= 0.20 or base_relevance >= 0.52)`。

Judge 触发逻辑：

```text
needs_judge = defense_profile is STRICT
```

也就是：

1. 短实体 `LOOKUP`
2. 关系型 `RELATION`
3. `X 的 Y` 属性型 query

会进入 Judge 路径。

### 5.5 回答模式

回答模式只剩四种：

1. `NO_CONTEXT`
2. `EXTRACTIVE`
3. `STRUCTURED`
4. `GENERATIVE`

路由原则：

1. 没有 usable 文档就 `NO_CONTEXT`
2. `SUMMARY / OVERVIEW` 和 `FULL_DOCUMENT` 需求直接走 `GENERATIVE`
3. 多文档 `RELATION / REASON` 偏向 `GENERATIVE`
4. 单文档 `REASON` 走 `STRUCTURED`
5. 单文档高置信直接证据偏向 `EXTRACTIVE`
6. 其他短答型问题走 `STRUCTURED`

### 5.6 抽取策略

当前抽取器没有用复杂 regex 去拆主谓宾，也没有接重型 NLP 解析。

它做的是：

1. 从文档正文切段落
2. 如果没有段落，就切句子窗口
3. 用 query token overlap 给窗口打分
4. 再给句子打分
5. 如果 top-2 句得分接近，会尝试拼接两句
6. 返回原文句子
7. 失败则优雅降级到 `STRUCTURED`

原因类问题额外加了因果句偏置：

```text
如果句子包含 “因为 / 原因 / 之所以”
则在抽取排序里额外 +0.8
```

### 5.7 当前已从代码库清理的历史逻辑

一些早期探索过的额外检索增强和生成后审核逻辑，已经不再保留在当前代码库中。

答辩和联调都不应再把这些历史逻辑描述成当前系统组成部分。

## 6. 当前最该固定的口径

当前最稳的表达是：

1. 核心算法贡献在 `adaptive hybrid + candidate pool + UnifiedEvidenceScorer`
2. `rerank` 是重要原子能力，但不再独占最终决策权
3. 当前主路径已经从旧的多层补丁收口为纯管道
4. `LLM Judge` 是选择性启用，不是全量启用
5. 生成层的稳定性依赖模式路由和优雅降级，不再依赖额外的生成后审核层
