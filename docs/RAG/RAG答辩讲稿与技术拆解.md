# RAG 答辩讲稿与技术拆解

## 1. 这份文档怎么用

这份文档只服务一件事：

`把当前 backend 真实运行的 RAG 主链路，讲成答辩老师能听懂、能追问、你也能稳住的版本。`

它只讲 RAG，不展开文本补全，也不展开前端。

当前口径严格对齐代码，不再沿用旧说法。答辩时不要再把下面这些说成“当前主路径”：

1. `AgentService`
2. `universal defense` 作为当前主控制面
3. 历史检索增强模块仍然是当前系统组成部分
4. 额外生成后审核层仍然挂在生成后

## 2. 当前最推荐的开场说法

如果老师让你先整体介绍，你可以直接这样讲：

“我这部分主要做的是 RAG 主链路优化。重点不是通用聊天，也不是前端界面，而是知识库问答里，如何把更相关的文档稳定找出来、排到前面，并在证据不足时明确拒答。当前主链路已经收口成三段：先由 `QueryIntentBuilder` 根据 query 决定检索深度、防御强度和证据要求；再由 `UnifiedEvidenceScorer` 统一融合召回、rerank、主题一致性和选择性 LLM Judge，判断文档到底能不能用；最后由 `AnswerModeRouter` 在拒答、抽取、结构化回答和生成式回答之间做模式路由。我的核心算法贡献主要在检索和粗排这一段，也就是向量召回与 BM25 召回之后、rerank 之前的自适应 hybrid 融合和候选池构造，同时在 rerank 之后没有完全丢掉检索先验，而是把证据可用性收口到统一评分器里。” 

这段话有四个好处：

1. 一开口就把范围锁死在 RAG。
2. 直接说清楚当前主链路已经重构过，不是老式大而全服务。
3. 老师能立刻知道你的核心贡献发生在检索侧，而不是只调 Prompt。
4. 也顺手把“拒答边界”交代了，显得系统意识更完整。

## 3. 30 秒版本

如果时间很短，直接背这一段：

“我的工作重点是把 RAG 从‘向量召回后直接让模型回答’升级成一条更稳定的知识库问答主链路。当前系统先做 query 意图构建，再做向量召回和文档级 BM25 的自适应 hybrid 融合，然后用 cross-encoder rerank 精排，接着通过统一证据评分器判断文档是否真的可用，最后再根据问题类型选择抽取式、结构化或生成式回答。我的核心改进主要在 hybrid 检索和候选构造这段，让编号、短实体、关系型问题更稳，同时通过选择性 LLM Judge 拦住主体错配问题。” 

## 4. 1 到 2 分钟版本

如果老师让你展开一点，你可以按下面这个顺序讲：

“我把整个 RAG 系统拆成四层。第一层是索引和查询准备，文档先按 Markdown 标题切分，再用 `chunk_size=1000`、`chunk_overlap=200` 做递归切块，写入 Chroma，并保留 `source` 和 `doc_id` 这两个核心元数据。第二层是检索，默认走单路向量召回，再用文档级 BM25 做词法补召回。第三层是我的核心算法，也就是查询自适应 hybrid 融合：根据 query 的长度、是否带编号、词项稀有度动态调整 dense 和 lexical 的权重，并把 dense、BM25、RRF、coverage、identifier 这些信号融合成候选排序，再构造混合候选池送给 rerank。第四层是可靠性和回答，`UnifiedEvidenceScorer` 会统一判断哪些文档可用，高风险 query 会选择性触发 LLM Judge；`AnswerModeRouter` 再决定是拒答、抽取、结构化短答还是生成式综述。这样系统不再是一个大模型自由发挥的黑盒，而是有显式意图、显式评分和显式路由的流水线。” 

## 5. 3 到 5 分钟完整版

这一版是最适合正式答辩的。

### 5.1 先讲问题背景

“如果只做最基础的 RAG，也就是切块、向量召回、拼 Prompt、交给大模型回答，系统很快会暴露三个问题。第一，编号、标题、短实体、关系型 query 的召回不稳。第二，即使召回到表面相关的文档，模型也可能把顺带提到的实体误当成真正主题。第三，系统缺少清晰的拒答边界，证据不够时还是可能顺着生成。因此我没有把重点放在 Prompt 微调上，而是把重心放在检索、证据评分和回答路由这三段主链路上。” 

### 5.2 再讲索引层

“当前索引层不是整篇文档直接入库，而是先按 Markdown 标题做结构切分，再做递归分块。具体参数是 `chunk_size=1000`、`chunk_overlap=200`。这样做是为了兼顾两件事：既让 embedding 保留局部语义单元，又不过度切碎上下文。每个 chunk 写入向量库时都会保留 `source` 和 `doc_id`，并且在入库前会注入一个轻量的 `【核心主题：...】` 语义锚点。后面检索命中 chunk 之后，评分和抽取继续用 chunk 文本，只有 `SUMMARY / OVERVIEW` 这类需要文档级综合的问题才会切到 `full_content` 做生成。” 

### 5.3 再讲查询意图构建

“query 进入系统之后，不会直接原封不动拿去搜，而是先经过 `QueryIntentBuilder`。这一步做的事很克制，但很关键。它先做口语前缀和句尾语气词清洗，然后构造 `normalized_query` 和 `keyword_query` 两个视图。接着用规则把 query 判成 `LOOKUP`、`FACTOID`、`RELATION`、`SUMMARY`、`OVERVIEW`、`REASON`、`LOCATION` 这些意图类型。不同意图会映射到不同的 `retrieval_depth`、`defense_profile` 和 `evidence_requirement`。例如，`LOOKUP` 默认召回深度是 6，`FACTOID` 是 8，`RELATION` 和 `SUMMARY` 是 12，`OVERVIEW` 是 14；如果 query 含编号，或者 token 很长，还会继续上调，但最多不超过 16。大体上 `LOOKUP`、`RELATION` 和属性型问题会被映射成 `STRICT` 防御画像，`SUMMARY` 和 `OVERVIEW` 是 `LOOSE`，其他一般是 `MODERATE`；但 `RELATION + 共同点` 是一个特例，它会直接下调到 `LOOSE`，因为这种问法本质上更像跨段综合，而不是原子事实核验。这样后面检索、Judge 和回答模式都不需要再各自重新理解 query。” 

### 5.4 再讲双路检索

“当前默认主路径只保留稳定、可解释的双路召回。第一路是向量召回，使用 `nomic-embed-text:latest` 做 embedding，向量库是 Chroma。第二路是文档级 BM25，不是对 chunk 做 BM25，而是对标题加正文的文档级快照做 BM25，并且标题会重复一次，增强标题命中。这样 dense 负责语义相近，BM25 负责精确词法锚定，两路各自覆盖对方的短板。” 

### 5.5 核心算法：自适应 hybrid 融合

“我最主要的算法改进就发生在这里。不是简单把 dense 和 BM25 固定按 0.5 比 0.5 相加，而是先做 query profile，再动态决定 lexical 和 dense 的权重。当前规则是：`lexical_weight` 初始值是 0.36；如果 query 含 identifier，加 0.18；如果 token 数不超过 4，再加 0.14；如果 token 数大于等于 10，减 0.08；如果存在显著高 IDF 稀有词，再加 0.10；最后裁剪到 `[0.24, 0.78]`。`dense_weight = 1 - lexical_weight`。短问题、带编号问题、强关键词问题会更依赖词法，长问题和抽象问题会更依赖 dense。” 

“在得到这两个自适应权重后，系统会继续把多种信号融合成 `adaptive_score`。当前公式是：

`adaptive_score = dense_weight * normalized_dense + lexical_weight * normalized_bm25 + 0.26 * normalized_rrf + 0.08 * coverage + 0.03 * identifier_overlap`

这里的设计思路是：dense 和 BM25 解决主信号，RRF 给双路高位稳定候选加分，coverage 用来约束 query 词覆盖率，identifier overlap 用来兜住编号和版本号一致性。这个阶段的目标不是直接输出最终答案，而是给 rerank 提供一个更干净的候选池。” 

### 5.6 再讲候选池构造

“候选池也不是简单取 adaptive top-k。当前代码会先取 adaptive 排序的前半偏多部分，再补 RRF 高位，再补 adaptive 剩余部分，如果还不够，再补向量列表和 BM25 列表。默认 `RAG_HYBRID_CANDIDATE_LIMIT = 30`。这样做是为了避免过早把互补候选扔掉。” 

### 5.7 再讲 rerank

“进入 rerank 阶段之后，系统使用的是 `BAAI/bge-reranker-v2-m3` 这个 cross-encoder。它的作用不是替代 hybrid，而是对已经筛过一轮的高质量候选做更深的相关性判断。当前 rerank 的输入长度上限是 `RERANK_MAX_LENGTH = 512`，单文档截断长度是 `RERANK_MAX_INPUT_CHARS = 1400`。” 

### 5.8 再讲统一证据评分器

“真正决定文档能不能进入回答上下文的，不再是 scattered rules，而是 `UnifiedEvidenceScorer`。当前基础相关性已经收口成两项：`adaptive_score` 和 `rerank_score`，权重是 `0.50 / 0.50`。也就是说，粗排阶段融合好的 hybrid 信号不会在评分器里再用另一套 dense、BM25、RRF 权重重算一遍，而是直接和 cross-encoder rerank 一起构成 `base_relevance`。在这个基础上，再叠加 `topic_alignment` 和 selective LLM Judge。topic alignment 的权重按防御画像区分：`STRICT 0.18`、`MODERATE 0.14`、`LOOSE 0.10`。Judge 信号权重是：`STRICT 0.24`、`MODERATE 0.18`、`LOOSE 0.0`。最终 `final_score` 是这些贡献项的加权平均，不再是散落的 if-else 门直接拍板。” 

“当前门槛也都是显式的。`MIN_BASE_RELEVANCE` 分别是：`STRICT 0.48`、`MODERATE 0.42`、`LOOSE 0.36`，但它现在只保留为观测 flag，不再单独作为拒答分支。真正会拦截的是：`MIN_TOPIC_ALIGNMENT`，分别是 `STRICT 0.20`、`MODERATE 0.08`、`LOOSE 0.05`；以及 `MIN_FINAL_SCORE`，分别是 `STRICT 0.58`、`MODERATE 0.45`、`LOOSE 0.46`。如果是 `STRICT + ATOMIC_SPAN` 场景，还必须满足 `direct_evidence = True`，否则直接拒掉。无 Judge 场景下，`direct_evidence` 的兜底条件也已经放宽到 `topic_alignment >= 0.35 and (title_alignment >= 0.20 or base_relevance >= 0.52)`。” 

### 5.9 再讲 selective LLM Judge

“Judge 这一层我采纳了，但不是全量开启。只有 `needs_judge = True` 时才会调，也就是高风险 query。当前 Judge 是并发跑的，不是串行阻塞主线程。评分器会先选 base relevance 最高的前几篇候选，最多 `MAX_JUDGE_CANDIDATES = 3`，然后用 `asyncio.gather(..., return_exceptions=True)` 并发调用 Judge，并且每篇文档都有 `asyncio.wait_for(..., timeout=6.0)` 的超时熔断。超时后直接记为 `judge_timeout`，该候选默认不可用。Judge 本身只输出结构化 JSON，判断三件事：核心主题是否匹配、是否有直接证据、是否真的可答。它不是生成器，只是高风险场景下的证据审核器。” 

### 5.10 再讲回答模式路由

“完成统一评分之后，系统进入 `AnswerModeRouter`。如果没有任何 usable 文档，直接 `NO_CONTEXT`。如果是 `SUMMARY` 或 `OVERVIEW`，或者证据要求是 `FULL_DOCUMENT`，直接走 `GENERATIVE`。如果是 `RELATION` 或 `REASON`，而且 usable 文档不止一篇，也走 `GENERATIVE`。如果是单文档 `REASON`，则走 `STRUCTURED`。如果是短答案型问题，并且主文档有直接证据、支持抽取、而且主文档比分第二名高出至少 `0.12`，就走 `EXTRACTIVE`。其他短答案型问题走 `STRUCTURED`，剩余场景再回落到 `GENERATIVE`。也就是说，路由器本身保持克制，像“多信息 FACTOID”这类补修主要落在生成器的上下文策略，而不是在路由器里再开新分支。” 

### 5.11 最后讲生成器与优雅降级

“当前生成层有三种模式。`ExtractiveGenerator` 优先从 Judge 的 `evidence_quote` 直接取证据；如果没有，再用段落窗口和句子窗口做轻量抽取，不引入复杂正则和重型 NLP。对于多信息点落在相邻句子的情况，它还会尝试拼接 top-2 句，减少答案被单句截断。抽取失败时不会崩，而是抛出受控的 `FallbackRequiredError`，自动降级到 `STRUCTURED`。`StructuredGenerator` 默认基于前两篇文档生成直接回答，但对 `RELATION`、`REASON` 和多信息 FACTOID 会扩大上下文上限，并改为读取 `full_content`。`GenerativeGenerator` 默认基于前三篇文档综合回答，只有在 `SUMMARY / OVERVIEW` 上才切到 `full_content`，并把上下文上限提到 `5`。整个降级链仍然是：`EXTRACTIVE -> STRUCTURED -> GENERATIVE -> NO_CONTEXT`。当前最后一道稳定性机制是路由和优雅降级，而不是生成后再做一轮审核。” 

### 5.12 最后用一句话收束

“所以如果总结我的贡献，最核心的是把 RAG 从一个检索后直接交给大模型的松散流程，收口成了意图驱动、统一评分、模式路由的可解释流水线。其中算法主贡献集中在 adaptive hybrid 和候选池构造，工程主贡献集中在 UnifiedEvidenceScorer 和 AnswerModeRouter 这两层。” 

## 6. 你要主动强调的“近期调整”

老师如果追问你“最近又做了什么改动”，你就按下面几条说：

1. 当前运行时入口已经从旧的大而全服务收口到 `RagService`，控制面明确拆成 `QueryIntentBuilder -> UnifiedEvidenceScorer -> AnswerModeRouter`。
2. 历史检索增强模块已经从代码库清理，主链路只保留稳定、可解释的 query 处理。
3. `LLM Judge` 已经接入，但只在 `STRICT` 高风险 query 上选择性启用，不做全量串行判断。
4. 数据流已经改成“chunk 做评分和抽取，`full_content` 只在概况类生成和关系 Judge 上按需使用”，不再是整条链路统一读全文。
5. 索引阶段新增了 `【核心主题：...】` 语义锚点注入，用来减少顺带提及误召回。
6. 多实体、多信息问法已经补修，但方式不是加新架构，而是在意图、评分和生成器上下文策略里补齐。
7. 生成后已经没有额外审核层，不要再讲成“系统里还有保留开关”。

## 7. 这几个设计点你要会解释

### 7.1 为什么要保留 BM25，不只靠 dense

推荐回答：

“dense 很擅长语义近似，但对编号、短标题、专有名词、短实体不一定稳定。BM25 恰好补 dense 的短板，所以我不是把 BM25 当老办法，而是把它作为 hybrid 系统里不可替代的一条词法视图。” 

### 7.2 为什么把历史检索增强模块彻底删掉

推荐回答：

“这些探索过的增强模块最后没有进入稳定运行面。它们会引入额外时延和不确定性，而且收益高度依赖 query 类型和语料域。为了让系统控制面更简单、配置更干净、联调口径更稳定，我把它们连同对应配置一起从代码库移除了。” 

### 7.3 为什么 Judge 不是全量开启

推荐回答：

“Judge 最适合解决的是高风险 query 的主题错配问题，比如关系型问题或 `X 的 Y` 属性型问题。对所有 query 全量开启会显著增加时延，也容易误杀本来能答的普通问题，所以我把它设计成选择性启用。” 

### 7.4 为什么不用复杂正则或 NLP 库做抽取

推荐回答：

“事实型问题看起来适合抽句，但自然语言表达非常多样，硬拆主谓宾在工程上并不稳。我这里用的是轻量窗口定位和句级打分，抽取失败就优雅降级到结构化回答，宁可多说一点，也不返回半句废话。” 

### 7.5 为什么当前不再设计额外的生成后审核层

推荐回答：

“生成后再做一轮审核会引入额外时延和新的不确定性，而且如果前面证据路由没做好，后处理审核也只是补丁。所以当前系统把稳定性前移到意图识别、统一评分和模式路由，不再保留额外的生成后审核层。” 

## 8. 你最该背熟的配置和数字

### 8.1 索引与模型

1. 向量切块：`chunk_size=1000`，`chunk_overlap=200`
2. Embedding 模型：`nomic-embed-text:latest`
3. Chat / Judge 模型：`qwen2.5:7b-instruct`
4. Rerank 模型：`BAAI/bge-reranker-v2-m3`
5. 向量库：`Chroma`

### 8.2 检索与候选

1. `RAG_RESULT_LIMIT = 3`
2. `RAG_VECTOR_SEARCH_LIMIT = 50`
3. `RAG_BM25_SEARCH_LIMIT = 50`
4. `RAG_HYBRID_CANDIDATE_LIMIT = 30`
5. `retrieval_depth` 由意图决定，`rag_service` 直接使用 `intent.retrieval_depth`

### 8.3 Judge 与评分

1. `MAX_JUDGE_CANDIDATES = 3`
2. `judge_timeout_seconds = 6.0`
3. `RAG_JUDGE_MAX_TOKENS = 220`
4. `RAG_JUDGE_CONTEXT_CHARS = 2200`

### 8.4 关键门槛

1. `MIN_BASE_RELEVANCE`: strict `0.48`，moderate `0.42`，loose `0.36`
2. `MIN_TOPIC_ALIGNMENT`: strict `0.20`，moderate `0.08`，loose `0.05`
3. `MIN_FINAL_SCORE`: strict `0.58`，moderate `0.45`，loose `0.46`
4. `PRIMARY_EVIDENCE_GAP = 0.12`

## 9. 一个稳妥的收尾

最后老师如果问“所以你到底做成了什么”，你可以这样收：

“我做的不是一个只会把文档塞给大模型的 RAG，而是一条可解释、可扩展的知识库问答主链路。它先通过 QueryIntentBuilder 明确问题类型和证据要求，再通过自适应 hybrid 和 rerank 找到更稳的候选，再通过 UnifiedEvidenceScorer 判断证据可用性，最后通过 AnswerModeRouter 决定应该拒答、抽取、结构化短答还是生成式回答。我的核心算法贡献主要在 hybrid 检索和候选池构造，工程价值主要在统一评分和显式路由。这套设计比单纯调 Prompt 更稳定，也更适合在真实后端系统里落地。” 

## 10. 答辩时不要讲错的几句话

1. 不要再说“当前主入口是 `AgentService`”。
2. 不要再说“当前代码里还保留着额外生成后审核层，随时可以打开”。
3. 不要再说“历史检索增强模块还是系统里的可选在线流程”。
4. 不要把旧的 `universal defense` 讲成当前唯一的主防御模块。
5. 要明确现在的主控制面是 `QueryIntentBuilder -> UnifiedEvidenceScorer -> AnswerModeRouter`。
