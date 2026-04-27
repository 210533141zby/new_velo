# RAG 主链路执行流程与联调手册

## 1. 这份文档的定位

这份文档现在负责三件事：

1. 记录当前 RAG 请求从进入接口到返回答案的真实执行顺序。
2. 记录本地启动、自检、探针、排障的标准联调流程。
3. 作为实时同步的工作日志，持续记录每一步操作、遇到的问题、分析和解决方案。

也就是说，这份文档不再只是“说明书”，而是：

`执行流程 + 联调手册 + 工作日志`

它不负责展开算法公式和答辩口径。那两部分请看：

1. `RAG与Rerank方案说明.md`
2. `RAG答辩讲稿与技术拆解.md`
3. `RAG主链路优化与问题闭环报告.md`

## 2. 实时同步规则

从现在开始，这份文档必须按下面规则持续更新：

1. 只要改了 RAG 主链路，就要同步追加一条工作日志。
2. 日志必须写清楚：
   - 时间
   - 本次目标
   - 操作范围
   - 现象 / 问题
   - 分析
   - 解决方案
   - 验证结果
   - 未解决风险
3. 不能只写“已优化”“已修复”这种空话，必须写出具体改了什么。
4. 如果只是文档同步，也要明确写“本次只同步文档，未改代码”。
5. 如果本次没有跑探针或单测，也必须写明“未执行验证”，不能省略。
6. 已废弃或已从代码库移除的能力，必须明确标注，不能混写成当前运行时能力。

推荐的同步节奏是：

1. 做完一次代码收口，立刻记一条。
2. 跑完一次探针，立刻补验证结果。
3. 出现一次误答或拒答异常，立刻补“问题现象 + 分析 + 修法”。

## 3. 当前端到端执行流程

一次知识库问题进入系统后，当前真实执行顺序是：

```text
用户输入问题
-> 前端请求 /api/v1/agent/chat
-> backend 进入 RagService.rag_qa
-> cache / 系统信息短路
-> QueryIntentBuilder.build
-> 单路向量召回
-> 文档级 BM25 补召回
-> adaptive hybrid 融合与候选池构造
-> cross-encoder rerank
-> UnifiedEvidenceScorer.assess_concurrently
-> AnswerModeRouter.route
-> GeneratorFactory.execute
-> 构造 sources
-> 前端展示答案与引用
```

当前默认主链路下必须记住四点：

1. 向量召回只走一条 `normalized_query`。
2. BM25 吃的是 `keyword_query`。
3. 检索前只保留 query 去噪和关键词视图，不再叠额外历史增强分支。
4. 生成后不再存在额外审核层，稳定性依赖模式路由和优雅降级。

## 4. 后端执行顺序展开

### 4.1 请求入口

当前知识库问答接口是：

- `/api/v1/agent/chat`

接口代码在：

- `backend/app/api/rag.py`

当前主服务入口是：

- `backend/app/services/rag/rag_service.py`

主函数是：

- `RagService.rag_qa(query: str) -> dict`

### 4.2 query 处理阶段

这一阶段由：

- `backend/app/services/rag/query_intent_builder.py`

负责，当前会输出：

1. `original_query`
2. `normalized_query`
3. `keyword_query`
4. `intent_type`
5. `retrieval_depth`
6. `defense_profile`
7. `evidence_requirement`
8. `needs_judge`
9. `trace_tags`

当前意图类型包括：

1. `LOOKUP`
2. `FACTOID`
3. `RELATION`
4. `SUMMARY`
5. `OVERVIEW`
6. `REASON`
7. `LOCATION`

### 4.3 候选召回阶段

这一阶段由两路信号组成：

1. Chroma 向量召回
2. 文档级 BM25 召回

当前默认行为：

1. 向量召回输入 `intent.normalized_query`
2. BM25 输入 `intent.keyword_query`
3. 默认不做多路向量循环
4. 检索前不再存在额外历史增强分支

### 4.4 hybrid 粗排阶段

这一阶段在：

- `backend/app/services/rag/hybrid_search.py`

当前负责：

1. 查询画像
2. 自适应 `dense_weight / lexical_weight`
3. RRF
4. `coverage / identifier` 融合
5. 混合候选池构造

### 4.5 rerank 精排阶段

这一阶段在：

- `backend/app/services/rag/rerank_service.py`

当前负责：

1. 加载 `BAAI/bge-reranker-v2-m3`
2. 对候选做 cross-encoder 打分
3. 返回归一化 `rerank_score`

当前要注意：

1. rerank 不是最终 `usable` 判定器。
2. 真正的可用性判断已经收口到 `UnifiedEvidenceScorer`。

### 4.6 统一证据评分阶段

这一阶段在：

- `backend/app/services/rag/evidence_scorer.py`

当前负责：

1. 基础相关性评分
2. `topic_alignment`
3. selective `LLM Judge`
4. `direct_evidence`
5. `supports_extractive`
6. `usable / reject_reason / flags`
7. 结构化 trace 日志

### 4.7 选择性 Judge 阶段

Judge 不是全量开启，而是高风险 query 才会启用。

当前相关代码在：

1. `backend/app/services/rag/evidence_judge.py`
2. `backend/app/services/rag/evidence_scorer.py`

当前运行方式：

1. `asyncio.gather(..., return_exceptions=True)` 并发判别
2. 每个文档用 `asyncio.wait_for(..., timeout=6.0)` 做超时熔断
3. Judge 最多评估前 `3` 个高相关候选

Judge 当前只做三件事：

1. 主题是否真匹配
2. 是否有直接证据
3. 是否真的可答

### 4.8 回答模式路由阶段

这一阶段在：

- `backend/app/services/rag/answer_mode_router.py`

当前模式包括：

1. `NO_CONTEXT`
2. `EXTRACTIVE`
3. `STRUCTURED`
4. `GENERATIVE`

### 4.9 生成与优雅降级阶段

这一阶段在：

- `backend/app/services/rag/answer_generators.py`

当前默认运行的是：

1. `EXTRACTIVE -> STRUCTURED`
2. `STRUCTURED -> GENERATIVE`
3. `GENERATIVE -> NO_CONTEXT`

当前生成后没有额外审核层。

### 4.10 前端展示阶段

最终返回给前端的是：

1. `response`
2. `sources`

每条 `source` 当前至少包含：

1. `title`
2. `doc_id`
3. `rank`
4. `confidence`

## 5. 本地启动顺序

当前推荐启动顺序固定为：

1. `ollama serve`
2. backend
3. frontend

### 5.1 启动 Ollama

```bash
cd /root/Velo
ollama serve
```

### 5.2 启动 backend

```bash
cd /root/Velo/backend
/root/Velo/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 5.3 启动 frontend

```bash
cd /root/Velo/frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

### 5.4 页面与健康检查

```text
前端: http://127.0.0.1:5173
后端: http://127.0.0.1:8000
健康检查: http://127.0.0.1:8000/health
```

## 6. 当前启动后的标准自检

建议按固定顺序检查。

### 6.1 先看健康检查

```bash
python - <<'PY'
import urllib.request
print(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=10).read().decode())
PY
```

### 6.2 再跑单测

```bash
/root/Velo/.venv/bin/python -m unittest \
  backend.tests.test_completion_policy \
  backend.tests.test_hybrid_retrieval \
  backend.tests.test_rag_pipeline_routing \
  backend.tests.test_unified_evidence_scorer \
  backend.tests.test_rag_pipeline_execution
```

### 6.3 再跑在线探针

```bash
/root/Velo/.venv/bin/python /root/Velo/backend/scripts/run_rag_probes.py
```

### 6.4 最后做页面实测

优先测这些类型：

1. 单文档事实抽取
2. 编号问题
3. 概况问题
4. 带噪问法
5. 短实体问题
6. 关系型问题
7. 属性型错配问题

## 7. 当前推荐的探针问题

当前 probe 脚本覆盖：

1. `RAG测试文档的默认聊天模型是什么`
2. `为什么研究者重视编号为A-17-204的蓝布账册`
3. `雾潮镇的历史`
4. `Shor算法能解决什么问题`
5. `请问一下，Shor主要解决什么问题吗？`
6. `青苗法是什么`
7. `沈阳`
8. `韩国与新加坡的关系`
9. `韩国`
10. `新加坡的档案馆`
11. `新加坡的风景`
12. `陈鹤汀`
13. `沈见川`
14. `韩启明与档案馆的关系是什么`
15. `沈见川和陈鹤汀有什么共同点`
16. `文章中提到了哪些人物？他们分别与档案馆有什么关系？`

这组问题分别对应：

1. 单文档抽取
2. 编号问题
3. 概况问题
4. 标准术语问题
5. 带噪问法问题
6. 普通知识问题
7. 短实体边界
8. 关系型边界
9. 单实体无上下文边界
10. 属性型对象错配边界
11. 主体词命中但主题对象错配边界
12. 短实体命中回归
13. 第二个短实体命中回归
14. 人物与机构关系问法回归
15. 多实体共同点综合回归
16. 多人物列举与关系综合回归

## 8. 当前最该看的日志

### 8.1 文件位置

当前主要看：

1. `logs/velo_app.log`
2. `logs/backend.log`
3. `logs/frontend.log`
4. `logs/ollama.log`

### 8.2 当前后端主链路事件

当前代码里真实存在、最值得看的事件有：

1. `rag_cache_hit`
2. `rag_intent_built`
3. `rag_pipeline_routed`
4. `rag_generator_fallback`
5. `rag_evidence_assessed`
6. `rag_qa_success`
7. `rag_qa_failed`
8. `rag_document_judge_failed`
9. `rag_hybrid_index_refreshed`
10. `rag_index_success`
11. `rag_index_retry`
12. `rag_index_failed`
13. `rerank_model_loaded`
14. `rerank_model_load_failed`
15. `rerank_inference_failed`

这里要特别注意：

1. 文档里旧的 `rag_qa_no_context`
2. `rag_query_plan_failed`
3. `rag_document_judge_reject`
4. 旧的生成后审核拒绝事件

这些不是当前运行时代码里的主事件，不能再按旧文档去查。

### 8.3 当前最值得盯的结构化字段

联调时优先看这些字段：

1. `intent_type`
2. `retrieval_depth`
3. `defense_profile`
4. `evidence_requirement`
5. `trace_tags`
6. `candidate_count`
7. `assessment_count`
8. `usable_count`
9. `answer_mode`
10. `reason`
11. `source_doc_ids`
12. `doc_id`
13. `adaptive_score`
14. `base_relevance`
15. `topic_match`
16. `judge_latency_ms`
17. `flags`
18. `usable`
19. `final_score`
20. `judge_status`

## 9. 常见问题的联调顺序

### 9.1 页面直接没有回答

先查：

1. `/health` 是否正常
2. backend 是否收到请求
3. `ollama serve` 是否还活着
4. `rag_qa_failed` 是否出现

### 9.2 页面有回答但明显不对

先查：

1. source 是否明显错了
2. `rag_intent_built` 是否把 query 判错了
3. `rag_evidence_assessed` 里 top 文档是否本来就被错判成 `usable`
4. `rag_pipeline_routed` 是否把问题送到了错误回答模式

### 9.3 页面总是拒答

先查：

1. `usable_count` 是否为 0
2. `flags` 是否出现 `LOW_BASE_RELEVANCE`
3. `flags` 是否出现 `WEAK_TOPIC_ALIGNMENT`
4. `flags` 是否出现 `FAILED_SHORT_ENTITY_MATCH`
5. `flags` 是否出现 `FAILED_LLM_JUDGE`
6. `flags` 是否出现 `LLM_JUDGE_TIMEOUT`

### 9.4 Judge 相关问题

先查：

1. query 是否真的走到了 `STRICT`
2. `needs_judge` 是否为 `True`
3. `judge_latency_ms` 是否异常高
4. 是否出现 `judge_timeout`
5. `rag_document_judge_failed` 是否存在

### 9.5 向量或 embedding 相关错误

先查：

1. `ollama serve` 是否存活
2. `http://127.0.0.1:11434` 是否可达
3. `nomic-embed-text:latest` 是否可用

### 9.6 rerank 相关错误

先查：

1. `rerank_model_loaded` 是否成功出现
2. 是否出现 `rerank_model_load_failed`
3. 是否出现 `rerank_inference_failed`
4. 本地模型缓存目录是否完整

## 10. 工作日志模板

以后每次都按这个格式追加，不要省：

### [日期] 本次标题

1. 目标
2. 操作范围
3. 问题现象
4. 分析
5. 解决方案
6. 验证结果
7. 未解决风险

示例写法：

```text
### [2026-04-17] 修正属性型 query 主体错配
1. 目标
   把“新加坡的档案馆”这类主体错配问题拦住。
2. 操作范围
   evidence_scorer.py / evidence_judge.py / probe 脚本。
3. 问题现象
   文档里同时出现“新加坡”和“档案馆”，但讨论对象是雾潮镇档案馆。
4. 分析
   旧规则只能看见词出现了，不能判断是不是同一主体。
5. 解决方案
   在 STRICT query 上选择性启用 LLM Judge，做结构化主题和证据判别。
6. 验证结果
   probe 中“新加坡的档案馆”返回 no-context。
7. 未解决风险
   Judge 超时会直接拒答，后续要继续观察时延。
```

## 11. 最近一轮同步日志

### [2026-04-17] 修复 chunk 与全文粒度错配导致的过度拒答

1. 目标
   修复“检索命中 chunk，但评分和抽取却读全文”导致的 topic alignment 稀释和 STRICT 模式过度拒答。
2. 操作范围
   `backend/app/services/rag/pipeline_models.py`、`rag_service.py`、`evidence_scorer.py`、`answer_generators.py`
3. 问题现象
   检索阶段命中的是单个 chunk，但 `rag_service` 在构造候选时把 `doc.page_content` 替换成整篇全文，后续评分器和抽取器都在全文上做 topic alignment 和窗口抽取，导致：
   - 关键词密度被全文稀释
   - `STRICT` query 更容易被判成 topic mismatch
   - 抽取器在全文上做窗口选择，命中噪声增加
4. 分析
   检索、评分、抽取、生成使用的文本粒度不一致。评分和抽取需要保留检索命中的 chunk 语义密度，而生成式回答才真正需要完整文档视角。
5. 解决方案
   - 在 `RetrievedCandidate` 中新增 `adaptive_score`、`chunk_text`、`full_content`
   - `rag_service` 构造候选时保留原始 chunk 到 `chunk_text`
   - `candidate.doc.page_content` 改回 chunk 文本，不再写入全文
   - `full_content` 单独保存，供生成式模式使用
   - `UnifiedEvidenceScorer` 评分时优先读取 `chunk_text`
   - `ExtractiveGenerator` 抽取时优先读取 `chunk_text`
   - `GenerativeGenerator` 组装上下文时改为读取 `full_content`
6. 验证结果
   通过 `py_compile` 和以下单测：
   - `backend.tests.test_unified_evidence_scorer`
   - `backend.tests.test_rag_pipeline_routing`
   - `backend.tests.test_rag_pipeline_execution`
   - `backend.tests.test_hybrid_retrieval`
7. 未解决风险
   当前仍是“每篇文档只保留最佳命中的一个 chunk”进入评分器；如果后续要支持多证据片段聚合，还需要单独设计候选聚合策略。

### [2026-04-17] 修复召回深度被主链路强制抬升

1. 目标
   让召回深度重新由 `QueryIntentBuilder` 控制，而不是在主链路里被统一抬高。
2. 操作范围
   `backend/app/services/rag/rag_service.py`
3. 问题现象
   原实现使用 `max(intent.retrieval_depth, settings.RAG_RESULT_LIMIT * 4)`，导致 `LOOKUP` 等本来应该走浅召回的 query 被强制拉高到至少 12 条候选，引入额外噪音。
4. 分析
   召回深度属于意图层配置，不应该在主链路再做统一覆盖，否则意图路由的设计会失效。
5. 解决方案
   将 `retrieval_limit` 改为直接使用 `intent.retrieval_depth`，不再做 `max(..., 12)` 式抬升。
6. 验证结果
   相关 RAG 主链路单测通过；本次未单独跑线上 probe。
7. 未解决风险
   如果后续某类 query 确实召回不足，应回到 `QueryIntentBuilder.DEFAULT_RETRIEVAL_DEPTH` 调整，不应再次在主链路硬覆盖。

### [2026-04-17] 简化分数门槛并复用 coarse 阶段 adaptive_score

1. 目标
   降低双重门槛导致的误拒答，并消除粗排与评分阶段对同类信号的重复计算。
2. 操作范围
   `backend/app/services/rag/evidence_scorer.py`
3. 问题现象
   旧实现中：
   - `base_relevance` 由 dense、BM25、RRF、coverage、identifier 在评分阶段重新算一遍
   - `_determine_usability()` 同时用 `MIN_BASE_RELEVANCE` 和 `MIN_FINAL_SCORE` 双重拦截
   这会导致 coarse 排序和后续评分逻辑不一致，也会把“base 达标但 final 略低”的文档误杀。
4. 分析
   `adaptive_score` 已经在粗排阶段融合了大部分检索信号，再在评分阶段用另一套权重重复计算，既增加维护成本，也容易让排序与拒答边界脱节。
5. 解决方案
   - 新增 `CandidateSnapshot.adaptive_score`
   - `base_relevance` 改为只复用 `adaptive_score` 和 `rerank_score` 计算，不再重新展开 dense / BM25 / RRF / coverage / identifier
   - `final_score` 改为以 `base_signal + topic_signal + judge_signal` 聚合
   - 保留 `LOW_BASE_RELEVANCE` flag 用于观测
   - 删除 `base_relevance < MIN_BASE_RELEVANCE` 的直接拒答分支，只保留 `MIN_FINAL_SCORE` 作为唯一分数门槛
6. 验证结果
   `test_unified_evidence_scorer`、`test_rag_pipeline_routing`、`test_rag_pipeline_execution` 全部通过。
7. 未解决风险
   `MIN_FINAL_SCORE` 仍然是经验阈值，后续最好结合 probe 结果继续校准，而不是长期静态使用。

### [2026-04-17] 在线 probe 回归验证通过

1. 目标
   确认本轮 P0/P1 修复没有破坏当前主链路的实际回答行为。
2. 操作范围
   `backend/scripts/run_rag_probes.py`
3. 问题现象
   本轮修改涉及候选数据粒度、评分器门槛和生成器读数路径，如果只看单测，仍然可能漏掉实际联调回归。
4. 分析
   需要用真实接口回归验证：
   - 单文档抽取
   - 编号问题
   - 概况问题
   - 带噪问法
   - 短实体拒答
   - 关系型拒答
   - 属性型错配拒答
5. 解决方案
   在本地 backend 存活的前提下执行 `run_rag_probes.py` 做端到端验证。
6. 验证结果
   probe 全部通过，关键结果包括：
   - `RAG测试文档的默认聊天模型是什么` 正常抽取
   - `为什么研究者重视编号为A-17-204的蓝布账册` 正常命中
   - `雾潮镇的历史` 正常生成概况
   - `沈阳`、`韩国与新加坡的关系`、`新加坡的档案馆`、`新加坡的风景` 均正确拒答
7. 未解决风险
   当前 probe 集仍是有限样本，后续如果再调 `MIN_FINAL_SCORE` 或 Judge 策略，仍需要继续补充更细的回归样例。

### [2026-04-17] 文档口径与当前运行时代码不一致

1. 目标
   把 RAG 文档口径和当前 backend 实现重新对齐。
2. 操作范围
   `docs/RAG/README.md`、`RAG与Rerank方案说明.md`、`RAG主链路执行流程与联调手册.md`、`RAG主链路优化与问题闭环报告.md`、`RAG答辩讲稿与技术拆解.md`。
3. 问题现象
   文档里仍在把 `AgentService`、`universal defense`、历史增强模块默认主路径、额外审核层默认启用写成当前运行时事实。
4. 分析
   代码已经收口到 `RagService + QueryIntentBuilder + UnifiedEvidenceScorer + AnswerModeRouter`，但文档没有同步，导致答辩口径、联调说明和真实运行态脱节。
5. 解决方案
   以当前代码为准，统一重写文档口径，明确：
   - 当前主入口是 `RagService`
   - 当前主控制面是 `QueryIntentBuilder -> UnifiedEvidenceScorer -> AnswerModeRouter`
   - 历史增强模块当前默认未挂接
   - 额外审核层当前默认未挂接
   - `LLM Judge` 为选择性启用
6. 验证结果
   本次按代码逐文件核对完成文档同步；本次同步中未重新跑 probe 和单测。
7. 未解决风险
   后续如果再次调整运行时代码而不同步文档，会再次出现口径漂移。

### [2026-04-17] 执行流程文档缺少“工作日志”能力

1. 目标
   把执行流程文档改成可以持续记录项目操作过程的工作台账。
2. 操作范围
   `docs/RAG/RAG主链路执行流程与联调手册.md`
3. 问题现象
   旧文档更像静态说明书，只有流程和联调步骤，没有记录“做了什么、遇到什么问题、怎么分析、怎么解决”的位置。
4. 分析
   没有持续日志时，后面回看时只能看到结论，看不到决策过程，也不利于答辩时解释系统为什么会这样演化。
5. 解决方案
   在这份文档里新增：
   - 实时同步规则
   - 工作日志模板
   - 最近一轮同步日志
   以后每次 RAG 主链路改动都必须追加记录。
6. 验证结果
   当前文档结构已经改为“执行流程 + 联调手册 + 工作日志”。
7. 未解决风险
   这套规则只有在后续每次改动都坚持追加时才有价值。

### [2026-04-17] 联调日志事件清单过时

1. 目标
   把执行手册里的日志事件改成当前代码真实存在的事件。
2. 操作范围
   `docs/RAG/RAG主链路执行流程与联调手册.md`
3. 问题现象
   旧文档中仍然写着 `rag_qa_no_context`、`rag_query_plan_failed`、`rag_document_judge_reject`、旧的生成后审核拒绝事件 这类当前主链路并不依赖的旧事件。
4. 分析
   这些事件属于旧阶段残留说法，当前代码里主要事件已经变成 `rag_intent_built`、`rag_evidence_assessed`、`rag_pipeline_routed`、`rag_generator_fallback` 等结构化日志。
5. 解决方案
   按当前代码实际事件重新整理日志检查清单，并把需要重点查看的结构化字段一起写入文档。
6. 验证结果
   本次通过代码搜索确认了当前主链路事件清单；本次未重新跑线上日志验证。
7. 未解决风险
   后续若新增事件但未补文档，联调说明仍可能再次过期。

### [2026-04-17] 当前默认主链路的运行边界重新标注

1. 目标
   把“当前默认运行能力”和“保留但未挂接能力”明确区分开。
2. 操作范围
   RAG 文档全集，尤其是执行手册和答辩稿。
3. 问题现象
   容易把早期残留口径里的历史增强模块和额外审核层讲成当前在线主路径能力。
4. 分析
   这种混写会直接影响联调判断和答辩表达，尤其会让人误以为当前每个 query 都会先走额外增强再走额外审核。
5. 解决方案
   明确标注：
   - 历史增强模块当前默认未挂接
   - 额外审核层当前默认未挂接
   - `LLM Judge` 当前只在 `STRICT` 高风险 query 上启用
6. 验证结果
   本次文档已经统一改口径；本次未重新跑探针。
7. 未解决风险
   如果后续重新接回额外增强或额外审核逻辑，必须第一时间在这里补一条新的运行日志。

### [2026-04-17] 修复多实体综合问题的过度拒答

1. 目标
   修复“共同点”“文章中提到了哪些人物，他们分别与档案馆有什么关系”这类多实体综合问题被直接拒答的现象。
2. 操作范围
   `backend/app/services/rag/query_intent_builder.py`、`prompt_templates.py`、`evidence_scorer.py`、`answer_mode_router.py`、`answer_generators.py`、`hybrid_search.py`、`RAG主链路优化与问题闭环报告.md`
3. 问题现象
   `沈见川和陈鹤汀有什么共同点？`、`文章中提到了哪些人物？他们分别与档案馆有什么关系？` 这两类 query 在已有知识库文档存在相关信息时仍然容易走到 `NO_CONTEXT`。
4. 分析
   多实体综合问题本质上不是单点抽取，而是跨段拼接和归纳。旧行为里：
   - `RELATION` 识别对“共同点”不够敏感
   - 单 chunk Judge 容易把“每段只覆盖部分人物”的证据误判为不可答
   - 路由仍偏向直接证据式回答，不适合综合型关系问法
5. 解决方案
   - 在 `RELATION_MARKERS` 中加入 `共同点`
   - 对多实体关系综合问题放宽 `LOOSE` 模式 topic alignment 阈值
   - Judge 在 `RELATION` 场景优先读取 `full_content`
   - hybrid 检索补词法回退，降低“同文档选错 chunk”概率
   - `共同点`、`哪些人物` 这类问法直接路由到 `GENERATIVE`
   - 关系型生成扩大上下文窗口，减少单段证据视角过窄的问题
6. 验证结果
   本轮补修后，这两类问题不再因为评分器和路由器过度保守而直接拒答；详细闭环说明已同步到 `RAG主链路优化与问题闭环报告.md` 第 `3.7` 节。
7. 未解决风险
   这类问题现在更多依赖全文级综合生成质量；如果后续文档更长、人物更多，仍需继续观察生成稳定性和引用可解释性。

### [2026-04-17] 启动前后端并完成本地可访问性校验

1. 目标
   启动本地前后端，保证可以继续做页面联调和知识库问答验证。
2. 操作范围
   frontend dev server、backend `uvicorn` 进程、本文档工作日志同步。
3. 问题现象
   检查时发现 frontend 已经在 `127.0.0.1:5173` 运行，但 backend `127.0.0.1:8000` 连接被拒绝。
4. 分析
   前端进程无需重启；后端只是未存活。这里如果继续用交互式会话临时拉起，后续很容易在会话结束时把服务一起带停，所以需要单独后台启动。
5. 解决方案
   - 复用现有 frontend 进程，不做重复启动
   - 用后台方式重新启动 backend：`/root/Velo/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`
   - 通过根路径和 `/health` 对服务做可访问性检查
6. 验证结果
   - frontend `http://127.0.0.1:5173/` 返回 `HTTP 200`
   - backend 完成启动预热，`http://127.0.0.1:8000/` 返回 `HTTP 405 Method Not Allowed`，说明服务在线但根路径不接受 `HEAD`
   - backend `http://127.0.0.1:8000/health` 返回 `{"status":"ok"}`
7. 未解决风险
   backend 的长期稳定性仍取决于本地依赖和模型服务是否持续可用；如果后续出现回答失败，需要继续联动检查 `ollama`、向量库和日志事件。

### [2026-04-17] 从代码库彻底移除历史增强与额外审核逻辑

1. 目标
   删除历史增强与额外审核逻辑，避免继续保留无效配置、无效 Prompt 和过期文档口径。
2. 操作范围
   `backend/app/core/config.py`、`backend/app/services/rag/prompt_templates.py`、`docs/RAG/README.md`、`docs/RAG/RAG与Rerank方案说明.md`、`docs/RAG/RAG主链路优化与问题闭环报告.md`、`docs/RAG/RAG主链路执行流程与联调手册.md`、`docs/RAG/RAG答辩讲稿与技术拆解.md`、对应的历史说明文档
3. 问题现象
   当前主链路早就不再依赖这些历史逻辑，但代码里还残留旧配置项和旧 Prompt，文档里也仍有“保留但未挂接”的描述，容易让联调、答辩和后续维护误以为这些能力还在系统里。
4. 分析
   这类“代码不调用但配置和文档还在”的状态最容易制造误解，也会让配置面持续膨胀。既然当前架构已经稳定收口到 `QueryIntentBuilder -> UnifiedEvidenceScorer -> AnswerModeRouter`，就不应该再保留这类历史能力的运行时痕迹。
5. 解决方案
   - 删除旧的生成后审核 Prompt 和相关过期说明
   - 删除历史增强与额外审核相关配置项，以及已无实际用途的 `RAG_SEARCH_LIMIT`、`RAG_MIN_FINAL_SCORE`、`RAG_KEEP_SCORE_RATIO`
   - 删除对应的历史说明文档
   - 将 RAG 文档口径统一改成“已从代码库移除”，不再写成“保留但未启用”
   - 删除后重新启动 backend 和 frontend，确保运行态与文档一致
6. 验证结果
   - `cd /root/Velo/backend && /root/Velo/.venv/bin/python -m py_compile app/services/rag/*.py` 通过
   - `/root/Velo/.venv/bin/python -m unittest discover -s backend/tests -p "test_*.py"` 通过，`33/33`
   - 重启后 frontend `http://127.0.0.1:5173/` 返回 `HTTP 200`
   - 重启后 backend `http://127.0.0.1:8000/health` 返回 `{"status":"ok"}`
7. 未解决风险
   旧运行日志文件里仍可能保留历史栈信息或旧事件名，但这些只是历史记录，不代表当前代码库仍然包含对应能力。

### [2026-04-20] 入库前增加语义锚点注入，修正顺带提及误召回

1. 目标
   降低“文档只顺带提到实体，却被误当成核心主题召回”的概率，尤其是地点名、机构名和人物名的泛匹配问题。
2. 操作范围
   `backend/app/services/rag/vector_index_service.py`、索引链路、本文档工作日志同步。
3. 问题现象
   之前部分 chunk 虽然包含“新加坡”“档案馆”之类词，但正文核心并不在讨论用户真正想问的对象，检索时容易被顺带提及误召回。
4. 分析
   只靠正文 chunk 本身，有时缺少足够强的主题锚点。标题主题、首段主题和局部句子的语义焦点不总是一致，导致 dense 和词法召回都可能把“提到了这个词”的 chunk 拉进候选池。
5. 解决方案
   - 在 `vector_index_service.py` 新增 `_extract_core_entities(title, first_paragraph)`
   - 用 `jieba.posseg` 提取标题和首段中的名词，过滤 `文档 / 资料 / 介绍 / 内容 / 背景 / 历史 / 概况 / 测试` 等停用词
   - 取权重最高的 1 到 3 个核心实体，生成 `【核心主题：实体1、实体2】`
   - 在 `index_document_chunks()` 中把该前缀注入每个 chunk 的 `page_content` 头部，不改 metadata
6. 验证结果
   重新索引后可在相似召回结果中直接看到 `【核心主题：...】` 前缀；该修正已进入当前代码主链路。
7. 未解决风险
   如果标题和首段本身就写得非常抽象，轻量实体抽取的收益会下降；当前也依赖 `jieba.posseg` 可用，缺依赖时会自动跳过注入。

### [2026-04-20] 补修多信息点与多实体综合问法

1. 目标
   修复“共同点”“文章中提到了哪些人物”“分别与某对象有什么关系”“承担什么角色”“捐出了哪些材料”这类问题的拒答和回答不完整问题。
2. 操作范围
   `backend/app/services/rag/query_intent_builder.py`、`evidence_scorer.py`、`answer_generators.py`、`prompt_templates.py`、本文档工作日志同步。
3. 问题现象
   这类问题之前常见两种失败：
   - 直接被拒答，因为单 chunk 证据不足以覆盖多个信息点
   - 没有拒答，但答案只覆盖一个人物、一个角色或一条材料信息
4. 分析
   这不是单纯的“模型生成能力不够”，而是当前管线原本偏向原子事实抽取。多信息问法往往需要跨段综合、列举多个要点，继续按单句抽取或短答约束容易丢信息。
5. 解决方案
   - `RELATION_MARKERS` 增加 `共同点`
   - `RELATION + 共同点` 下调到更宽松的防御画像，减少过度拒答
   - `MIN_TOPIC_ALIGNMENT` 改为：`STRICT 0.20 / MODERATE 0.08 / LOOSE 0.05`
   - `MIN_FINAL_SCORE` 改为：`STRICT 0.58 / MODERATE 0.45 / LOOSE 0.46`
   - 无 Judge 场景下的 `direct_evidence` 判定放宽到 `topic_alignment >= 0.35 and (title_alignment >= 0.20 or base_relevance >= 0.52)`
   - Judge Prompt 增加“比较两个实体 / 共同点 / 多人物关系”判定规则
   - `answer_generators.py` 中新增多信息标记词，并让 `StructuredGenerator` 对这类问题扩大上下文、读取 `full_content`
   - `ExtractiveGenerator` 允许 top-2 句拼接，减少列举型答案被单句截断
6. 验证结果
   相关 probe 已覆盖：
   - `陈鹤汀`
   - `沈见川`
   - `韩启明与档案馆的关系是什么`
   - `沈见川和陈鹤汀有什么共同点`
   - `文章中提到了哪些人物？他们分别与档案馆有什么关系？`
   当前这些问题已被纳入本手册第 7 节的常规回归列表。
7. 未解决风险
   多信息问题现在更依赖全文级综合和模型列举能力，如果文档跨度再变大，后续仍需要继续观察答案完整性与引用可解释性。

### [2026-04-20] Prompt 去领域特调，保留通用边界防御

1. 目标
   去掉对《雾潮镇档案馆》语料过于贴身的提示词和历史段落加权词，确保当前链路在公共基准和其他知识库上也成立。
2. 操作范围
   `backend/app/services/rag/prompt_templates.py`、`backend/app/services/rag/answer_generators.py`
3. 问题现象
   旧版本 Prompt 和历史段落打分里出现过于具体的领域词，例如“馆长年龄”“档案馆”“渔业”“捕鱼”等，这些词在当前语料有效，但不应该作为通用系统行为长期固化。
4. 分析
   这类特调虽然能短期提高本地样本表现，但会污染答辩口径，也会让 RGB 这类公共基准对比失真。
5. 解决方案
   - `build_general_rag_prompt()` 删除领域化示例，保留“概括类问题先回答核心对象”和“顺带提及必须拒答”两条通用规则
   - `build_structured_rag_prompt()` 改成按问题需要输出适当长度，明确多信息问题要完整列出
   - `_history_paragraph_score()` 改成通用年份与历史词：`\d{4}年`、`最早 / 建于 / 创立 / 成立 / 起步 / 发展 / 后来 / 直到 / 最初 / 早期`
6. 验证结果
   当前 Prompt 已不再写死《雾潮镇档案馆》特征词；这一步也为后续 RGB 对比评测提供了更干净的基线。
7. 未解决风险
   去特调后，本地语料上的个别历史类问题可能失去一部分场景红利，因此后续只能继续靠通用检索、评分和路由逻辑提升，而不能再回退到特定语料关键词硬编码。

### [2026-04-20] 搭建“本地知识库 + RGB”双数据集对比评测链路

1. 目标
   把自研 RAG 与 LlamaIndex 的对比从“本地知识库体感”提升到“本地数据集 + RGB 公共基准”的双视角评测，并输出可直接展示的图表。
2. 操作范围
   `llamaindex_rag_eval/compare_rag_systems.py`、`llamaindex_rag_eval/outputs/`、`llamaindex_rag_eval/data/rgb/outputs/`
3. 问题现象
   之前的对比主要依赖本地知识库，容易被追问“是不是只对你自己的文档有效”，缺少公共基准证明。
4. 分析
   本地知识库评测可以证明系统贴近真实业务，但不能单独证明通用性；RGB 可以补足这一点。两者一起看，才能把“本地适配效果”和“跨数据集泛化能力”区分开。
5. 解决方案
   - 用 `compare_rag_systems.py` 固定对齐参数，对比自研 RAG 与 LlamaIndex
   - 输出本地知识库对比结果：`rag_comparison_summary.json/.csv/.svg`
   - 输出 RGB 对比结果：`rgb_rag_comparison_summary.json/.csv/.svg`
   - 再汇总为双数据集总图：`dataset_comparison_summary.json/.csv/.svg`
6. 验证结果
   当前已保存的一组结果为：
   - 本地知识库：自研 RAG `faithfulness 1.0`、`answer_relevancy 0.8112`、`accuracy 0.3`；LlamaIndex `faithfulness 0.85`、`answer_relevancy 0.8545`、`accuracy 0.2`
   - RGB：自研 RAG `faithfulness 0.92`、`answer_relevancy 0.8113`、`context_precision 0.93`、`accuracy 0.8033`；LlamaIndex `faithfulness 0.8517`、`answer_relevancy 0.6429`、`context_precision 0.7333`、`accuracy 0.59`
   - 时延权衡：RGB 上自研 RAG `P95 3597.06ms`，LlamaIndex `P95 1412.95ms`
7. 未解决风险
   这组结果是当前一次运行的快照，不代表永久稳定上界；只要继续调整阈值、路由或评测样本，就必须重新生成对应图表与 CSV。

### [2026-04-20] 同步补写 RAG 近期变动到技术文档和工作日志

1. 目标
   把已经落地的 RAG 主链路改动同步写回技术文档和工作日志，避免文档继续滞后于代码。
2. 操作范围
   `docs/RAG/RAG主链路优化与问题闭环报告.md`、`docs/RAG/RAG主链路执行流程与联调手册.md`、`docs/RAG/RAG答辩讲稿与技术拆解.md`
3. 问题现象
   文档里还保留了“全文评分”“旧阈值”“旧路由”和未及时补写的近期变动，导致代码、日志和答辩口径没有完全对齐。
4. 分析
   这种错位会直接影响联调、复盘和答辩表达。尤其是近期做过数据流修正、语义锚点注入、多信息问法补修和双数据集评测，如果不写回文档，后续就很难追踪“为什么现在代码是这样”。
5. 解决方案
   - 在技术文档中补写 chunk / full_content 真实分工、语义锚点注入、Prompt 去特调和双数据集评测
   - 在执行手册中追加 2026-04-20 的工作日志条目
   - 在答辩稿中把评分阈值、路由规则、生成器行为和近期调整同步到当前实现
6. 验证结果
   本次为文档同步，不涉及新的代码行为变更；已按当前代码状态完成文档更新。
7. 未解决风险
   只要后续再改 `evidence_scorer.py`、`answer_mode_router.py`、`answer_generators.py` 或评测脚本，就必须继续实时追加日志，不能再攒到下一轮统一补写。

## 12. 当前目录里的文档分工

为了避免一份文档里什么都写，当前 RAG 目录按下面方式分工：

1. `README.md`
   目录索引和阅读顺序。
2. `RAG与Rerank方案说明.md`
   技术总览、模块职责、模型选型、打分逻辑。
3. `RAG主链路执行流程与联调手册.md`
   启动、自检、探针、日志、联调流程、实时工作日志。
4. `RAG主链路优化与问题闭环报告.md`
   问题模式、演进过程、修正闭环。
5. `RAG答辩讲稿与技术拆解.md`
   面向答辩的话术和技术拆解。

## 13. 一句话总结

这份手册现在不是单纯讲“怎么跑”，而是讲：

`当前这条 RAG 主链路具体怎么跑、怎么验、怎么排查，以及最近到底做过哪些操作、遇到过什么问题、怎么解决。`
