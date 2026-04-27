# RAG 主链路优化与问题闭环报告

## 1. 文档定位

这份文档只做三件事：

1. 记录 RAG 主链路是怎么从早期堆叠方案收口到当前运行时架构的。
2. 记录近期做过哪些关键调整，哪些旧说法已经过期。
3. 记录典型错误类型在当前主链路里分别由哪一层负责兜底。

它不负责展开所有公式和所有 Prompt。那部分请看：

1. `RAG与Rerank方案说明.md`
2. `RAG答辩讲稿与技术拆解.md`
3. `RAG主链路执行流程与联调手册.md`

## 2. 当前主链路的真实状态

当前 backend 实际运行的主链路已经收口为：

```text
query
-> QueryIntentBuilder
-> 向量召回
-> 文档级 BM25
-> adaptive hybrid 候选构造
-> cross-encoder rerank
-> UnifiedEvidenceScorer
-> AnswerModeRouter
-> Extractive / Structured / Generative
-> sources 返回
```

当前主路径对应代码：

1. `backend/app/services/rag/rag_service.py`
2. `backend/app/services/rag/query_intent_builder.py`
3. `backend/app/services/rag/hybrid_search.py`
4. `backend/app/services/rag/rerank_service.py`
5. `backend/app/services/rag/evidence_scorer.py`
6. `backend/app/services/rag/answer_mode_router.py`
7. `backend/app/services/rag/answer_generators.py`

## 3. 近期已经落地的调整

这一节就是最近这轮必须写进文档的内容。

### 3.1 从 `AgentService` 收口到 `RagService`

旧口径里经常讲 `AgentService.rag_qa`，这已经不是当前主链路的真实入口。

现在真实入口是：

- `backend/app/services/rag/rag_service.py`

这一步的变化不是简单改文件名，而是把之前堆在一个大文件里的检索、规则、防御、回答分流拆成了稳定的管道。

### 3.2 从“补丁式防御”收口到三段式控制面

旧口径里常说：

1. `universal defense`
2. 一堆零散规则门
3. 回答前后再叠额外判断

这些说法现在不能再当成主链路描述。

当前真正的控制面是三段：

1. `QueryIntentBuilder`
   负责把 query 变成明确的检索深度、防御强度和证据要求。
2. `UnifiedEvidenceScorer`
   负责把多路召回、rerank、topic alignment、selective judge 收口成统一的 `usable` 判断。
3. `AnswerModeRouter`
   负责把“能不能答、怎么答”收口成 `NO_CONTEXT / EXTRACTIVE / STRUCTURED / GENERATIVE`。

也就是说，现在不是“规则上面再补判断”，而是“意图驱动 + 统一评分 + 模式路由”。

### 3.3 历史检索增强模块已从代码库移除

这一轮清理后，这类历史增强模块不再是“保留但未挂接”，而是已经从代码库中彻底移除。

这一步的意义是：

1. 主链路只保留真实在线使用的组件
2. 配置面不再出现无效开关
3. 文档和答辩口径不再需要区分“保留但未启用”的历史能力

### 3.4 `LLM as a Judge` 已采纳，但改成选择性启用

这一点不是“计划中”，而是已经接进了运行时代码。

当前逻辑是：

1. `QueryIntentBuilder` 先给 query 打 `defense_profile`
2. 只有严格防御且符合原子证据要求的 query 才更容易触发 `needs_judge = True`
3. `UnifiedEvidenceScorer` 才会并发调用 `judge_rag_document`

当前更容易走到 Judge 的主要是：

1. `LOOKUP`
2. `LOCATION`
3. 带明显 `X 的 Y` 属性结构的问题

这意味着：

1. Judge 不是全量开启
2. Judge 不是默认替代所有规则
3. Judge 是高风险 query 的语义复核层

### 3.5 生成后额外审核层已从代码库移除

这一步同样需要写清楚。

当前生成后的最后一道稳定性机制已经明确收口为：

1. selective judge
2. `EXTRACTIVE -> STRUCTURED -> GENERATIVE -> NO_CONTEXT` 的优雅降级

也就是说，当前系统不再存在额外的生成后审核层。

### 3.6 从“chunk 命中后直接替换全文”修正为“chunk 评分 / 抽取 + 按需全文生成”

这也是这轮很关键的收口。

当前检索时仍然先命中 chunk，但不再把 chunk 直接改写成全文去做后续所有判断。

当前实际数据流是：

1. `rag_service._build_retrieved_candidates()` 保留 `candidate.doc.page_content = chunk_text`
2. 同时额外挂载 `candidate.chunk_text = chunk_text`
3. 再额外挂载 `candidate.full_content = full_content`
4. `UnifiedEvidenceScorer` 构造 `CandidateSnapshot` 时读取 chunk 文本做 `topic_alignment`
5. `ExtractiveGenerator` 读取 chunk 文本做窗口抽取
6. `GenerativeGenerator` 只在 `SUMMARY / OVERVIEW` 场景读取 `full_content`
7. `RELATION` 类型的 Judge 为了减少跨段关系误杀，会优先读取 `full_content`

这样做的原因是：

1. 检索仍然保留 chunk 级命中灵敏度
2. 评分和抽取不再被数万字全文稀释
3. 只有真正需要文档级综合时才切到全文视角
4. 事实型问题和短实体问题的可答性判断更稳定

### 3.7 多实体综合问题做了专项补修，但没有引入新架构

这一轮不是新增第四套控制面，而是在现有三段式主链路里补齐多实体综合问法。

本轮直接覆盖的典型 query 是：

1. `沈见川和陈鹤汀有什么共同点`
2. `文章中提到了哪些人物？他们分别与档案馆有什么关系`

这类问题之前的主要症状是：

1. 意图虽然会被识别成 `RELATION`
2. 但向量命中的单个 chunk 往往只覆盖一个人物或一组局部人物
3. `STRICT + Judge` 在单 chunk 上容易把“跨段可综合”误判成“不可答”
4. 最终直接路由到 `NO_CONTEXT`

本轮已落地的补修包括：

1. `QueryIntentBuilder`
   - 在 `RELATION_MARKERS` 中加入 `共同点`
   - 明确识别 `X 和 Y 有什么共同点` 这类关系综合题
   - 对 `RELATION + 共同点` 这类问法下调到 `DefenseProfile.LOOSE`，避免继续按单点精确问法过度拒答
2. `UnifiedEvidenceScorer`
   - `DefenseProfile.MODERATE` 的 `MIN_TOPIC_ALIGNMENT` 从 `0.14` 下调到 `0.08`
   - `DefenseProfile.LOOSE` 的 `MIN_TOPIC_ALIGNMENT` 从 `0.08` 下调到 `0.05`
   - `DefenseProfile.MODERATE` 的 `MIN_FINAL_SCORE` 从 `0.52` 下调到 `0.45`
   - `direct_evidence` 的无 Judge 兜底条件改成 `topic_alignment >= 0.35 and (title_alignment >= 0.20 or base_relevance >= 0.52)`
   - 对 `RELATION` 类型的 Judge 输入改为优先读取 `full_content`，不再只看单个 chunk
3. `prompt_templates`
   - Judge Prompt 新增“比较两个实体 / 询问共同点 / 询问文中人物及其与对象关系”规则
   - 只要文档中明确给出了至少一侧人物或实体的身份、职责、参与方式、贡献或关系信息，就允许判为 `contains_direct_evidence = yes`
4. `answer_generators`
   - `MULTI_INFO_MARKERS` 新增 `分别`
   - `RELATION / REASON` 默认视为多信息综合题
   - `FACTOID` 里出现 `哪些 / 有哪些 / 什么角色 / 包括 / 包含 / 几个 / 多少 / 分别` 时，也会按多信息问题处理
   - `StructuredGenerator` 对这类问题会扩大上下文上限，并改为读取 `full_content`
   - `ExtractiveGenerator` 支持 top-2 句拼接，减少多信息点被单句截断
5. `AnswerModeRouter`
   - 当前没有单独为“多信息 FACTOID”新增特殊分支
   - 真正的修正点落在 `multi_document_synthesis`、`structured_context_limit` 和 `structured_use_full_content`
   - 也就是不改架构，只在已有评分、生成和路由边界内补能力

这一轮的关键结论是：

1. 多实体综合问题的核心矛盾不是“模型不会答”
2. 而是“单 chunk 级直接证据判定不适合跨段综合问题”
3. 所以当前修法仍然落在既有架构里：意图识别、统一评分、模式路由三层各补一点，而不是新增独立分支

### 3.8 入库前新增轻量语义锚点注入

这一项已经落进当前索引链路，不是实验分支。

当前代码位置：

- `backend/app/services/rag/vector_index_service.py`

当前做法：

1. 文档先按 Markdown 标题切分
2. 再按 `chunk_size=1000`、`chunk_overlap=200` 递归切块
3. 入库前调用 `_extract_core_entities(title, first_paragraph)`
4. 使用 `jieba.posseg` 提取标题和首段中的名词
5. 过滤 `文档 / 资料 / 介绍 / 内容 / 情况 / 背景 / 历史 / 概况 / 总结 / 概述 / 说明 / 方案 / 测试` 等停用词
6. 取权重最高的 1 到 3 个实体，生成前缀 `【核心主题：实体1、实体2】`
7. 把这个前缀注入每个 chunk 的 `page_content` 头部，再写入向量库

这一项的目标不是生成更长文本，而是给召回侧补一个轻量实体锚点，减少：

1. 文档只顺带提到某个地点或人物却被错误命中
2. 标题主题和 chunk 内局部句子主题错位
3. “新加坡的风景”这类顺带提及误召回

### 3.9 Prompt 已去领域特调，保留通用防御逻辑

这一轮还做了一次必须写进文档的收口：

1. `build_general_rag_prompt()` 已删除“馆长年龄”“渔业”“沿海”“档案馆、论文、新闻”等特定领域示例
2. 当前只保留两条通用约束：
   - 概括类问题先回答用户真正问的对象
   - 参考上下文如果只是顺带提到实体、没有直接讨论用户问的主题或关系，必须拒答
3. `_history_paragraph_score()` 已从领域词改成通用历史词：
   - 年份模式：`\d{4}年`
   - 通用标志词：`最早 / 建于 / 创立 / 成立 / 起步 / 发展 / 后来 / 直到 / 最初 / 早期`

这一步的意义是：

1. 保护当前系统在本地知识库上有效
2. 但不靠对《雾潮镇档案馆》文本做硬编码特调
3. 方便后续把同一条链路拿去 RGB 这类公共数据集做对比

### 3.10 已补齐“本地知识库 + RGB 公共基准”的双数据集评测链路

为了避免只在本地语料上自证，本轮已经补了独立对比评测框架：

- `llamaindex_rag_eval/compare_rag_systems.py`

当前会对比：

1. 自研 RAG
2. 参照实现的 LlamaIndex RAG

当前已落地的输出包括：

1. `llamaindex_rag_eval/outputs/rag_comparison_summary.json`
2. `llamaindex_rag_eval/outputs/rgb_rag_comparison_summary.json`
3. `llamaindex_rag_eval/outputs/dataset_comparison_summary.svg`

当前一组已保存结果显示：

1. 本地知识库：
   - 自研 RAG：`faithfulness 1.0`、`answer_relevancy 0.8112`、`accuracy 0.3`
   - LlamaIndex：`faithfulness 0.85`、`answer_relevancy 0.8545`、`accuracy 0.2`
2. RGB：
   - 自研 RAG：`faithfulness 0.92`、`answer_relevancy 0.8113`、`context_precision 0.93`、`accuracy 0.8033`
   - LlamaIndex：`faithfulness 0.8517`、`answer_relevancy 0.6429`、`context_precision 0.7333`、`accuracy 0.59`
3. 代价也要诚实写清楚：
   - RGB 上自研 RAG 的 `P95` 延迟约 `3597.06ms`
   - LlamaIndex 对照约 `1412.95ms`

## 4. 主链路是怎么一步步演进到现在的

### 4.1 第一阶段：只有向量召回

最早的链路非常直接：

1. 文档切块入库
2. query 直接向量搜
3. 把召回结果塞给大模型

这个阶段的问题很典型：

1. 编号、标题、短实体命中不稳
2. 候选一偏，生成就跟着偏
3. 没有稳定的拒答边界

### 4.2 第二阶段：加 BM25 和 rerank

第二阶段解决的是：

`仅靠 dense 不够，仅靠召回顺序也不够。`

这个阶段补进了：

1. 文档级 BM25
2. adaptive hybrid
3. cross-encoder rerank

这一阶段的目标是：

1. 先把候选找全
2. 再把候选找准

### 4.3 第三阶段：补边界控制

随着系统开始能回答之后，问题从“答不上来”变成了“答得像，但主题不对”。

暴露出来的主要类型有：

1. 超短实体误召回
2. 关系型只命中单边实体
3. `X 的 Y` 属性型只命中主体，不命中属性
4. 文档顺带提到实体，但核心主题并不在回答用户问题

这一阶段我们尝试过较多规则和回答后兜底。

### 4.4 第四阶段：出现“补丁叠补丁”迹象

旧链路最大的问题不是单个模块没用，而是：

1. 判断散在各处
2. 同类信号在多处重复计算
3. 主函数 if-else 变长
4. 讲不清“到底是谁决定可答性”

这也是后来决定做强制收口的原因。

### 4.5 第五阶段：收口成当前运行时架构

当前这一版的核心变化有三点：

1. 用 `QueryIntentBuilder` 统一意图和风险画像
2. 用 `UnifiedEvidenceScorer` 统一证据可用性判断
3. 用 `AnswerModeRouter` 统一回答模式

这一步的收益不是“多加了一个模块”，而是把原来散在各处的决策权收拢了。

## 5. 当前系统分别如何兜住典型错误

### 5.1 超短实体或超短查找类问题

典型类型：

1. 单个实体名
2. 很短的标题式问题
3. 依赖精确命中的短查找

当前处理方式：

1. `QueryIntentBuilder` 把它归到 `LOOKUP` 或短 query
2. `defense_profile = STRICT`
3. `UnifiedEvidenceScorer` 对 `LOOKUP` 使用更高的 topic alignment 和 base relevance 门槛
4. 若标题对齐和主题对齐太弱，会打上 `FAILED_SHORT_ENTITY_MATCH`
5. 这类 query 还会触发 selective judge

对应收益：

1. 短实体更容易拒掉“看起来像”的误召回
2. 不再只靠 rerank 分高低决定可答性

### 5.2 关系型问题

典型类型：

1. `A 与 B 的关系`
2. `A 和 B 的区别`
3. `A 和 B 的联系`

当前处理方式：

1. `QueryIntentBuilder` 识别为 `RELATION`
2. `retrieval_depth` 提高
3. `evidence_requirement = MULTI_SPAN`
4. `defense_profile = STRICT`
5. `needs_judge = True`
6. 如果 usable 文档不止一篇，`AnswerModeRouter` 直接路由到 `GENERATIVE`

对应收益：

1. 关系型问题不会再被单句硬抽取误导
2. 也不会把只沾到单边实体的文档轻易送进最终回答

### 5.3 `X 的 Y` 属性错配

典型类型：

1. `新加坡的档案馆`
2. `苹果的创始人`
3. `某地的风景`

这类问题的难点是：

1. 文档里可能同时出现 `X`
2. 文档里也可能在讲 `Y`
3. 但 `Y` 并不是 `X` 的属性，而是别的主体的属性

当前处理方式：

1. `QueryIntentBuilder` 会把属性型 query 拉到更严格的防御画像
2. `UnifiedEvidenceScorer` 先做 topic alignment
3. 高风险候选再走 LLM Judge
4. Judge 只做结构化判别，不直接自由生成
5. Judge 不通过时直接 `usable = False`

对应收益：

1. 可以拦住“实体顺带出现，但主题主体错了”的问题
2. 这也是当前替代旧 `universal defense` 说法的关键点

### 5.4 精确事实问题

典型类型：

1. `正式名称是什么`
2. `建于哪一年`
3. `地址是什么`
4. `今年多少岁`

当前处理方式：

1. 检索先用 chunk 命中把候选找出来
2. `UnifiedEvidenceScorer` 在 chunk 上计算 `topic_alignment` 和 `direct_evidence`
3. `AnswerModeRouter` 在高置信单文档下路由到 `EXTRACTIVE`
4. `ExtractiveGenerator` 先尝试直接使用 Judge 的 `evidence_quote`
5. 若没有，再对 chunk 做段落窗口和句子窗口抽取
6. 如果 top-2 句得分接近，会尝试拼接两句，减少“角色 / 材料 / 列举项”丢失
7. 抽取失败时优雅降级到 `STRUCTURED`

对应收益：

1. 不会因为抽不到一句完美原话就整条链路崩掉
2. 也不会为了抽取而引入复杂正则和重型 NLP 解析

### 5.5 概况 / 历史 / 背景类问题

典型类型：

1. `雾潮镇的历史`
2. `某个对象的概况`
3. `某主题的背景`

当前处理方式：

1. `QueryIntentBuilder` 会判成 `SUMMARY` 或 `OVERVIEW`
2. `evidence_requirement = FULL_DOCUMENT`
3. `defense_profile = LOOSE`
4. `AnswerModeRouter` 直接路由到 `GENERATIVE`
5. `GenerativeGenerator` 在这两类问题上读取 `full_content`
6. 生成上下文上限从 `3` 提到 `5`
7. 段落窗口会优先保留历史 / 概况相关段落，而不是只取单段最高分

对应收益：

1. 这类问题不再硬往单句抽取上塞
2. 问题和回答模式的匹配关系更稳定

## 6. 当前真正已从代码库移除或降级的旧逻辑

这一节是为了避免后续文档再次回写旧口径。

### 6.1 已从代码库移除或彻底淘汰的逻辑

1. `AgentService` 作为主 RAG 执行入口
2. 历史检索增强模块
3. 生成后额外审核层
4. 用大量硬编码 query 规则驱动整个主流程的做法

### 6.2 不应再写成“当前运行态”的旧术语

1. `AgentService.rag_qa`
2. `universal defense 是当前唯一主防线`
3. `额外生成后审核层是默认运行路径`
4. `历史检索增强模块仍是默认主路径的一部分`

## 7. 当前版本的主要收益

如果只总结成几句话，当前这版收益是：

1. 主链路更容易讲清楚了，不再是防御逻辑叠罗汉。
2. 主体贡献重新收回到了检索、评分和路由三段主链上。
3. 高风险 query 有 selective judge，但正常 query 不会被全量拖慢。
4. 事实型问题可以优先抽取，抽取失败还能平滑降级。
5. 主链路的可观测性更好，评分器会输出结构化 trace。

## 8. 当前仍需诚实说明的边界

### 8.1 Judge 不是万能的

Judge 解决的是高风险 query 的语义错配问题，不是替代检索。

如果前面候选根本找错了，Judge 只能拒答，不能把答案凭空找出来。

### 8.2 规则仍然存在，但不再散落

当前不是“彻底没有规则”，而是：

1. 规则更多内化为意图识别和统一评分门槛
2. 而不是散落在主流程 if-else 里

### 8.3 生成后不再存在额外审核层

当前如果还要补生成后审核，应该重新设计一套独立、可观测、可验证的后处理机制，而不是把旧的历史逻辑原样接回。

## 9. 当前答辩时应该怎么表述这轮调整

最稳的一句话是：

`最近一次重构不是继续叠补丁，而是把旧的补丁式防御收口成了 QueryIntentBuilder、UnifiedEvidenceScorer 和 AnswerModeRouter 三段式主链路。`

再补一句就够：

`LLM Judge 已经采纳，但只在高风险原子证据型 query 上选择性启用；生成阶段的稳定性来自模式路由和优雅降级，不再依赖额外审核层。`
