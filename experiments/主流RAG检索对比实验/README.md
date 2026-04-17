# 主流RAG检索对比实验

这个目录存放当前这轮可对外讲述的主实验材料，重点不是“弱基线刷分”，而是：

1. 用公开中文 retrieval benchmark 做可复现对比。
2. 把检索改进放在算法链路，而不是只调提示词。
3. 同时保留结果、流程、排障、消融和复试表达材料。

## 1. 核心入口

- [方法改进与结果解读.md](/root/Velo/experiments/主流RAG检索对比实验/方法改进与结果解读.md)
  主文档。讲方法、公式、指标、结果、消融结论和答辩口径。
- [详细流程与方法说明.md](/root/Velo/experiments/主流RAG检索对比实验/详细流程与方法说明.md)
  讲实验为什么重做、脚本实际怎么跑、工程上做了哪些处理。
- [真实执行与排障日志.md](/root/Velo/experiments/主流RAG检索对比实验/真实执行与排障日志.md)
  讲真实执行、恢复、排障、消融协议重做和图表修复过程。
- [主流RAG方案对比实验.ipynb](/root/Velo/experiments/主流RAG检索对比实验/主流RAG方案对比实验.ipynb)
  自动生成的结果展示 Notebook。

## 2. 配套材料

- [复试答辩话术.md](/root/Velo/experiments/主流RAG检索对比实验/复试答辩话术.md)
  1 分钟版、3 分钟版和常见追问回答。
- [复试简历写法.md](/root/Velo/experiments/主流RAG检索对比实验/复试简历写法.md)
  简历标题、项目描述和复试展开口径。
- [消融实验说明.md](/root/Velo/experiments/消融实验_模块贡献分析/消融实验说明.md)
  DuRetrieval 去模块消融说明。

## 3. 原始输出

- [outputs](/root/Velo/experiments/主流RAG检索对比实验/outputs)
  主实验的 CSV、JSON、SVG 图表和配置文件。
- [/root/Velo/experiments/消融实验_模块贡献分析/outputs](/root/Velo/experiments/消融实验_模块贡献分析/outputs)
  消融实验的 CSV、JSON、SVG 图表和配置文件。

## 4. 对比链路

- 主流 RAG 方案 A：`Dense Retrieval + Rerank`
- 主流 RAG 方案 B：`Hybrid Retrieval + RRF + Rerank`
- 我们的方案：`Adaptive Hybrid + Multi-signal Fusion + Rerank`

## 5. 当前最稳结论

1. 相对主流方案 B，我们的方案在 `DuRetrieval` 和 `T2Retrieval` 上都保持正提升。
2. 去模块消融显示，当前主要贡献来自“查询自适应权重”和“混合候选池”。
3. 覆盖率奖励有小幅正贡献。
4. 标识符约束在当前公开 benchmark 上没有显示稳定正收益，不能硬讲成核心贡献。
5. 这套检索思路已经接入 backend 实际 RAG 主链路，不再只停留在 `experiments/` 目录。

## 6. 运行命令

主实验：

```bash
/root/Velo/.venv/bin/python /root/Velo/experiments/mainstream_rag_benchmark.py
```

消融实验：

```bash
/root/Velo/.venv/bin/python /root/Velo/experiments/ablation_study.py
```
