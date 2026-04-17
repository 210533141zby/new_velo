# Experiments

这个目录存放独立于前后端工程的实验材料。现在这里的重点已经不是业务代码，而是 RAG 检索方案的公开 benchmark 与实验记录。

当前正式 benchmark 已完成，最终结果目录是：

- [主流RAG检索对比实验](/root/Velo/experiments/主流RAG检索对比实验)

## 当前重点文件

- [mainstream_rag_benchmark.py](/root/Velo/experiments/mainstream_rag_benchmark.py)
  主实验脚本。负责公开数据集下载、预处理、检索、重排、指标统计、图表和 Notebook 生成。
- [.cache/public_benchmarks](/root/Velo/experiments/.cache/public_benchmarks)
  公开 benchmark parquet 文件和 embedding 缓存目录。
- [主流RAG检索对比实验](/root/Velo/experiments/主流RAG检索对比实验)
  正式实验目录，包含 Notebook、图表、原始结果、方法分析、真实执行日志和详细流程文档。
- [消融实验_模块贡献分析](/root/Velo/experiments/消融实验_模块贡献分析)
  模块贡献消融实验目录。当前用于补齐“为什么有效”的归因分析。

## 建议阅读顺序

如果你想先看懂整体，不要一上来就看脚本，按下面顺序更容易：

1. [详细流程与方法说明.md](/root/Velo/experiments/主流RAG检索对比实验/详细流程与方法说明.md)
2. [方法改进与结果解读.md](/root/Velo/experiments/主流RAG检索对比实验/方法改进与结果解读.md)
3. [真实执行与排障日志.md](/root/Velo/experiments/主流RAG检索对比实验/真实执行与排障日志.md)
4. [主流RAG方案对比实验.ipynb](/root/Velo/experiments/主流RAG检索对比实验/主流RAG方案对比实验.ipynb)
5. [消融实验说明.md](/root/Velo/experiments/消融实验_模块贡献分析/消融实验说明.md)
6. [mainstream_rag_benchmark.py](/root/Velo/experiments/mainstream_rag_benchmark.py)

## 运行方式

正式实验：

```bash
/root/Velo/.venv/bin/python /root/Velo/experiments/mainstream_rag_benchmark.py \
  --datasets du t2 \
  --output-root /root/Velo/experiments/主流RAG检索对比实验 \
  --embedding-model nomic-embed-text:latest
```

快速验证：

```bash
/root/Velo/.venv/bin/python /root/Velo/experiments/mainstream_rag_benchmark.py \
  --datasets du \
  --output-root /root/Velo/experiments/快速验证_主流RAG检索对比 \
  --query-limit 20 \
  --corpus-limit 4000 \
  --embedding-model nomic-embed-text:latest
```
