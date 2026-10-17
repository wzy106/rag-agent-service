# 25. 为什么要加 Rerank？

**问题**：混合检索（向量 + BM25 + RRF）是“双塔 + 关键词”的粗排——快但精度有限。Top-3 里可能混进语义相关但实际帮助不大的 chunk。

**方案**：用 `bge-reranker-base` 交叉编码器对混合检索召回的 Top-20 精排到 Top-3。

**双塔 vs 交叉编码器**：

| 维度 | 双塔（bge-small） | 交叉编码器（bge-reranker） |
|---|---|---|
| 结构 | query 和 doc 分别编码 | (query, doc) 拼一起送进模型 |
| 速度 | 快（毫秒级） | 慢（逐对推理，CPU 上 20 候选约 2~7 秒，负载高可达十几秒） |
| 精度 | 粗 | 精 |
| 用途 | 大规模召回 | 小规模精排 |

**关键设计：两阶段检索**
1. **粗排**：混合检索召回 Top-20（保证召回率）
2. **精排**：Reranker 打分保留 Top-3（保证精确率）

**为什么不能只用 Reranker**：要对每条 (query, doc) 跑一次模型。当前语料约 600 个 chunk → 600 次推理，即使按乐观的 0.65 秒/条也要 6 分钟/查询——太慢。所以先用双塔快速筛到 Top-20（`RAG_RECALL_K` 可配），再用 Reranker 精排。

**效果**：
- 具体召回率/MRR/nDCG 以根目录 `README.md` 和 `eval/output/retrieval_v3.json` 为准。
- 评估时同时看 candidate recall@20 和 final recall@3：如果 candidate 高、final 低，瓶颈在 Rerank；反过来才是粗排召回不足。

**诚实的局限**：
- 267 条查询 × 20 候选 = 5340 次推理，CPU 上精排 20 候选约 2~7 秒（本机实测），267 条约 10~30 分钟。
- 高并发场景需要优化（更小模型 / GPU / ONNX 量化）。

**面试话术**：
> “我加了两阶段检索：混合检索召回 Top-20，用 bge-reranker-base 交叉编码器精排 Top-3。Reranker 是懒加载的，第一次调用才加载模型，不拖慢服务启动。效果我用 source/chunk 双粒度、candidate/final 分阶段指标来评，具体数字见 README。性能上我会诚实说：CPU 精排 20 候选约 2~7 秒，高并发得换更小的 reranker 或上 ONNX 量化。”
