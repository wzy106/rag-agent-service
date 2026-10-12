# 2. RAG 的完整流程？

1. 文档加载（Document）
2. 切分（chunking）
3. Embedding（向量化）
4. 存入向量库
5. 用户提问 → 向量化 → 相似度检索 Top-K
6. 检索结果拼进 Prompt
7. 模型生成回答

**进阶版（你项目用的）**：
1. 文档加载 → 2. 切分 → 3. Embedding → 4. 存入 FAISS
5. 用户提问 → **混合检索召回 Top-20（向量 + BM25 + RRF）**
6. **Rerank 精排 Top-3（CrossEncoder）**
7. **降级判断（距离阈值 0.96）**
8. 检索结果拼进 Prompt → 模型生成回答
