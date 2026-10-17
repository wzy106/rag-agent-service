# 9. MMR 是什么？

**全称**：Maximal Marginal Relevance（最大边际相关性）。

**公式**：
```
MMR = λ × Sim(query, chunk) - (1-λ) × max(Sim(chunk, 已选chunk))
```

**关键理解**：
- MMR 在 **chunk 级别**做多样性，不是文档级别
- 它避免的是“内容重复”，不是“来源重复”
- 同一个文档里的两个差异大的 chunk，MMR 也会都选
- 想保证“Top-3 来自不同文档”，需要额外按文档去重

**v3 测试集上的表现**：
- MMR chunk any 36.0%（低于相似度 48.7%）——**在“精确匹配”任务上吃亏**
- 多样性 2.79（高于相似度 2.52）—— **多样性导向是对的**
- source strict 53.2%，只报 source 会掩盖 chunk 级差异
