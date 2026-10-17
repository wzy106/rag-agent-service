# 14. RAG 评估用什么指标？

**检索层：source + chunk 双粒度**

| 指标 | 含义 |
|---|---|
| source strict | Top-K 命中生成问题的那篇文档 |
| source relaxed | Top-K 命中多金标扩展后的任一等价文档 |
| chunk any | Top-K 命中任一 gold chunk |
| chunk all | Top-K 覆盖全部 evidence chunk（多证据问题） |
| candidate recall@N | 粗排 Top-N 是否命中 gold chunk |
| final recall@3 | 精排后 Top-3 是否命中 |
| precision / MRR / nDCG | 以 chunk 为相关性单位 |
| 随机基线 | 从 N 个 chunk 随机抽 K 个的命中概率 |

**为什么要双粒度**：
- 只看 source 会把“命中文档但没命中正确 chunk”算对，高估。
- 只认生成问题的那一个 chunk，会把“另一篇文档里的等价答案”算错，低估。语料里同一结论常出现在 notes、notes_interview、stories 多处。
- 所以用 `evidence_chunk_ids` + 近重复扩展的 `gold_chunk_ids` / `gold_sources`，同时报告 source 和 chunk。

**生成层：端到端指标**
- faithfulness：回答中的事实性陈述是否都能在检索资料中找到依据
- relevance：回答是否切题
- refusal accuracy：资料不足时是否正确拒答
- 2×2 交叉表：检索命中 × 回答忠实，用来区分“检索背锅”还是“生成背锅”

**统计口径**
- 二值指标用 Wilson 95% CI
- 连续指标用 bootstrap CI
- 策略对比用 paired bootstrap，报告差值 CI，不要只说“显著提升”

具体数字以根目录 `README.md` 和 `eval/output/*.json` 为准。
