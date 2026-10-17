# 十三、RAG 评估（面试重点）

### 评估方法（v3）

1. 遍历索引里的 chunk，按 `source` 分组
2. 每篇文档取最长的 2 个 chunk
3. 调 DeepSeek 生成 3 个问题，并让模型标注 `evidence`（哪些 chunk 真正包含答案）
4. 写入 `evidence_chunk_ids`，同时保留 `expected_sources`
5. 用 3-gram Jaccard ≥ 0.8 扩展近重复块，生成 `gold_chunk_ids` / `gold_sources`（多金标）
6. 检索指标同时看 source strict/relaxed、chunk any/all、candidate recall@20、final recall@3
7. 端到端用 LLM-as-judge 评 faithfulness / relevance / refusal

为什么不能只标 source：

- 同一文档任意 chunk 命中就算对 → 高估
- 只认生成问题的那一个 chunk → 语料里另一篇文档有等价答案时算错 → 低估
- 所以必须双粒度 + 多金标，并同时报告

### 随机基线（组合公式精确计算）

```python
from math import comb
from collections import Counter

def random_baseline(test_set, source_counts, k=3):
    n_total = sum(source_counts.values())
    total = 0.0
    for item in test_set:
        n_expected = sum(source_counts.get(s, 0) for s in item["gold_sources"])
        total += 1 - comb(n_total - n_expected, k) / comb(n_total, k)
    return total / len(test_set)
```

随机基线用于判断测试集是否有判别力，不用于证明策略“好”。

### 分阶段指标

| 指标 | 回答什么问题 |
|---|---|
| candidate recall@20 | 粗排有没有把正确 chunk 捞进来 |
| final recall@3 | 精排后有没有留下来 |
| chunk precision | Top-3 里有多少是 gold |
| MRR / nDCG | 正确 chunk 排得多前 |
| source relaxed | 近重复文档命中是否被正确计入 |
| faithfulness | 回答是否真的基于检索资料 |
| refusal accuracy | 资料不足时是否拒答 |

如果 candidate 很高、final 很低，瓶颈在 Rerank；反过来才是粗排召回不足。

### 统计口径

- 二值指标用 Wilson 95% CI
- 连续指标用 bootstrap CI
- 策略对比用 paired bootstrap，报告差值 CI，不要只说“显著提升”

### 关键发现（定性）

1. **语料扩充后，随机基线大幅下降，策略差异才变得可辨**。旧的小语料有天花板效应。
2. **Rerank 主要提升召回率**：把 Top-20 里被埋没的相关 chunk 顶上来。
3. **混合检索优于纯相似度**：BM25 补上了关键词匹配。
4. **MMR 在精确匹配任务上吃亏**：多样性导向会牺牲相关性。
5. **source 级和 chunk 级会给出不同结论**：必须同时报告，不能只报一个。
6. **检索对不等于回答对**：端到端 faithfulness 可能明显低于检索命中率。

具体数字统一以根目录 `README.md` 和 `eval/output/*.json` 为准。

### 失败案例分析

**案例 1**：口语化表达
- 用户问：“怎么让回答一个字一个字显示出来？”
- 期望文档：`readme/04_启动.md`
- 实际可能命中：`notes_interview/q05_流式输出怎么实现的.md`
- 原因：口语化表达和文档术语有语义鸿沟；多金标可以部分修正

**案例 2**：跨主题问题
- 用户问：“多 Agent 系统怎么保证消息不丢？”
- 期望文档：`stories/18_故事_17双源记忆打架架构重构.md`
- 实际可能命中：`notes_interview/q12_Agent_多轮对话怎么实现.md`
- 原因：语义相近但主题不同；source 级标签偏窄

### 已知不足

**评估层面**：
- 语料是项目自述，测试集也由这些 chunk 生成；多金标不能消除外部效度问题
- 阈值 0.96 是检索层接受率，不是端到端幻觉率
- 阈值只 gate 检索，不 gate 生成；Agent 仍可能继续生成
- 端到端评估是直接 RAG 链路，不覆盖 MCP 工具选择和多轮改写
- judge 默认与回答模型相同，存在自评偏好

**架构层面**：
| 位置 | 问题 | 生产级解法 |
|---|---|---|
| `/chat` 和 `/chat/stream` | pymysql 同步客户端阻塞事件循环 | 用 `aiomysql` |
| `save_conversation` | 每次请求新建连接，无连接池 | 用连接池 |
| 流式状态机 | 布尔变量表达状态，边界情况可能出错 | 显式状态机 + 测试 |
| Reranker 性能 | CPU 精排 20 候选约 2~7 秒 | 更小模型 / GPU / ONNX 量化 |
| 依赖 | 只有下界，没有 lockfile | 锁版本 |

### 改进方案

| 方案 | 状态 |
|---|---|
| 混合检索（向量 + BM25 + 加权 RRF） | 已做 |
| 降级策略 | 已做（但只 gate 检索） |
| 多指标评估 | 已做 |
| 随机基线 | 已做 |
| 阈值标定（easy + hard 负样本） | 已做 |
| Rerank 精排 | 已做 |
| chunk 级 evidence 标注 | 已做 |
| 多金标（近重复扩展） | 已做 |
| 端到端 LLM-as-judge | 已做 |
| 真实用户查询 holdout | 待做 |
| graph 层拒答 / 充分性判断 | 待做 |

### 面试答题模板

> “我的评估不是只报一个召回率。测试集由脚本从 chunk 生成，但我让模型标注 evidence chunk，并用近重复扩展成多金标，因为语料里同一结论经常出现在多篇文档。指标上我同时报 source strict/relaxed、chunk any/all、candidate recall@20 和 final recall@3，并用 Wilson/bootstrap 给置信区间，策略对比用 paired bootstrap。检索之外我还跑了端到端 faithfulness/relevance/refusal，做‘检索命中 × 回答忠实’的 2×2 交叉表——因为检索对了不等于回答对了。阈值 0.96 我也会说明口径：它是检索层接受率，不是幻觉率；而且它只看向量 top-1、在 BM25 融合之前判断，近域但库外的问题仍可能漏过去。所有数字都能用仓库里的脚本复现。”
