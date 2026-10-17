# RAG 评估

当前评估口径是 **source + chunk 双粒度 + 多金标 + 端到端生成评估**。具体实测数字统一以根目录 `README.md` 的「RAG 评估」章节和 `eval/output/*.json` 为准，本页只描述方法与口径，避免多处数字漂移。

## 测试集

- v3 测试集：每篇文档取 2 个最长 chunk，调 DeepSeek 生成问题，并让模型标注 `evidence`（哪些 chunk 真正包含答案）。
- 每条样本同时保存 `expected_sources`（生成来源文档）和 `evidence_chunk_ids`（证据块）。
- 语料内部有大量重复/近重复内容（同一结论散落在 notes、notes_interview、stories 中），所以 `scripts/label_gold.py` 会把 3-gram Jaccard ≥ 0.8 的近似块扩展为 `gold_chunk_ids` / `gold_sources`。这就是“多金标”：命中另一篇文档里的等价答案，不再算错。
- 同时输出 0.6 阈值的敏感度统计，主指标仍用 0.8。

## 检索指标

| 指标 | 口径 |
|---|---|
| source strict | Top-K 命中 `expected_sources` |
| source relaxed | Top-K 命中 `gold_sources`（含近重复来源） |
| chunk any | Top-K 命中任意 `gold_chunk_ids` |
| chunk all | Top-K 覆盖 `evidence_chunk_ids` 的全部块（多证据问题） |
| candidate recall@N | 粗排 Top-N（默认 20）是否命中 gold chunk |
| final recall@3 | 精排后 Top-3 是否命中 |
| precision / MRR / nDCG | 以 chunk 为相关性单位 |
| diversity | Top-K 中不同 source 的数量 |

`candidate recall@N` 和 `final recall@3` 要分开看：如果 candidate 很高、final 很低，瓶颈在 Rerank；反过来才是粗排召回不足。

## 置信区间与显著性

- 二值指标（召回率等）用 Wilson 95% CI。
- 连续指标用 bootstrap 95% CI。
- 策略对比用 paired bootstrap（同一批 query 配对重采样），报告差值 CI 和 p 值。不要在没有区间的情况下说“显著提升”。

## 随机基线

从所有 chunk 里随机抽 K 个，命中至少一个 gold chunk 的概率：

```text
P(hit) = 1 - C(N - n_gold, K) / C(N, K)
```

source 级和 chunk 级各算一套。随机基线用于判断测试集是否有判别力，不用于证明策略“好”。

## 端到端评估

`eval/eval_e2e.py` 跑“检索 + 生成 + LLM-as-judge”：

- 80 条可回答问题 + 20 条负样本（easy/hard 混合）。
- 生成：只给检索到的 Top-3，要求“只依据资料、资料不足就拒答、结论标来源”。
- Judge 输出：`context_sufficient`、`faithful`、`relevant`、`refusal_appropriate`、`unsupported_claims`。
- 汇总：faithfulness、relevance、refusal accuracy、abstention rate、unsupported-claim rate、延迟、token 用量。
- 额外输出“检索命中 × faithful”的 2×2 交叉表。

它是**直接 RAG 链路**评估，不包含 MCP 工具选择、多轮记忆和 Agent 循环；这些属于另一层问题，不要混在一起报。

## 复现

```bash
# 重建索引（文档或切分代码变更后）
python scripts/build_index.py --force
python scripts/build_index.py --check

# v3 测试集 + 多金标
python scripts/generate_testset.py --output eval/test_set_v3.json
python scripts/label_gold.py --test-set eval/test_set_v3.json

# 检索评估
python eval/eval_retrieval.py \
  --test-set eval/test_set_v3_gold.json --k 3 --bootstrap 2000 \
  --json eval/output/retrieval_v3.json

# 阈值标定（easy + hard 负样本）
python eval/calibrate_distance.py \
  --positives eval/test_set_v3_gold.json \
  --negatives eval/negatives.json eval/negatives_hard.json \
  --output eval/output/calibration_v3.json

# 端到端评估
python eval/eval_e2e.py \
  --test-set eval/test_set_v3_gold.json \
  --negatives eval/negatives.json eval/negatives_hard.json \
  --answerable 80 --negative 20 --seed 42 --concurrency 3
```

## 已知限制

- 语料是项目自述（98 篇笔记/面试题/踩坑故事），测试集也由这些 chunk 生成。多金标只能修正“同一答案在别的文档”，不能消除“自己考自己”的外部效度问题。
- 阈值 `MAX_DISTANCE=0.96` 衡量的是检索层是否接受 query，**不是**端到端幻觉率；它用 top-1 向量距离、且在 BM25/RRF 之前判断，近域但库内没有的问题可能通过。
- 端到端评估是直接链路，不覆盖 Agent 的工具选择、拒答策略和多轮改写。
