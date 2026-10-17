# 24. `MAX_DISTANCE` 阈值怎么定的？

**不是拍脑袋定的，是标定出来的；但要知道它衡量的是什么。**

**方法**：
1. 正样本：从测试集抽样（默认 100 条，seed=42）
2. 负样本分两组：
   - easy：`eval/negatives.json`（天气、菜谱、电影）
   - hard：`eval/negatives_hard.json`（版本号、新特性、价格等近域但库内没有的问题）
3. 对每条问题算 `best_distance`（top-1 的 L2 距离）
4. 扫 101 个候选阈值（0.50~1.50），算 Precision / Recall / F1
5. 分别报告 easy / hard / combined 的 FPR

**口径（重要）**：
- `FPR = FP / (FP + TN)`：负样本被接受的比例
- `FDR = FP / (TP + FP)`：被接受的结果里负样本占的比例，依赖抽样基率
- 旧文档把 `1 - precision` 写成“幻觉率 1%”，这是**不准确**的；它不是端到端幻觉率
- 阈值只判断“检索层是否接受 query”，不判断“模型回答是否忠实”

**当前限制**：
- 只看向量 top-1，且在 BM25/RRF 融合之前判断；关键词强相关但向量距离大的 query 会被提前拒绝
- 近域但库内没有的问题可能通过阈值，进入生成阶段
- 真正的拒答需要在生成层做充分性判断或 graph 分支

**产出**：
- `eval/calibrate_distance.py`
- `eval/negatives.json` / `eval/negatives_hard.json`
- `eval/output/calibration_v3.json`
- `eval/output/distance_distribution.png` / `pr_curve.png`

**面试话术**：
> “阈值不是拍脑袋定的。我用正样本和两组负样本（明显离题 + 近域但库内没有）扫 PR 曲线，当前取 0.96。但我会明确说：这个阈值衡量的是检索层接受率，不是幻觉率；而且它只看向量 top-1、在 BM25 融合之前判断，所以近域但库外的问题仍可能漏过去。真正的拒答要在生成层做充分性判断。”
