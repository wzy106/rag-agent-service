# 真实使用日志

## 2026-10-03
- 问："我的项目用了什么检索策略？"
- 答：准确命中 PROJECT_NOTES.md，返回 MMR、k=3、fetch_k=10
- 评价：完全正确，无需改进

## 2026-10-17
- 重建索引：98 篇 / 617 chunk，全部带 chunk_id + section_path
- 生成 v3 测试集：89 篇文档、267 条问题、24 条多 evidence（因语料二次修订，生成过一次）
- DeepSeek 调用：测试集生成 2 × 98 次，约 ¥0.4–1.0
- 端到端评估：100 次回答 + 100 次 judge，约 12 万 token，估计 < ¥0.5
- 评估结果：`eval/output/retrieval_v3.json`、`calibration_v3.json`、`e2e_summary.json`
