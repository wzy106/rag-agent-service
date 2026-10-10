"""
MAX_DISTANCE 阈值标定脚本

思路：
- 正样本（来自 test_set.json）：知识库中应该能命中
- 负样本（来自 negatives.json）：知识库中明显没有
- 对每条问题用 similarity_search_with_score 拿 best_distance
- 画正负样本距离分布直方图，找分离点
- 扫 MAX_DISTANCE 从 0.5 到 1.5，画 PR 曲线
- 找 F1 最大点 + Precision >= 0.95 的最高召回点

跑法：
    cd D:/Projects/Agent
    python eval/calibrate_distance.py

输出：
    eval/output/distance_distribution.png
    eval/output/pr_curve.png
    命令行统计表 + 推荐阈值
"""
import os
import sys
import json
from pathlib import Path

# ===== 确保能找到项目根的 rag_core 包 =====
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_core.search import load_vectorstore, LocalEmbeddings, DEFAULT_MODEL


# ===== 加载数据 =====
print("加载 embedding 和 FAISS 索引...")
embedding = LocalEmbeddings(DEFAULT_MODEL)
vectorstore = load_vectorstore(embedding=embedding)

EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EVAL_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

with open(EVAL_DIR / "test_set.json", encoding="utf-8") as f:
    positives = [item["question"] for item in json.load(f)]

with open(EVAL_DIR / "negatives.json", encoding="utf-8") as f:
    negatives = json.load(f)["questions"]

print(f"正样本：{len(positives)} 条")
print(f"负样本：{len(negatives)} 条\n")


# ===== 计算 best_distance =====
def best_distance(query: str) -> float:
    """返回 top-1 的 L2 距离（越小越相关）"""
    results = vectorstore.similarity_search_with_score(query, k=1)
    return results[0][1] if results else float("inf")


print("计算正样本距离...")
pos_dists = [best_distance(q) for q in positives]

print("计算负样本距离...")
neg_dists = [best_distance(q) for q in negatives]


# ===== 统计 =====
def stats(name, dists):
    dists_sorted = sorted(dists)
    n = len(dists_sorted)
    return {
        "name": name,
        "min": dists_sorted[0],
        "max": dists_sorted[-1],
        "mean": sum(dists_sorted) / n,
        "median": dists_sorted[n // 2],
    }

pos_stats = stats("正样本", pos_dists)
neg_stats = stats("负样本", neg_dists)

print("\n" + "=" * 70)
print("距离分布统计")
print("=" * 70)
print(f"{'类别':<10} {'min':>8} {'median':>8} {'mean':>8} {'max':>8}")
print("-" * 70)
for s in [pos_stats, neg_stats]:
    print(f"{s['name']:<10} {s['min']:>8.3f} {s['median']:>8.3f} {s['mean']:>8.3f} {s['max']:>8.3f}")

# 分离度：正样本 max 和负样本 min 之间的差距
print(f"\n正样本 max = {pos_stats['max']:.3f}")
print(f"负样本 min = {neg_stats['min']:.3f}")
gap = neg_stats["min"] - pos_stats["max"]
if gap > 0:
    print(f"完全分离，间隔 = {gap:.3f}")
    print(f"任何落在 ({pos_stats['max']:.3f}, {neg_stats['min']:.3f}) 区间的阈值都是最优的")
else:
    print(f"有重叠，重叠区 = [{-gap:.3f}]（正样本有超出负样本 min 的，需权衡）")


# ===== 扫阈值 =====
def evaluate_threshold(thr):
    tp = sum(1 for d in pos_dists if d <= thr)   # 正样本被接受
    fn = len(pos_dists) - tp                     # 正样本被拒绝（漏答）
    fp = sum(1 for d in neg_dists if d <= thr)   # 负样本被接受（幻觉）
    tn = len(neg_dists) - fp                     # 负样本被拒绝

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn, tn


thresholds = [round(0.50 + i * 0.01, 2) for i in range(101)]  # 0.50 ~ 1.50
results = [(t, *evaluate_threshold(t)) for t in thresholds]

# 找 F1 最大
best_f1 = max(results, key=lambda r: r[3])
# 找 Precision >= 0.95 且召回最高的
valid = [r for r in results if r[1] >= 0.95]
best_p95 = max(valid, key=lambda r: r[2]) if valid else None

print("\n" + "=" * 70)
print("推荐阈值")
print("=" * 70)
print(f"当前阈值 = 1.10")
print(f"  Precision = {evaluate_threshold(1.10)[0]:.3f}")
print(f"  Recall    = {evaluate_threshold(1.10)[1]:.3f}")
print(f"  F1        = {evaluate_threshold(1.10)[2]:.3f}")

t, p, r, f1, tp, fp, fn, tn = best_f1
print(f"\nF1 最大点 = {t:.2f}")
print(f"  Precision = {p:.3f}")
print(f"  Recall    = {r:.3f}")
print(f"  F1        = {f1:.3f}")
print(f"  正样本被接受 = {tp}/{len(pos_dists)}，负样本被误接受 = {fp}/{len(neg_dists)}")

if best_p95:
    t, p, r, f1, tp, fp, fn, tn = best_p95
    print(f"\nPrecision >= 0.95 的最高召回点 = {t:.2f}")
    print(f"  Precision = {p:.3f}")
    print(f"  Recall    = {r:.3f}")
    print(f"  F1        = {f1:.3f}")


# ===== 画图（如果 matplotlib 装了）=====
try:
    import matplotlib
    matplotlib.use("Agg")  # 无 GUI
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # --- 图 1：距离分布直方图 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = [round(0.4 + i * 0.05, 2) for i in range(25)]  # 0.4 ~ 1.6
    ax.hist(pos_dists, bins=bins, alpha=0.6, label=f"正样本 (n={len(pos_dists)})", color="#2E86DE")
    ax.hist(neg_dists, bins=bins, alpha=0.6, label=f"负样本 (n={len(neg_dists)})", color="#EE5A52")
    ax.axvline(1.10, color="black", linestyle="--", linewidth=2, label="当前 MAX_DISTANCE=1.10")
    if best_f1[0] != 1.10:
        ax.axvline(best_f1[0], color="green", linestyle=":", linewidth=2, label=f"F1 最大 ={best_f1[0]:.2f}")
    ax.set_xlabel("best_distance (L2 距离，越小越相关)")
    ax.set_ylabel("样本数")
    ax.set_title("正负样本的 best_distance 分布")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "distance_distribution.png", dpi=150)
    print(f"\n图 1 已保存：{OUTPUT_DIR / 'distance_distribution.png'}")
    plt.close()

    # --- 图 2：PR 曲线 ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ts = [r[0] for r in results]
    ps = [r[1] for r in results]
    rs = [r[2] for r in results]
    fs = [r[3] for r in results]
    ax1.plot(ts, ps, label="Precision", color="#2E86DE", linewidth=2)
    ax1.plot(ts, rs, label="Recall", color="#EE5A52", linewidth=2)
    ax1.plot(ts, fs, label="F1", color="green", linewidth=2, linestyle="--")
    ax1.axvline(1.10, color="black", linestyle=":", alpha=0.7, label="当前阈值 1.10")
    if best_f1[0] != 1.10:
        ax1.axvline(best_f1[0], color="orange", linestyle=":", alpha=0.7, label=f"F1 最大 ={best_f1[0]:.2f}")
    ax1.set_xlabel("MAX_DISTANCE 阈值")
    ax1.set_ylabel("指标值")
    ax1.set_title("阈值扫描：Precision / Recall / F1")
    ax1.legend(loc="lower left")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pr_curve.png", dpi=150)
    print(f"图 2 已保存：{OUTPUT_DIR / 'pr_curve.png'}")
    plt.close()

except ImportError:
    print("\n[提示] matplotlib 未安装，跳过画图")
    print("       如需生成图表：pip install matplotlib")