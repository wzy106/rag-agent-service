"""MAX_DISTANCE 阈值标定（区分 easy / hard 负样本）。

注意：这里衡量的是**检索层是否接受一个 query**，不是端到端幻觉率。
- FPR = FP / (FP + TN)：负样本被接受的比例。
- FDR = FP / (TP + FP)：被接受的结果里，负样本占的比例；依赖正负样本的抽样基率。

跑法：
    python eval/calibrate_distance.py \
        --positives eval/test_set_v3_gold.json \
        --negatives eval/negatives.json eval/negatives_hard.json \
        --output eval/output/calibration_v3.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_core.search import DEFAULT_MODEL, MAX_DISTANCE, LocalEmbeddings, load_vectorstore


def load_questions(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x if isinstance(x, str) else x["question"] for x in data]
    return list(data["questions"])


def stats(name, dists):
    dists = sorted(dists)
    n = len(dists)
    return {
        "name": name,
        "n": n,
        "min": dists[0] if dists else None,
        "median": dists[n // 2] if dists else None,
        "mean": sum(dists) / n if n else None,
        "max": dists[-1] if dists else None,
    }


def threshold_metrics(threshold, pos_dists, neg_groups):
    tp = sum(1 for d in pos_dists if d <= threshold)
    fn = len(pos_dists) - tp
    fp = sum(1 for group in neg_groups.values() for d in group if d <= threshold)
    tn = sum(1 for group in neg_groups.values() for d in group if d > threshold)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr_by_group = {
        name: (sum(1 for d in group if d <= threshold) / len(group) if group else 0.0)
        for name, group in neg_groups.items()
    }
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fdr = fp / (tp + fp) if (tp + fp) else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "negative_fpr": fpr,
        "negative_fdr": fdr,
        "fpr_by_group": fpr_by_group,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives", default="eval/test_set_v3_gold.json")
    parser.add_argument(
        "--negatives",
        nargs="+",
        default=["eval/negatives.json", "eval/negatives_hard.json"],
    )
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval/output/calibration_v3.json")
    args = parser.parse_args()

    print("加载 embedding 和 FAISS 索引...")
    embedding = LocalEmbeddings(DEFAULT_MODEL)
    vectorstore = load_vectorstore(embedding=embedding)

    positives_all = load_questions(Path(args.positives))
    rng = random.Random(args.seed)
    positives = rng.sample(positives_all, min(args.sample, len(positives_all)))

    neg_groups = {}
    for path_str in args.negatives:
        path = Path(path_str)
        name = path.stem
        neg_groups[name] = load_questions(path)

    print(f"正样本：{len(positives)} 条（从 {len(positives_all)} 条抽样）")
    for name, questions in neg_groups.items():
        print(f"负样本[{name}]：{len(questions)} 条")
    print(f"当前 MAX_DISTANCE = {MAX_DISTANCE}\n")

    def best_distance(query: str) -> float:
        results = vectorstore.similarity_search_with_score(query, k=1)
        # FAISS returns numpy.float32; normalize for JSON/stats.
        return float(results[0][1]) if results else float("inf")

    print("计算正样本距离...")
    pos_dists = [best_distance(q) for q in positives]
    neg_dists = {}
    for name, questions in neg_groups.items():
        print(f"计算负样本距离[{name}]...")
        neg_dists[name] = [best_distance(q) for q in questions]

    print("\n" + "=" * 78)
    print("距离分布统计")
    print("=" * 78)
    print(f"{'类别':<24} {'n':>4} {'min':>8} {'median':>8} {'mean':>8} {'max':>8}")
    print("-" * 78)
    for s in [stats("positive", pos_dists)] + [stats(k, v) for k, v in neg_dists.items()]:
        print(
            f"{s['name']:<24} {s['n']:>4} {s['min']:>8.3f} {s['median']:>8.3f} "
            f"{s['mean']:>8.3f} {s['max']:>8.3f}"
        )

    thresholds = [round(0.50 + i * 0.01, 2) for i in range(101)]
    results = [threshold_metrics(t, pos_dists, neg_dists) for t in thresholds]
    best_f1 = max(results, key=lambda r: r["f1"])
    current = threshold_metrics(MAX_DISTANCE, pos_dists, neg_dists)

    print("\n" + "=" * 78)
    print("指标（FPR/FDR 不是端到端幻觉率）")
    print("=" * 78)
    for label, m in [("当前阈值", current), ("F1 最大点", best_f1)]:
        print(f"\n{label} = {m['threshold']:.2f}")
        print(f"  Precision = {m['precision']:.3f}  Recall = {m['recall']:.3f}  F1 = {m['f1']:.3f}")
        print(f"  TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']}")
        print(f"  negative FPR = {m['negative_fpr']:.3f}  ({m['fp']}/{m['fp'] + m['tn']})")
        print(f"  negative FDR among accepted = {m['negative_fdr']:.3f}")
        for name, value in m["fpr_by_group"].items():
            print(f"    FPR[{name}] = {value:.3f}")

    out = {
        "config": {
            "positives": args.positives,
            "negatives": args.negatives,
            "sample": len(positives),
            "seed": args.seed,
            "max_distance": MAX_DISTANCE,
        },
        "distributions": {
            "positive": pos_dists,
            **{name: values for name, values in neg_dists.items()},
        },
        "current": current,
        "best_f1": best_f1,
        "sweep": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存：{out_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(10, 5))
        bins = [round(0.4 + i * 0.05, 2) for i in range(25)]
        ax.hist(pos_dists, bins=bins, alpha=0.6, label=f"positive (n={len(pos_dists)})")
        for name, values in neg_dists.items():
            ax.hist(values, bins=bins, alpha=0.6, label=f"{name} (n={len(values)})")
        ax.axvline(MAX_DISTANCE, color="black", linestyle="--", linewidth=2,
                   label=f"MAX_DISTANCE={MAX_DISTANCE}")
        ax.set_xlabel("best_distance (L2, lower is more relevant)")
        ax.set_ylabel("count")
        ax.set_title("positive vs negative best_distance")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_path.parent / "distance_distribution.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot([r["threshold"] for r in results], [r["precision"] for r in results], label="precision")
        ax.plot([r["threshold"] for r in results], [r["recall"] for r in results], label="recall")
        ax.plot([r["threshold"] for r in results], [r["f1"] for r in results], label="f1", linestyle="--")
        ax.axvline(MAX_DISTANCE, color="black", linestyle=":", label=f"current {MAX_DISTANCE}")
        ax.set_xlabel("threshold")
        ax.set_ylabel("metric")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_path.parent / "pr_curve.png", dpi=150)
        plt.close()
        print(f"图已保存：{out_path.parent / 'distance_distribution.png'} / pr_curve.png")
    except ImportError:
        print("matplotlib 未安装，跳过画图")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
