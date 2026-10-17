"""RAG 检索策略评估（source + chunk 双粒度、多金标、置信区间）。

跑法：
    python eval/eval_retrieval.py \
        --test-set eval/test_set_v3_gold.json \
        --k 3 --bootstrap 2000 \
        --json eval/output/retrieval_v3.json

兼容旧格式：只有 question + expected_sources 的测试集仍可跑，
但 chunk 级指标会显示 n/a。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_core.search import (
    DEFAULT_MODEL,
    ENABLE_RERANK,
    LocalEmbeddings,
    RECALL_K,
    build_bm25,
    diverse_by_source,
    hybrid_search,
    hybrid_search_with_rerank,
    load_vectorstore,
)
from rag_core.eval_metrics import (
    bootstrap_ci,
    evidence_coverage,
    mrr,
    ndcg_at_k,
    paired_bootstrap_diff,
    precision_at_k,
    recall_all,
    recall_any,
    wilson_ci,
)


def _ids(docs):
    return [d.metadata.get("chunk_id", "") for d in docs]


def _sources(docs):
    return [d.metadata.get("source", "") for d in docs]


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def _hit_list(values):
    return [1.0 if v else 0.0 for v in values if v is not None]


def load_test_set(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        item.setdefault("gold_sources", item.get("expected_sources", []))
        item.setdefault("gold_chunk_ids", [])
        item.setdefault("evidence_chunk_ids", [])
    return data


def random_baseline_source(test_set, source_counts, k=3):
    total = 0.0
    n_total = sum(source_counts.values())
    for item in test_set:
        n_expected = sum(source_counts.get(s, 0) for s in item["gold_sources"])
        if n_total - n_expected < k:
            prob = 1.0 if n_expected else 0.0
        else:
            prob = 1 - math.comb(n_total - n_expected, k) / math.comb(n_total, k)
        total += prob
    return total / len(test_set)


def random_baseline_chunk(test_set, n_total, k=3):
    total = 0.0
    for item in test_set:
        n_gold = len(set(item.get("gold_chunk_ids", [])))
        if n_gold == 0:
            continue
        if n_total - n_gold < k:
            prob = 1.0
        else:
            prob = 1 - math.comb(n_total - n_gold, k) / math.comb(n_total, k)
        total += prob
    return total / len(test_set)


def evaluate_strategy(test_set, search_fn, k, capture_candidates=False):
    rows = []
    for i, item in enumerate(test_set, 1):
        results, candidates = search_fn(item["question"], k)
        retrieved_ids = _ids(results)
        retrieved_sources = _sources(results)
        evidence = item.get("evidence_chunk_ids") or []
        gold_chunks = item.get("gold_chunk_ids") or []

        row = {
            "question": item["question"],
            "expected_sources": item.get("expected_sources", []),
            "gold_sources": item.get("gold_sources", []),
            "evidence_chunk_ids": evidence,
            "retrieved": [
                {
                    "chunk_id": d.metadata.get("chunk_id"),
                    "source": d.metadata.get("source"),
                    "section_path": d.metadata.get("section_path"),
                }
                for d in results
            ],
            "source_strict": recall_any(retrieved_sources, item.get("expected_sources", [])),
            "source_relaxed": recall_any(retrieved_sources, item.get("gold_sources", [])),
            "diversity": len(set(retrieved_sources)),
        }
        if gold_chunks:
            row.update(
                {
                    "chunk_any": recall_any(retrieved_ids, gold_chunks),
                    "chunk_all": recall_all(retrieved_ids, evidence) if evidence else None,
                    "evidence_coverage": evidence_coverage(retrieved_ids, evidence) if evidence else None,
                    "chunk_precision": precision_at_k(retrieved_ids, gold_chunks, k),
                    "mrr": mrr(retrieved_ids, gold_chunks),
                    "ndcg": ndcg_at_k(retrieved_ids, gold_chunks, k),
                }
            )
        if capture_candidates and candidates is not None:
            row["candidate_recall"] = (
                recall_any(_ids(candidates), gold_chunks) if gold_chunks else None
            )
        rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(test_set)}", file=sys.stderr)
    return rows


def summarize(rows):
    binary = ["source_strict", "source_relaxed", "chunk_any", "chunk_all", "candidate_recall"]
    summary = {}
    for key in binary:
        values = [r[key] for r in rows if r.get(key) is not None]
        hits = int(sum(values))
        lo, hi = wilson_ci(hits, len(values)) if values else (0.0, 0.0)
        summary[key] = {
            "value": hits / len(values) if values else None,
            "hits": hits,
            "n": len(values),
            "wilson_low": lo,
            "wilson_high": hi,
        }
    for key in ["chunk_precision", "evidence_coverage", "mrr", "ndcg", "diversity"]:
        values = [r[key] for r in rows if r.get(key) is not None]
        if not values:
            summary[key] = {"value": None, "n": 0}
            continue
        lo, hi = bootstrap_ci(values)
        summary[key] = {"value": _mean(values), "n": len(values), "ci_low": lo, "ci_high": hi}
    return summary


def _fmt(value):
    return "n/a" if value is None else f"{value:.1%}"


def _fmtf(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_table(results):
    print()
    print("| 策略 | source strict | source relaxed | chunk any | chunk all | candidate@20 | precision | MRR | nDCG@3 | diversity |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in results.items():
        print(
            f"| {name} | {_fmt(s['source_strict']['value'])} | {_fmt(s['source_relaxed']['value'])} | "
            f"{_fmt(s['chunk_any']['value'])} | {_fmt(s['chunk_all']['value'])} | "
            f"{_fmt(s['candidate_recall']['value'])} | "
            f"{_fmtf(s['chunk_precision']['value'])} | {_fmtf(s['mrr']['value'])} | "
            f"{_fmtf(s['ndcg']['value'])} | {_fmtf(s['diversity']['value'], 2)} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="eval/test_set_v3_gold.json")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--json", default="eval/output/retrieval_v3.json")
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--strategies",
        default="similarity,mmr,diverse,hybrid,hybrid_rerank",
        help="comma separated subset",
    )
    args = parser.parse_args()

    test_set = load_test_set(Path(args.test_set))
    if args.limit:
        test_set = test_set[: args.limit]
    print(f"测试集：{len(test_set)} 条")

    print("加载 embedding 和 FAISS 索引...")
    embedding = LocalEmbeddings(DEFAULT_MODEL)
    vectorstore = load_vectorstore(embedding=embedding)
    bm25, all_docs = build_bm25(vectorstore)
    print(f"索引：{len(all_docs)} chunk")

    def search_similarity(q, k):
        return vectorstore.similarity_search(q, k=k), None

    def search_mmr(q, k):
        return vectorstore.max_marginal_relevance_search(q, k=k, fetch_k=10), None

    def search_diverse(q, k):
        return diverse_by_source(q, vectorstore, k=k), None

    def search_hybrid(q, k):
        return hybrid_search(q, vectorstore, bm25, all_docs, k=k), None

    def search_hybrid_rerank(q, k):
        final = hybrid_search_with_rerank(
            q, vectorstore=vectorstore, bm25=bm25, all_docs=all_docs, final_k=k
        )
        candidates = hybrid_search(q, vectorstore, bm25, all_docs, k=RECALL_K)
        return final, candidates

    factories = {
        "similarity": ("普通相似度", search_similarity, False),
        "mmr": ("MMR", search_mmr, False),
        "diverse": ("MMR + 按来源去重", search_diverse, False),
        "hybrid": ("混合检索（向量+BM25+RRF）", search_hybrid, False),
        "hybrid_rerank": ("混合检索 + Rerank", search_hybrid_rerank, True),
    }
    selected = [s.strip() for s in args.strategies.split(",") if s.strip()]

    results = {}
    per_query = {}
    for key in selected:
        name, fn, capture = factories[key]
        print(f"跑 [{name}]...", flush=True)
        rows = evaluate_strategy(test_set, fn, args.k, capture_candidates=capture)
        results[name] = summarize(rows)
        per_query[name] = rows
        print(f"  source strict={_fmt(results[name]['source_strict']['value'])} "
              f"chunk any={_fmt(results[name]['chunk_any']['value'])}")

    source_counts = Counter(d.metadata.get("source", "") for d in all_docs)
    baselines = {
        "source_strict": random_baseline_source(
            [{"gold_sources": t.get("expected_sources", [])} for t in test_set],
            source_counts,
            args.k,
        ),
        "source_relaxed": random_baseline_source(test_set, source_counts, args.k),
        "chunk_any": random_baseline_chunk(test_set, len(all_docs), args.k),
    }
    print(f"\n随机基线：source strict={baselines['source_strict']:.1%} "
          f"source relaxed={baselines['source_relaxed']:.1%} "
          f"chunk any={baselines['chunk_any']:.1%}")

    print_table(results)

    # 配对 bootstrap：Rerank vs Hybrid、Rerank vs Similarity
    comparisons = {}
    if "混合检索 + Rerank" in per_query:
        rerank_rows = per_query["混合检索 + Rerank"]
        for other in ["混合检索（向量+BM25+RRF）", "普通相似度"]:
            if other not in per_query:
                continue
            other_rows = per_query[other]
            for metric in ["source_strict", "chunk_any"]:
                a = [r.get(metric) for r in rerank_rows if r.get(metric) is not None]
                b = [r.get(metric) for r in other_rows if r.get(metric) is not None]
                if len(a) != len(b) or not a:
                    continue
                diff = paired_bootstrap_diff(a, b, n=args.bootstrap)
                comparisons[f"Rerank vs {other} / {metric}"] = diff
                print(
                    f"配对 bootstrap [{metric}] Rerank - {other}: "
                    f"{diff['mean_diff']:+.1%} "
                    f"(95% CI {diff['ci_low']:+.1%} ~ {diff['ci_high']:+.1%}, p={diff['p_value']:.3f})"
                )

    out = {
        "config": {
            "test_set": args.test_set,
            "k": args.k,
            "recall_k": RECALL_K,
            "enable_rerank": ENABLE_RERANK,
            "chunks": len(all_docs),
            "test_items": len(test_set),
            "bootstrap": args.bootstrap,
        },
        "random_baseline": baselines,
        "strategies": results,
        "comparisons": comparisons,
        "per_query": per_query,
    }
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存：{out_path}")

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for name, rows in per_query.items():
                for row in rows:
                    f.write(json.dumps({"strategy": name, **row}, ensure_ascii=False) + "\n")
        print(f"逐条结果已保存：{jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
