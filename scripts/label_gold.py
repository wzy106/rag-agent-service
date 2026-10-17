"""Expand single evidence labels into multi-gold chunk/source labels.

A question generated from one chunk may be answerable by a near-duplicate chunk
in another document (the corpus has many overlapping notes). This script adds
all chunks with 3-gram Jaccard >= threshold to `gold_chunk_ids`, and their
sources to `gold_sources`, while keeping the original evidence ids.

Usage:
    python scripts/label_gold.py \
        --test-set eval/test_set_v3.json \
        --threshold 0.8 --sensitivity 0.6 \
        --output eval/test_set_v3_gold.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_index_docs(index_dir: Path) -> list:
    obj = pickle.load(open(index_dir / "index.pkl", "rb"))
    docstore = obj[0] if isinstance(obj, tuple) else obj
    return list(docstore._dict.values())


def grams(text: str) -> set[str]:
    text = re.sub(r"\s+", "", text)
    return {text[i : i + 3] for i in range(max(0, len(text) - 2))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def expand(test_set, id_to_doc, threshold):
    grams_cache = {cid: grams(doc.page_content) for cid, doc in id_to_doc.items()}
    out = []
    ambiguous_items = 0
    added_chunks = 0
    added_sources = 0
    missing_evidence = set()

    for item in test_set:
        evidence = item.get("evidence_chunk_ids", [])
        gold_chunks = set(evidence)
        gold_sources = set(item.get("expected_sources", []))
        original_sources = set(gold_sources)

        for evidence_id in evidence:
            doc = id_to_doc.get(evidence_id)
            if doc is None:
                missing_evidence.add(evidence_id)
                continue
            base_grams = grams_cache[evidence_id]
            for chunk_id, candidate in id_to_doc.items():
                if chunk_id in gold_chunks:
                    continue
                if jaccard(base_grams, grams_cache[chunk_id]) >= threshold:
                    gold_chunks.add(chunk_id)
                    gold_sources.add(candidate.metadata.get("source", ""))

        if gold_chunks != set(evidence):
            ambiguous_items += 1
        added_chunks += len(gold_chunks) - len(evidence)
        added_sources += len(gold_sources - original_sources)

        out.append(
            {
                **item,
                "gold_chunk_ids": sorted(gold_chunks),
                "gold_sources": sorted(gold_sources),
                "label_ambiguous": gold_chunks != set(evidence),
            }
        )
    stats = {
        "items": len(out),
        "ambiguous_items": ambiguous_items,
        "ambiguous_rate": ambiguous_items / len(out) if out else 0.0,
        "added_chunk_labels": added_chunks,
        "added_source_labels": added_sources,
        "threshold": threshold,
        "missing_evidence_ids": sorted(missing_evidence),
    }
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="eval/test_set_v3.json")
    parser.add_argument("--index-dir", default="faiss_index")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--sensitivity", type=float, default=0.6)
    parser.add_argument("--output", default="eval/test_set_v3_gold.json")
    args = parser.parse_args()

    test_set = json.loads(Path(args.test_set).read_text(encoding="utf-8"))
    docs = load_index_docs(Path(args.index_dir))
    id_to_doc = {d.metadata["chunk_id"]: d for d in docs if d.metadata.get("chunk_id")}
    print(f"索引 chunks: {len(docs)}，带 chunk_id: {len(id_to_doc)}")

    gold, stats = expand(test_set, id_to_doc, args.threshold)
    sensitivity, sens_stats = expand(test_set, id_to_doc, args.sensitivity)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(
            {"primary": stats, "sensitivity": sens_stats},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[阈值 {args.threshold}] 歧义项 {stats['ambiguous_items']}/{stats['items']} "
        f"({stats['ambiguous_rate']:.1%})，额外 chunk 标签 {stats['added_chunk_labels']}，"
        f"额外 source 标签 {stats['added_source_labels']}"
    )
    print(
        f"[敏感度 {args.sensitivity}] 歧义项 {sens_stats['ambiguous_items']}/{sens_stats['items']} "
        f"({sens_stats['ambiguous_rate']:.1%})，额外 source 标签 {sens_stats['added_source_labels']}"
    )
    if stats["missing_evidence_ids"]:
        print(f"警告：{len(stats['missing_evidence_ids'])} 个 evidence id 不在索引中")
    print(f"保存：{out_path}")
    print(f"统计：{stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
