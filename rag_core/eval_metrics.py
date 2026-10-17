"""Pure-Python retrieval metrics and bootstrap confidence intervals."""
from __future__ import annotations

import math
import random
from statistics import mean
from typing import Callable, Iterable, Sequence


def recall_any(retrieved_ids: Sequence[str], gold_ids: Iterable[str]) -> float:
    gold = set(gold_ids)
    return 1.0 if gold and any(item in gold for item in retrieved_ids) else 0.0


def recall_all(retrieved_ids: Sequence[str], gold_ids: Iterable[str]) -> float:
    gold = set(gold_ids)
    return 1.0 if gold and gold.issubset(set(retrieved_ids)) else 0.0


def evidence_coverage(retrieved_ids: Sequence[str], evidence_ids: Iterable[str]) -> float:
    evidence = set(evidence_ids)
    if not evidence:
        return 0.0
    return len(evidence & set(retrieved_ids)) / len(evidence)


def precision_at_k(
    retrieved_ids: Sequence[str],
    gold_ids: Iterable[str],
    k: int | None = None,
) -> float:
    denominator = k if k is not None else len(retrieved_ids)
    if denominator <= 0:
        return 0.0
    gold = set(gold_ids)
    return sum(1 for item in retrieved_ids[:denominator] if item in gold) / denominator


def mrr(retrieved_ids: Sequence[str], gold_ids: Iterable[str]) -> float:
    gold = set(gold_ids)
    for rank, item in enumerate(retrieved_ids, 1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved_ids[:k], 1)
        if item in gold
    )
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = mean,
    n: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    size = len(values)
    samples = [stat([values[rng.randrange(size)] for _ in range(size)]) for _ in range(n)]
    samples.sort()
    alpha = (1 - confidence) / 2
    lo = samples[int(alpha * n)]
    hi = samples[min(n - 1, int((1 - alpha) * n))]
    return (lo, hi)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap the mean paired difference a - b."""
    if len(a) != len(b) or not a:
        raise ValueError("a and b must be non-empty and the same length")
    rng = random.Random(seed)
    size = len(a)
    point = mean(a) - mean(b)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(size) for _ in range(size)]
        diffs.append(mean(a[i] - b[i] for i in idx))
    diffs.sort()
    alpha = (1 - confidence) / 2
    lo = diffs[int(alpha * n)]
    hi = diffs[min(n - 1, int((1 - alpha) * n))]
    # Two-sided bootstrap p-value around zero.
    below = sum(1 for d in diffs if d <= 0) / n
    above = sum(1 for d in diffs if d >= 0) / n
    p_value = min(1.0, 2 * min(below, above))
    return {"mean_diff": point, "ci_low": lo, "ci_high": hi, "p_value": p_value}
