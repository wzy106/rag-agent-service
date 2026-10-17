import math

import pytest

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


def test_recall_any_and_all():
    assert recall_any(["a", "b"], ["b", "c"]) == 1.0
    assert recall_any(["a"], ["b"]) == 0.0
    assert recall_all(["a", "b", "x"], ["a", "b"]) == 1.0
    assert recall_all(["a"], ["a", "b"]) == 0.0
    assert recall_any([], ["a"]) == 0.0


def test_evidence_coverage():
    assert evidence_coverage(["a", "b"], ["a", "c"]) == 0.5
    assert evidence_coverage([], ["a"]) == 0.0
    assert evidence_coverage(["a"], []) == 0.0


def test_precision_and_mrr():
    assert precision_at_k(["a", "x", "b"], ["a", "b"], 3) == pytest.approx(2 / 3)
    assert precision_at_k(["x", "a"], ["a"], 3) == pytest.approx(1 / 3)
    assert mrr(["x", "a"], ["a"]) == pytest.approx(0.5)
    assert mrr(["x"], ["a"]) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k(["a"], ["a"], 3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], ["a"], 3) == pytest.approx(
        (1 / math.log2(3)) / 1.0
    )
    assert ndcg_at_k(["x"], ["a"], 3) == 0.0


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(0, 10)
    assert 0 <= lo <= hi <= 1
    lo, hi = wilson_ci(10, 10)
    assert 0 <= lo <= hi <= 1
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_bootstrap_is_deterministic_with_seed():
    values = [1, 0, 1, 1, 0, 1]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)
    lo, hi = bootstrap_ci(values, seed=7)
    assert lo <= hi


def test_paired_bootstrap_diff():
    a = [1, 1, 1, 0, 1]
    b = [0, 0, 0, 0, 0]
    result = paired_bootstrap_diff(a, b, n=500, seed=1)
    assert result["mean_diff"] == pytest.approx(0.8)
    assert result["ci_low"] <= result["mean_diff"] <= result["ci_high"]
    assert 0 <= result["p_value"] <= 1


def test_paired_bootstrap_requires_equal_length():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1], [1, 0])
