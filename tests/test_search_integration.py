"""检索层集成测试：需要本地 HF 模型 + FAISS 索引，CI 默认跳过。

跑法：pytest -m integration
"""
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def retrieval():
    from rag_core.search import (
        DEFAULT_INDEX_DIR,
        DEFAULT_MODEL,
        LocalEmbeddings,
        build_bm25,
        load_vectorstore,
    )

    embedding = LocalEmbeddings(DEFAULT_MODEL)
    vectorstore = load_vectorstore(DEFAULT_INDEX_DIR, embedding)
    bm25, all_docs = build_bm25(vectorstore)
    return vectorstore, bm25, all_docs


def test_index_loads(retrieval):
    _, _, all_docs = retrieval
    assert len(all_docs) > 100
    assert all(doc.metadata.get("source") for doc in all_docs)


def test_hybrid_search_returns_at_most_k(retrieval):
    from rag_core.search import hybrid_search

    vectorstore, bm25, all_docs = retrieval
    results = hybrid_search("MMR 是什么？", vectorstore, bm25, all_docs, k=3)
    assert 0 < len(results) <= 3
    assert all(doc.metadata.get("source") for doc in results)


def test_degradation_when_threshold_is_impossible(retrieval):
    """距离阈值收紧到不可能满足时，应当降级返回空（宁可不答，也不编造）。"""
    from rag_core.search import hybrid_search

    vectorstore, bm25, all_docs = retrieval
    assert hybrid_search("MMR 是什么？", vectorstore, bm25, all_docs, k=3, max_distance=-1.0) == []


def test_rerank_can_be_disabled_without_loading_model(monkeypatch, retrieval):
    """ENABLE_RERANK=False 时应跳过精排，且不加载 reranker。"""
    import rag_core.search as s

    vectorstore, bm25, all_docs = retrieval
    monkeypatch.setattr(s, "ENABLE_RERANK", False)
    monkeypatch.setattr(s, "_reranker", None)

    results = s.hybrid_search_with_rerank("MMR 是什么？", vectorstore, bm25, all_docs, final_k=3)

    assert len(results) == 3
    assert s._reranker is None      # 没有被加载
