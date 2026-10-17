from langchain_core.documents import Document

from rag_core.search import doc_key, hybrid_search


class FakeVectorStore:
    def __init__(self, rows):
        self.rows = rows

    def similarity_search_with_score(self, query, k=4):
        return self.rows[:k]


class FakeBM25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, tokenized_query):
        return self.scores


def _docs():
    return [
        Document(page_content="same content", metadata={"source": "a.md", "chunk_id": "a"}),
        Document(page_content="same content", metadata={"source": "b.md", "chunk_id": "b"}),
        Document(page_content="unique content", metadata={"source": "c.md", "chunk_id": "c"}),
    ]


def test_doc_key_uses_chunk_id_not_content():
    a, b, _ = _docs()
    assert doc_key(a) == "a"
    assert doc_key(b) == "b"
    assert doc_key(a) != doc_key(b)


def test_cross_source_duplicates_do_not_collapse():
    docs = _docs()
    vectorstore = FakeVectorStore([(docs[0], 0.1), (docs[1], 0.2), (docs[2], 0.3)])
    bm25 = FakeBM25([0.0, 0.0, 0.0])

    raw = hybrid_search("same", vectorstore, bm25, docs, k=3, dedupe_content=False)
    assert {d.metadata["chunk_id"] for d in raw} == {"a", "b", "c"}

    deduped = hybrid_search("same", vectorstore, bm25, docs, k=3, dedupe_content=True)
    assert {d.metadata["chunk_id"] for d in deduped} == {"a", "c"}


def test_threshold_gate_still_returns_empty():
    docs = _docs()
    vectorstore = FakeVectorStore([(docs[0], 2.0), (docs[1], 2.1), (docs[2], 2.2)])
    bm25 = FakeBM25([0.0, 10.0, 0.0])
    assert hybrid_search("same", vectorstore, bm25, docs, k=3, max_distance=0.96) == []
