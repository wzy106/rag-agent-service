"""The committed FAISS index must match the committed docs.

This runs without torch: it reads index.pkl directly and re-splits docs with
the pure ingestion helpers. It catches the classic drift where docs are edited
but the index is not rebuilt.
"""
import pickle
from pathlib import Path

from rag_core.ingestion import load_documents, split_documents

ROOT = Path(__file__).resolve().parent.parent


def _index_docs():
    obj = pickle.load(open(ROOT / "faiss_index" / "index.pkl", "rb"))
    docstore = obj[0] if isinstance(obj, tuple) else obj
    return list(docstore._dict.values())


def test_index_matches_docs():
    expected = split_documents(load_documents(ROOT / "docs"))
    actual = _index_docs()

    assert len(actual) == len(expected)
    assert {d.metadata.get("chunk_id") for d in actual} == {
        d.metadata.get("chunk_id") for d in expected
    }


def test_every_index_chunk_has_metadata():
    for doc in _index_docs():
        assert doc.metadata.get("source")
        assert doc.metadata.get("chunk_id")
        assert "section_path" in doc.metadata
