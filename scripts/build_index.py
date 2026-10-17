"""Build or verify the FAISS index from docs/.

Usage:
    python scripts/build_index.py --force
    python scripts/build_index.py --check
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import search first: its module-level HF_HOME / *_OFFLINE setup must run
# before any library imports huggingface_hub.
from rag_core.search import LocalEmbeddings
from rag_core.ingestion import build_vectorstore, load_documents, split_documents


def _index_docs(index_dir: Path) -> list:
    pkl = index_dir / "index.pkl"
    if not pkl.exists():
        return []
    obj = pickle.load(open(pkl, "rb"))
    docstore = obj[0] if isinstance(obj, tuple) else obj
    return list(docstore._dict.values())


def check(index_dir: Path, docs_dir: Path) -> bool:
    expected = split_documents(load_documents(docs_dir))
    actual = _index_docs(index_dir)
    expected_ids = [c.metadata.get("chunk_id") for c in expected]
    actual_ids = [c.metadata.get("chunk_id") for c in actual]

    print(f"docs chunks: {len(expected)}")
    print(f"index chunks: {len(actual)}")
    missing_meta = sum(1 for i in actual_ids if not i)
    print(f"index chunks without chunk_id: {missing_meta}")

    if len(expected) != len(actual):
        print("DRIFT: chunk count differs")
        return False
    if set(expected_ids) != set(actual_ids):
        print("DRIFT: chunk_id sets differ")
        print(f"  only in docs: {len(set(expected_ids) - set(actual_ids))}")
        print(f"  only in index: {len(set(actual_ids) - set(expected_ids))}")
        return False
    print("OK: index matches docs")
    return True


def build(index_dir: Path, docs_dir: Path) -> None:
    print("Loading embedding model...")
    embedding = LocalEmbeddings()
    print("Building FAISS index...")
    vectorstore = build_vectorstore(docs_dir, embedding)
    chunks = list(vectorstore.docstore._dict.values())
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    missing = sum(1 for c in chunks if not c.metadata.get("chunk_id"))
    print(f"Saved {len(chunks)} chunks to {index_dir}")
    print(f"chunks without chunk_id: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default=str(_PROJECT_ROOT / "docs"))
    parser.add_argument("--index-dir", default=str(_PROJECT_ROOT / "faiss_index"))
    parser.add_argument("--force", action="store_true", help="rebuild the index")
    parser.add_argument("--check", action="store_true", help="compare index with docs")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    index_dir = Path(args.index_dir)

    if args.check:
        return 0 if check(index_dir, docs_dir) else 1
    if args.force:
        build(index_dir, docs_dir)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
