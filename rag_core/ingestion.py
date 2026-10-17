"""Document loading and FAISS index construction.

The splitter parameters are intentionally unchanged from the original inline
implementation in mcp_test/mcp_rag_server.py: only metadata is added.
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_core.chunking import enrich_chunks

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", "。", "！", "？", " ", ""]


def load_documents(docs_dir: str | Path) -> list[Document]:
    docs_dir = Path(docs_dir)
    documents: list[Document] = []
    for md_file in sorted(docs_dir.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        rel_path = md_file.relative_to(docs_dir).as_posix()
        documents.append(Document(page_content=text, metadata={"source": rel_path}))
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    docs_by_source = {doc.metadata["source"]: doc.page_content for doc in documents}
    return enrich_chunks(chunks, docs_by_source)


def build_vectorstore(docs_dir: str | Path, embedding: Embeddings) -> FAISS:
    documents = load_documents(docs_dir)
    chunks = split_documents(documents)
    return FAISS.from_documents(chunks, embedding)


def ensure_index(index_dir: str | Path, docs_dir: str | Path, embedding: Embeddings) -> FAISS:
    index_dir = Path(index_dir)
    if (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists():
        return FAISS.load_local(
            str(index_dir),
            embedding,
            allow_dangerous_deserialization=True,
        )
    vectorstore = build_vectorstore(docs_dir, embedding)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore
