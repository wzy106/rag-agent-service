import os
import sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from fastmcp import FastMCP
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

class LocalEmbeddings(Embeddings):
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode(text).tolist()

model_path = str(next((Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5" / "snapshots").iterdir()))
embedding = LocalEmbeddings(model_path)

INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"

if INDEX_DIR.exists():
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embedding,
        allow_dangerous_deserialization=True
    )
    print("[RAG Server] 从磁盘加载索引", file=sys.stderr)
else:
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    documents = []
    for md_file in docs_dir.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"source": md_file.name}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[RAG Server] 首次构建索引，{len(chunks)} 块", file=sys.stderr)

    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"[RAG Server] 索引已保存到 {INDEX_DIR}", file=sys.stderr)

mcp = FastMCP("RAG Knowledge Server")

@mcp.tool
def search_knowledge(query: str) -> str:
    """从知识库中检索与问题相关的信息，返回最相关的文档片段。"""
    results = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=10)
    return "\n\n".join([f"[{doc.metadata['source']}]\n{doc.page_content}" for doc in results])

if __name__ == "__main__":
    mcp.run()