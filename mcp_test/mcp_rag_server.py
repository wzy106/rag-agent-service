import os
import sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import jieba
from pathlib import Path
from rank_bm25 import BM25Okapi
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

# ===== 构建 BM25 索引 =====
all_docs = list(vectorstore.docstore._dict.values())
tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in all_docs]
bm25 = BM25Okapi(tokenized_corpus)
print(f"[RAG Server] BM25 索引构建完成，{len(all_docs)} 个 chunk", file=sys.stderr)

# ===== 降级阈值 =====
MAX_DISTANCE = 1.1   # 向量 L2 距离阈值，超过认为不相关（bge 归一化后范围 0~2）

# ===== 混合检索函数（含降级策略） =====
def hybrid_search(query, k=3, rrf_k=60, vector_weight=0.7, bm25_weight=0.3, max_distance=MAX_DISTANCE):
    # 向量检索（带距离分数）
    vector_results_with_scores = vectorstore.similarity_search_with_score(query, k=k*3)

    if not vector_results_with_scores:
        return []

    # 降级判断：最佳向量距离太大 → 认为知识库中无相关内容
    best_distance = vector_results_with_scores[0][1]
    print(f"[RAG] 最佳向量距离: {best_distance:.4f}", file=sys.stderr)

    if best_distance > max_distance:
        print(f"[RAG] 降级触发（距离 {best_distance:.4f} > {max_distance}）", file=sys.stderr)
        return []

    # 拆出文档
    vector_results = [doc for doc, _ in vector_results_with_scores]
    vector_ranks = {doc.page_content: i for i, doc in enumerate(vector_results)}

    # BM25 检索
    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k*3]
    bm25_ranks = {all_docs[i].page_content: rank for rank, i in enumerate(bm25_top_indices)}

    # 加权 RRF 合并
    all_contents = set(vector_ranks.keys()) | set(bm25_ranks.keys())
    rrf_scores = {}
    for content in all_contents:
        score = 0
        if content in vector_ranks:
            score += vector_weight / (rrf_k + vector_ranks[content])
        if content in bm25_ranks:
            score += bm25_weight / (rrf_k + bm25_ranks[content])
        rrf_scores[content] = score

    top_contents = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)[:k]
    content_to_doc = {doc.page_content: doc for doc in all_docs}
    return [content_to_doc[c] for c in top_contents]

# ===== MCP Server =====
mcp = FastMCP("RAG Knowledge Server")

@mcp.tool
def search_knowledge(query: str) -> str:
    """从知识库中检索与问题相关的信息，返回最相关的文档片段。"""
    results = hybrid_search(query, k=3)

    # 降级：没有相关结果
    if not results:
        return "知识库中没有找到相关信息。"

    return "\n\n".join([f"[{doc.metadata['source']}]\n{doc.page_content}" for doc in results])

if __name__ == "__main__":
    mcp.run()