import os
import sys
from pathlib import Path

# ===== 确保能找到项目根的 rag_core 包（本地/容器都适用）=====
# Python 启动时 sys.path[0] 是脚本所在目录（mcp_test/），不是项目根，
# 所以需要手动把项目根加进去。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ===== 从公共模块导入（import 时会自动设置 HF_HOME + HF_HUB_OFFLINE）=====
from rag_core.search import (
    LocalEmbeddings,
    build_bm25,
    hybrid_search_with_rerank,
    DEFAULT_MODEL,
    DEFAULT_INDEX_DIR,
)
from rag_core.ingestion import ensure_index

import anyio
from fastmcp import FastMCP

print(f"[RAG Server] HF_HOME={os.environ.get('HF_HOME')}", file=sys.stderr)


# ===== 初始化 embedding =====
embedding = LocalEmbeddings(DEFAULT_MODEL)


# ===== 加载或构建 FAISS 索引 =====
docs_dir = _PROJECT_ROOT / "docs"
vectorstore = ensure_index(DEFAULT_INDEX_DIR, docs_dir, embedding)
print(f"[RAG Server] 索引就绪（{len(vectorstore.docstore._dict)} chunk）", file=sys.stderr)


# ===== 构建 BM25 索引 =====
bm25, all_docs = build_bm25(vectorstore)
print(f"[RAG Server] BM25 索引构建完成，{len(all_docs)} 个 chunk", file=sys.stderr)


# ===== MCP Server =====
mcp = FastMCP("RAG Knowledge Server")


@mcp.tool
async def search_knowledge(query: str) -> str:
    """从知识库中检索与问题相关的信息，返回最相关的文档片段。"""
    # 两阶段检索：混合检索召回 Top-N（配置 RECALL_K）→ Reranker 精排 Top-3。
    # 检索是 CPU 密集的同步逻辑，丢到线程池执行，避免阻塞 MCP server 的事件循环
    # （否则并发的工具调用会被串行化）。
    results = await anyio.to_thread.run_sync(
        lambda: hybrid_search_with_rerank(
            query,
            vectorstore=vectorstore,
            bm25=bm25,
            all_docs=all_docs,
            final_k=3,
            verbose=True,
        )
    )

    # 降级：没有相关结果
    if not results:
        return "知识库中没有找到相关信息。"

    return "\n\n".join(
        f"[{doc.metadata['source']}]\n{doc.page_content}" for doc in results
    )


if __name__ == "__main__":
    mcp.run()