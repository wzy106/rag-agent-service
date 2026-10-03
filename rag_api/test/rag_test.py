import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from dotenv import load_dotenv
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# 1. 本地 embedding
class LocalEmbeddings(Embeddings):
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode(text).tolist()

model_path = str(next((Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5" / "snapshots").iterdir()))
embedding = LocalEmbeddings(model_path)

# 2. 准备知识库
docs = [
    Document(page_content="LangChain 是一个用于构建大模型应用的框架。"),
    Document(page_content="LangGraph 是 LangChain 生态里的多 Agent 编排工具。"),
    Document(page_content="RAG 是检索增强生成，用私有知识补充模型知识。"),
    Document(page_content="MCP 是模型上下文协议，用于标准化工具调用。"),
    Document(page_content="DeepSeek 是国产大模型，兼容 OpenAI API 格式。"),
]

vectorstore = InMemoryVectorStore(embedding=embedding)
vectorstore.add_documents(docs)

# 3. 把 RAG 包装成工具
@tool
def search_knowledge(query: str) -> str:
    """从知识库中检索与问题相关的信息，返回最相关的文档片段。"""
    results = vectorstore.similarity_search(query, k=2)
    return "\n".join([f"- {doc.page_content}" for doc in results])

# 4. 创建 Agent，把 RAG 工具交给它
model = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)

agent = create_agent(
    model,
    tools=[search_knowledge],
    system_prompt="你是一个知识助手，回答问题时优先调用 search_knowledge 工具查询知识库。"
)

# 5. 测试
result = agent.invoke({
    "messages": [{"role": "user", "content": "MCP 是什么？"}]
})

for msg in result["messages"]:
    if msg.content:
        print(f"[{msg.type}] {msg.content}")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  → 调用工具: {tc['name']}, 参数: {tc['args']}")