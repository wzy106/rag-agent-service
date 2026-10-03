import os
import json
import time
import hashlib
import redis
import pymysql
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

# 找 .env（在上一级目录）
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# ===== Redis 缓存 =====
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def get_cached(query: str):
    key = f"chat:{hashlib.md5(query.encode()).hexdigest()}"
    return r.get(key)

def set_cache(query: str, answer: str, ttl: int = 3600):
    key = f"chat:{hashlib.md5(query.encode()).hexdigest()}"
    r.set(key, answer, ex=ttl)

# ===== MySQL 对话历史 =====
def save_conversation(thread_id: str, role: str, content: str):
    conn = pymysql.connect(
        host="localhost", port=3306, user="root",
        password=os.getenv("MYSQL_PASSWORD", "root123"), database="agent", charset="utf8mb4"
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (thread_id, role, content) VALUES (%s, %s, %s)",
                (thread_id, role, content)
            )
        conn.commit()
    finally:
        conn.close()

def ensure_table():
    """启动时确保 conversations 表存在"""
    conn = pymysql.connect(
        host="localhost", port=3306, user="root",
        password=os.getenv("MYSQL_PASSWORD", "root123"), database="agent", charset="utf8mb4"
    )
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    thread_id VARCHAR(64),
                    role VARCHAR(20),
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
    finally:
        conn.close()

# 全局 agent，lifespan 里初始化
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时加载 MCP 工具并创建 Agent"""
    global agent
    print("[启动] 正在连接 MCP Server...")

    client = MultiServerMCPClient({
        "rag": {
            "command": "python",
            "args": [str(Path(__file__).resolve().parent.parent / "mcp_test" / "mcp_rag_server.py")],
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()
    print(f"[启动] 从 MCP 加载了 {len(tools)} 个工具：{[t.name for t in tools]}")

    model = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0
    )

    agent = create_agent(
        model,
        tools=tools,
        system_prompt="你是一个知识助手，回答问题时优先调用 search_knowledge 工具查询知识库。"
    )
    print("[启动] Agent 创建完成")

    # 确保 MySQL 表存在
    ensure_table()
    print("[启动] MySQL 表检查完成")

    yield
    print("[关闭] 服务停止")

# ===== FastAPI =====
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def root():
    return {"message": "Agent API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    thread_id = "user_001"

    # 1. 先查缓存
    cached = get_cached(payload.message)
    if cached:
        print(f"[缓存命中] {payload.message}")
        save_conversation(thread_id, "human", payload.message)
        save_conversation(thread_id, "ai", cached)
        return ChatResponse(reply=cached)

    # 2. 缓存未命中，调 Agent
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": payload.message}]
    })
    answer = result["messages"][-1].content

    # 3. 写缓存 + 存历史
    set_cache(payload.message, answer)
    save_conversation(thread_id, "human", payload.message)
    save_conversation(thread_id, "ai", answer)
    print(f"[缓存写入] {payload.message}")
    return ChatResponse(reply=answer)

@app.get("/history")
def get_history(thread_id: str = "user_001", limit: int = 20):
    conn = pymysql.connect(
        host="localhost", port=3306, user="root",
        password="root123", database="agent", charset="utf8mb4"
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content, created_at FROM conversations "
                "WHERE thread_id = %s ORDER BY created_at DESC LIMIT %s",
                (thread_id, limit)
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return {
        "thread_id": thread_id,
        "count": len(rows),
        "messages": [
            {"role": row[0], "content": row[1], "created_at": str(row[2])}
            for row in rows
        ]
    }

class StreamRequest(BaseModel):
    message: str

@app.post("/chat/stream")
async def chat_stream(payload: StreamRequest):
    start_time = time.time()
    print(f"[请求] 收到消息：{payload.message}")

    async def event_generator():
        try:
            saw_tool = False
            pending = []
            started = False

            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": payload.message}]},
                stream_mode="messages"
            ):
                msg_chunk, metadata = chunk
                if msg_chunk.type == "tool":
                    saw_tool = True
                    pending = []
                    continue
                if not msg_chunk.content:
                    continue
                if saw_tool:
                    yield f"data: {json.dumps({'text': msg_chunk.content}, ensure_ascii=False)}\n\n"
                    started = True
                else:
                    pending.append(msg_chunk.content)

            if not started and pending:
                for text in pending:
                    yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': f'服务出错了：{str(e)}'}, ensure_ascii=False)}\n\n"
        finally:
            elapsed = time.time() - start_time
            print(f"[请求] 耗时 {elapsed:.2f} 秒")
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")