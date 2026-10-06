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
from langgraph.checkpoint.memory import InMemorySaver

# 找 .env（在上一级目录）
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# ===== 护栏配置 =====
RECURSION_LIMIT = 10    # Agent 最大循环次数，超过自动终止
MAX_TOKENS = 2000       # 单次输出最大 token 数
HISTORY_ROUNDS = 3      # 长期记忆：加载最近 N 轮对话

# ===== Redis 缓存 =====
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def get_cached(thread_id: str, query: str):
    key = f"chat:{thread_id}:{hashlib.md5(query.encode()).hexdigest()}"
    return r.get(key)

def set_cache(thread_id: str, query: str, answer: str, ttl: int = 3600):
    key = f"chat:{thread_id}:{hashlib.md5(query.encode()).hexdigest()}"
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

def load_recent_history(thread_id: str, rounds: int = HISTORY_ROUNDS):
    """加载最近 N 轮对话，按时间正序返回（用于拼进 messages）"""
    conn = pymysql.connect(
        host="localhost", port=3306, user="root",
        password=os.getenv("MYSQL_PASSWORD", "root123"), database="agent", charset="utf8mb4"
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM conversations "
                "WHERE thread_id = %s ORDER BY created_at DESC LIMIT %s",
                (thread_id, rounds * 2)   # 每轮 = human + ai
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    # SQL 查出来是最新在前，要反转为时间正序
    history = []
    for role, content in reversed(rows):
        history.append({"role": role, "content": content})
    return history

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

# 全局 agent
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
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
        temperature=0,
        max_tokens=MAX_TOKENS,
    )

    memory = InMemorySaver()

    agent = create_agent(
        model,
        tools=tools,
        checkpointer=memory,
        system_prompt="你是一个知识助手，回答问题时优先调用 search_knowledge 工具查询知识库。"
    )
    print(f"[启动] Agent 创建完成（护栏：recursion_limit={RECURSION_LIMIT}, max_tokens={MAX_TOKENS}, 历史轮数={HISTORY_ROUNDS}）")

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
    thread_id: str = "user_001"

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def root():
    return {"message": "Agent API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    thread_id = payload.thread_id

    # 1. 先查缓存
    cached = get_cached(thread_id, payload.message)
    if cached:
        print(f"[缓存命中] {payload.message}")
        save_conversation(thread_id, "human", payload.message)
        save_conversation(thread_id, "ai", cached)
        return ChatResponse(reply=cached)

    # 2. 加载长期记忆（MySQL 最近 N 轮）
    history = load_recent_history(thread_id)
    print(f"[长期记忆] 加载了 {len(history)} 条历史消息")

    # 3. 拼 messages：历史 + 本次问题
    messages = history + [{"role": "user", "content": payload.message}]

    # 4. 调 Agent
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    result = await agent.ainvoke({"messages": messages}, config=config)
    answer = result["messages"][-1].content

    # 5. 写缓存 + 存历史
    set_cache(thread_id, payload.message, answer)
    save_conversation(thread_id, "human", payload.message)
    save_conversation(thread_id, "ai", answer)
    print(f"[缓存写入] {payload.message}")
    return ChatResponse(reply=answer)

@app.get("/history")
def get_history(thread_id: str = "user_001", limit: int = 20):
    conn = pymysql.connect(
        host="localhost", port=3306, user="root",
        password=os.getenv("MYSQL_PASSWORD", "root123"), database="agent", charset="utf8mb4"
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
    thread_id: str = "user_001"

@app.post("/chat/stream")
async def chat_stream(payload: StreamRequest):
    start_time = time.time()
    thread_id = payload.thread_id
    print(f"[请求] 收到消息：{payload.message}（thread_id={thread_id}）")

    async def event_generator():
        try:
            saw_tool = False
            pending = []
            started = False
            full_answer = ""

            # 加载长期记忆
            history = load_recent_history(thread_id)
            print(f"[长期记忆] 加载了 {len(history)} 条历史消息")

            messages = history + [{"role": "user", "content": payload.message}]

            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": RECURSION_LIMIT,
            }

            async for chunk in agent.astream(
                {"messages": messages},
                config=config,
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
                    full_answer += msg_chunk.content
                    started = True
                else:
                    pending.append(msg_chunk.content)

            if not started and pending:
                for text in pending:
                    yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
                    full_answer += text

            # 流式结束后存历史
            save_conversation(thread_id, "human", payload.message)
            save_conversation(thread_id, "ai", full_answer)

        except Exception as e:
            yield f"data: {json.dumps({'error': f'服务出错了：{str(e)}'}, ensure_ascii=False)}\n\n"
        finally:
            elapsed = time.time() - start_time
            print(f"[请求] 耗时 {elapsed:.2f} 秒")
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")