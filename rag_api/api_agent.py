import os
import json
import time
import hashlib
import redis
import pymysql
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# 找 .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# ===== 护栏配置 =====
RECURSION_LIMIT = 10
MAX_TOKENS = 2000

# ===== Redis 缓存 =====
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def get_context_version(thread_id: str) -> int:
    """
    获取该会话的上下文版本号（= MySQL 里的消息数）。
    同一个问题在不同上下文里答案可能不同，所以缓存 key 必须带版本号。
    """
    conn = pymysql.connect(
        host="localhost", port=3306, user="root",
        password=os.getenv("MYSQL_PASSWORD", "root123"), database="agent", charset="utf8mb4"
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM conversations WHERE thread_id = %s",
                (thread_id,)
            )
            count = cur.fetchone()[0]
    finally:
        conn.close()
    return count

def get_cached(thread_id: str, query: str, context_version: int):
    key = f"chat:{thread_id}:v{context_version}:{hashlib.md5(query.encode()).hexdigest()}"
    return r.get(key)

def set_cache(thread_id: str, query: str, answer: str, context_version: int, ttl: int = 3600):
    key = f"chat:{thread_id}:v{context_version}:{hashlib.md5(query.encode()).hexdigest()}"
    r.set(key, answer, ex=ttl)

# ===== 友好错误提示 =====
def friendly_error(e: Exception) -> str:
    """
    把异常映射成用户可读的错误提示。
    不把 str(e) 直接返回客户端（可能泄漏 API key、路径、SQL 结构等内部信息）。
    """
    msg = str(e).lower()
    if "authentication" in msg or "401" in msg or "api key" in msg:
        return "模型服务认证失败，请联系管理员"
    if "timeout" in msg or "timed out" in msg:
        return "请求超时，请稍后再试"
    if "rate" in msg or "429" in msg:
        return "请求过于频繁，请稍后再试"
    if "connection" in msg or "connect" in msg:
        return "服务暂时不可用，请稍后再试"
    return "服务暂时不可用，请稍后再试"

# ===== MySQL 对话历史（展示层） =====
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

    checkpoint_path = str(Path(__file__).resolve().parent / "checkpoints.db")
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
        agent = create_agent(
            model,
            tools=tools,
            checkpointer=saver,
            system_prompt="你是一个知识助手，回答问题时优先调用 search_knowledge 工具查询知识库。"
        )
        print(f"[启动] Agent 创建完成（持久化记忆 + 护栏：recursion_limit={RECURSION_LIMIT}, max_tokens={MAX_TOKENS}）")

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

    # 1. 查缓存（key 带上下文版本号）
    context_version = get_context_version(thread_id)
    cached = get_cached(thread_id, payload.message, context_version)
    if cached:
        print(f"[缓存命中] {payload.message} (v{context_version})")
        save_conversation(thread_id, "human", payload.message)
        save_conversation(thread_id, "ai", cached)
        return ChatResponse(reply=cached)

    # 2. 调 Agent
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": payload.message}]},
            config=config
        )
        answer = result["messages"][-1].content
    except Exception as e:
        # 服务端日志记录完整异常
        print(f"[错误] {type(e).__name__}: {str(e)}")
        # 客户端只看到友好提示
        raise HTTPException(status_code=500, detail=friendly_error(e))

    # 3. 写缓存（用当前 version）+ 存历史
    set_cache(thread_id, payload.message, answer, context_version)
    save_conversation(thread_id, "human", payload.message)
    save_conversation(thread_id, "ai", answer)
    print(f"[缓存写入] {payload.message} (v{context_version})")
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

            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": RECURSION_LIMIT,
            }

            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": payload.message}]},
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

            # 兜底：如果整个流程没输出任何文本，给个友好提示
            if not started and pending:
                for text in pending:
                    yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
                    full_answer += text
            elif not started and not pending:
                fallback = "抱歉，我无法生成回答。"
                yield f"data: {json.dumps({'text': fallback}, ensure_ascii=False)}\n\n"
                full_answer = fallback

            # 存 MySQL（展示层）
            save_conversation(thread_id, "human", payload.message)
            save_conversation(thread_id, "ai", full_answer)

        except Exception as e:
            # 服务端日志记录完整异常
            print(f"[错误] {type(e).__name__}: {str(e)}")
            # 客户端只看到友好提示
            yield f"data: {json.dumps({'error': friendly_error(e)}, ensure_ascii=False)}\n\n"
        finally:
            elapsed = time.time() - start_time
            print(f"[请求] 耗时 {elapsed:.2f} 秒")
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")