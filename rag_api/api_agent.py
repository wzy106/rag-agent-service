import os
import json
import time
import asyncio
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

# ===== 友好错误提示 =====
def friendly_error(e: Exception) -> str:
    """
    把异常映射成用户可读的错误提示。
    用异常类型和 HTTP 状态码判断，不做自然语言子串匹配（避免 rate 匹配到 generate 之类的误判）。
    """
    status = getattr(e, "status_code", None)
    if status == 401:
        return "模型服务认证失败，请联系管理员"
    if status == 429:
        return "请求过于频繁，请稍后再试"
    if isinstance(e, asyncio.TimeoutError):
        return "请求超时，请稍后再试"
    if isinstance(e, pymysql.MySQLError):
        return "存储服务暂时不可用，请稍后再试"
    return "服务暂时不可用，请稍后再试"

# ===== MySQL 对话历史（展示层） =====
def save_conversation(thread_id: str, role: str, content: str):
    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "root123"),
        database=os.getenv("MYSQL_DATABASE", "agent"),
        charset="utf8mb4"
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
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "root123"),
        database=os.getenv("MYSQL_DATABASE", "agent"),
        charset="utf8mb4"
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

    # checkpointer 持久化路径：优先用 DATA_DIR（容器里由 compose 设定），
    # 否则用 rag_api/data/（本地开发），避免 DB 文件散落在源码目录
    data_dir = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(data_dir / "checkpoints.db")
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
        print(f"[错误] {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=friendly_error(e))

    # 写 MySQL（展示层，失败不影响主流程）
    try:
        save_conversation(thread_id, "human", payload.message)
        save_conversation(thread_id, "ai", answer)
    except Exception as e:
        print(f"[MySQL 写入失败] {type(e).__name__}: {str(e)}")

    return ChatResponse(reply=answer)

@app.get("/history")
def get_history(thread_id: str = "user_001", limit: int = 20):
    # limit 上限保护
    limit = min(limit, 100)

    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "root123"),
        database=os.getenv("MYSQL_DATABASE", "agent"),
        charset="utf8mb4"
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

            # 写 MySQL（展示层，失败不影响主流程）
            try:
                save_conversation(thread_id, "human", payload.message)
                save_conversation(thread_id, "ai", full_answer)
            except Exception as e:
                print(f"[MySQL 写入失败] {type(e).__name__}: {str(e)}")

            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            # 客户端断连，不要 yield，直接抛出
            print("[请求] 客户端断连")
            raise
        except Exception as e:
            print(f"[错误] {type(e).__name__}: {str(e)}")
            yield f"data: {json.dumps({'error': friendly_error(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            elapsed = time.time() - start_time
            print(f"[请求] 耗时 {elapsed:.2f} 秒")

    return StreamingResponse(event_generator(), media_type="text/event-stream")