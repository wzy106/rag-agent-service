import os
import json
import time
import hmac
import hashlib
import asyncio
import pymysql
import redis
from typing import Annotated
from datetime import timedelta
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# 找 .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# ===== 护栏配置 =====
RECURSION_LIMIT = 10
MAX_TOKENS = 2000

# ===== API Key 鉴权（认证 + 租户识别）=====
# 支持逗号分隔的多个 key：每个 key 是一个独立租户（principal）。
# 只有一个 key 时，所有持有者属于同一租户——真正的多租户隔离需要一 key 一人。
# 未配置 API_KEYS 时回退到单个 API_KEY（向后兼容）。
_RAW_KEYS = os.getenv("API_KEYS") or os.getenv("API_KEY", "dev-key-change-me")
_API_KEYS = [k.strip() for k in _RAW_KEYS.split(",") if k.strip()]

def principal_of(token: str) -> str:
    """把 API key 映射成稳定的、不含密钥本身的租户 id（用于隔离各租户数据）。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]

def _match_key(token: str) -> str | None:
    """遍历所有已配置 key 做恒时比较，避免用 == 比密钥引入的时序侧信道。"""
    matched = None
    for key in _API_KEYS:
        if hmac.compare_digest(key, token):
            matched = key
    return matched

async def verify_key(authorization: str = Header(default="")) -> str:
    """校验 Bearer token，返回该 key 对应的租户 id（principal）——而不是 key 本身。

    认证（你是谁）≠ 授权（你能访问什么）：这里只负责认证并给出身份，
    具体能读到哪些数据，由各接口按 principal 过滤。
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="缺少 Authorization 头")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization 格式错误，应为 'Bearer <token>'")
    token = authorization[7:].strip()
    matched = _match_key(token)
    if matched is None:
        raise HTTPException(status_code=403, detail="API Key 无效")
    return principal_of(matched)

# ===== Redis 限流 =====
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))

_redis_client = None

def _get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _redis_client

def rate_limit(principal: str):
    """固定窗口限流：每分钟最多 RATE_LIMIT_PER_MINUTE 次，按租户（API key）计。

    Redis 不可用时放行（fail-open）——限流是保护措施，不能让它挂掉主流程。
    """
    try:
        r = _get_redis()
        redis_key = f"rate:{principal}"
        current = r.incr(redis_key)
        if current == 1:
            r.expire(redis_key, 60)
        if current > RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail=f"请求过于频繁，每分钟最多 {RATE_LIMIT_PER_MINUTE} 次，请稍后再试",
            )
    except HTTPException:
        # 限流本身触发，正常抛出
        raise
    except redis.RedisError as e:
        # Redis 挂了：放行，只打日志
        print(f"[限流] Redis 不可用，放行：{type(e).__name__}: {e}")

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
def _connect():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "root123"),
        database=os.getenv("MYSQL_DATABASE", "agent"),
        charset="utf8mb4",
    )

def save_conversation(owner: str, thread_id: str, role: str, content: str):
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (owner, thread_id, role, content) VALUES (%s, %s, %s, %s)",
                (owner, thread_id, role, content)
            )
        conn.commit()
    finally:
        conn.close()

def ensure_table():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    owner VARCHAR(32) NOT NULL,
                    thread_id VARCHAR(64) NOT NULL,
                    role VARCHAR(20),
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_owner_thread (owner, thread_id)
                )
            """)
            # 老库补 owner 列（幂等：列/索引已存在时报 1060/1061，忽略）
            for ddl in (
                "ALTER TABLE conversations ADD COLUMN owner VARCHAR(32) NOT NULL DEFAULT ''",
                "ALTER TABLE conversations ADD INDEX idx_owner_thread (owner, thread_id)",
            ):
                try:
                    cur.execute(ddl)
                except pymysql.err.OperationalError:
                    pass
        conn.commit()
    finally:
        conn.close()

# 全局 agent
agent = None

async def _prewarm(tools):
    """启动时跑一次检索，触发 MCP Server 里 Reranker 的懒加载，避免首个请求替我们付加载成本。"""
    search_tool = next((t for t in tools if t.name == "search_knowledge"), None)
    if search_tool is None:
        return
    try:
        t0 = time.time()
        await search_tool.ainvoke({"query": "预热"})
        print(f"[启动] 检索预热完成（{time.time() - t0:.1f}s）")
    except Exception as e:
        # 预热失败不阻断启动，首个真实请求再加载即可
        print(f"[启动] 预热失败（不影响启动）：{type(e).__name__}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    print("[启动] 正在连接 MCP Server...")

    client = MultiServerMCPClient({
        "rag": {
            "command": "python",
            "args": [str(Path(__file__).resolve().parent.parent / "mcp_test" / "mcp_rag_server.py")],
            "transport": "stdio",
            # 常驻 session 默认没有读超时；设一个上限，避免工具卡死时请求永久挂起
            "session_kwargs": {"read_timeout_seconds": timedelta(seconds=60)},
        }
    })

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
        # 常驻 session：MCP Server 子进程整个服务生命周期只拉起一次，模型/索引也只加载一次。
        # 注意不要用 client.get_tools()——它每次工具调用都会新建 session，
        # 等于每次调用都重新 spawn 子进程 + 重载全部模型（实测 35~44s/次）。
        async with client.session("rag") as session:
            tools = await load_mcp_tools(session)
            print(f"[启动] 从 MCP 加载了 {len(tools)} 个工具：{[t.name for t in tools]}")

            agent = create_agent(
                model,
                tools=tools,
                checkpointer=saver,
                system_prompt="你是一个知识助手，回答问题时优先调用 search_knowledge 工具查询知识库。"
            )
            print(f"[启动] Agent 创建完成（持久化记忆 + 护栏：recursion_limit={RECURSION_LIMIT}, max_tokens={MAX_TOKENS}）")

            ensure_table()
            print("[启动] MySQL 表检查完成")

            await _prewarm(tools)

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

# thread_id 必填且限长：不设默认值，避免所有调用方不传时挤进同一个会话（串记忆/串历史）
ThreadId = Annotated[str, Field(min_length=1, max_length=48, description="会话 id，同一租户内用于区分会话")]

class ChatRequest(BaseModel):
    message: str
    thread_id: ThreadId

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def root():
    return {"message": "Agent API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, principal: str = Depends(verify_key)):
    rate_limit(principal)
    thread_id = payload.thread_id
    # 记忆按租户命名空间隔离：同一个 thread_id 在不同 key 下互不影响
    effective_thread = f"{principal}:{thread_id}"

    config = {
        "configurable": {"thread_id": effective_thread},
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
        save_conversation(principal, thread_id, "human", payload.message)
        save_conversation(principal, thread_id, "ai", answer)
    except Exception as e:
        print(f"[MySQL 写入失败] {type(e).__name__}: {str(e)}")

    return ChatResponse(reply=answer)

@app.get("/history")
def get_history(thread_id: str, limit: int = 20, principal: str = Depends(verify_key)):
    # 只返回「调用方自己租户」下的会话记录，避免越权读别人的历史
    limit = min(max(limit, 1), 100)

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content, created_at FROM conversations "
                "WHERE owner = %s AND thread_id = %s ORDER BY created_at DESC LIMIT %s",
                (principal, thread_id, limit)
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
    thread_id: ThreadId

@app.post("/chat/stream")
async def chat_stream(payload: StreamRequest, principal: str = Depends(verify_key)):
    rate_limit(principal)
    start_time = time.time()
    thread_id = payload.thread_id
    effective_thread = f"{principal}:{thread_id}"
    print(f"[请求] 收到消息：{payload.message}（thread_id={thread_id}）")

    async def event_generator():
        try:
            saw_tool = False
            pending = []
            started = False
            full_answer = ""

            config = {
                "configurable": {"thread_id": effective_thread},
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
                save_conversation(principal, thread_id, "human", payload.message)
                save_conversation(principal, thread_id, "ai", full_answer)
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