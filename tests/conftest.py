"""pytest 共享夹具。

注意：`API_KEYS` 必须在 import `rag_api.api_agent` 之前设置——租户列表是模块导入时读的。
用固定测试 key，不依赖真实 `.env`（`load_dotenv` 默认不覆盖已存在的环境变量）。
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["API_KEYS"] = "key_alice,key_bob"


class _FakeResult(dict):
    """模拟 LangGraph 的返回值：代码里用 result["messages"][-1].content。"""


class _FakeMessage:
    def __init__(self, content):
        self.content = content
        self.type = "ai"


class _FakeAgent:
    """假 Agent：不调真实 LLM，只把 config 记下来供断言。"""

    def __init__(self, sink):
        self._sink = sink

    async def ainvoke(self, payload, config=None):
        self._sink["configs"].append(config)
        return _FakeResult(messages=[_FakeMessage("好的")])


class _FakeCursor:
    def __init__(self, sink):
        self._sink = sink

    def execute(self, sql, params=None):
        self._sink["queries"].append((" ".join(sql.split()), params))

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    """假 MySQL 连接：只记录执行过的 SQL，不真的连库。"""

    def __init__(self, sink):
        self._sink = sink

    def cursor(self):
        return _FakeCursor(self._sink)

    def commit(self):
        pass

    def close(self):
        pass


class _FakeRedis:
    def __init__(self):
        self.counts = {}

    def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def expire(self, key, ttl):
        pass


@pytest.fixture
def api(monkeypatch):
    """打桩后的 api_agent 模块 + 记录到的调用（不连真实 LLM / MySQL / Redis）。"""
    import rag_api.api_agent as m

    sink = {"queries": [], "configs": []}
    monkeypatch.setattr(m, "agent", _FakeAgent(sink))
    monkeypatch.setattr(m, "_connect", lambda: _FakeConn(sink))
    monkeypatch.setattr(m, "_get_redis", lambda: _FakeRedis())
    return m, sink


@pytest.fixture
def client(api):
    """TestClient。刻意不用 with —— 避免触发 lifespan（那会去拉 MCP 子进程 + 加载模型）。"""
    from fastapi.testclient import TestClient

    m, _ = api
    return TestClient(m.app)
