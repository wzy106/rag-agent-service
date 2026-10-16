"""错误映射、限流（含 fail-open）等不依赖模型的单元测试。"""
import asyncio

import pymysql
import pytest
import redis
from fastapi import HTTPException

from rag_api import api_agent as m


class _FakeRedis:
    def __init__(self):
        self.counts = {}

    def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def expire(self, key, ttl):
        pass


# ---------- 友好错误映射 ----------

def test_friendly_error_maps_known_types():
    class AuthError(Exception):
        status_code = 401

    class RateError(Exception):
        status_code = 429

    assert "认证失败" in m.friendly_error(AuthError())
    assert "频繁" in m.friendly_error(RateError())
    assert "超时" in m.friendly_error(asyncio.TimeoutError())
    assert "存储" in m.friendly_error(pymysql.err.OperationalError(2003, "down"))
    assert "服务暂时不可用" in m.friendly_error(ValueError("boom"))


def test_friendly_error_does_not_leak_internal_details():
    msg = m.friendly_error(RuntimeError("api_key=sk-secret /path/to/secret"))
    assert "sk-secret" not in msg and "/path/to/secret" not in msg


# ---------- 限流 ----------

def test_rate_limit_blocks_after_quota(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(m, "RATE_LIMIT_PER_MINUTE", 3)
    monkeypatch.setattr(m, "_get_redis", lambda: fake)
    for _ in range(3):
        m.rate_limit("tenant-a")               # 配额内放行
    with pytest.raises(HTTPException) as ei:
        m.rate_limit("tenant-a")
    assert ei.value.status_code == 429


def test_rate_limit_is_per_tenant(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(m, "RATE_LIMIT_PER_MINUTE", 1)
    monkeypatch.setattr(m, "_get_redis", lambda: fake)
    m.rate_limit("tenant-a")
    m.rate_limit("tenant-b")                   # 不同租户各有配额，不应抛
    with pytest.raises(HTTPException):
        m.rate_limit("tenant-a")


def test_rate_limit_fails_open_when_redis_down(monkeypatch):
    class _BrokenRedis:
        def incr(self, key):
            raise redis.exceptions.ConnectionError("redis down")

    monkeypatch.setattr(m, "_get_redis", lambda: _BrokenRedis())
    m.rate_limit("tenant-a")                   # Redis 挂了应放行，不抛异常


def test_rate_limit_key_does_not_leak_secret(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(m, "_get_redis", lambda: fake)
    m.rate_limit(m.principal_of("key_alice"))
    assert all("key_alice" not in key for key in fake.counts)
