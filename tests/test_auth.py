"""鉴权与租户隔离（对应 OWASP A01 Broken Access Control / 越权 IDOR 的修复）。"""
from rag_api.api_agent import _match_key, principal_of

ALICE = {"Authorization": "Bearer key_alice"}
BOB = {"Authorization": "Bearer key_bob"}


# ---------- 纯函数：租户身份 ----------

def test_principal_is_deterministic_and_hides_key():
    p = principal_of("key_alice")
    assert p == principal_of("key_alice")      # 稳定
    assert "key_alice" not in p                # 不含密钥本身
    assert len(p) == 16


def test_principal_differs_per_key():
    assert principal_of("key_alice") != principal_of("key_bob")


def test_match_key_is_exact():
    assert _match_key("key_alice") == "key_alice"
    assert _match_key("key_alic") is None      # 前缀
    assert _match_key("key_aliceX") is None    # 后缀
    assert _match_key("Xkey_alice") is None
    assert _match_key("") is None
    assert _match_key("KEY_ALICE") is None     # 大小写敏感


# ---------- 接口：/history 越权 ----------

def test_history_without_token_is_401(client):
    assert client.get("/history", params={"thread_id": "t1"}).status_code == 401


def test_history_with_malformed_header_is_401(client):
    r = client.get("/history", params={"thread_id": "t1"}, headers={"Authorization": "key_alice"})
    assert r.status_code == 401


def test_history_with_wrong_token_is_403(client):
    r = client.get("/history", params={"thread_id": "t1"}, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 403


def test_history_ok_with_valid_token(client):
    r = client.get("/history", params={"thread_id": "t1"}, headers=ALICE)
    assert r.status_code == 200


def test_history_query_filters_by_owner(api, client):
    m, sink = api
    client.get("/history", params={"thread_id": "t1"}, headers=ALICE)
    sql, params = sink["queries"][-1]
    assert "owner = %s AND thread_id = %s" in sql
    assert params == (principal_of("key_alice"), "t1", 20)


# ---------- 输入校验：thread_id 必填 ----------

def test_missing_thread_id_rejected(client):
    assert client.post("/chat", json={"message": "hi"}, headers=ALICE).status_code == 422


def test_blank_thread_id_rejected(client):
    assert client.post("/chat", json={"message": "hi", "thread_id": ""}, headers=ALICE).status_code == 422


def test_too_long_thread_id_rejected(client):
    assert client.post("/chat", json={"message": "hi", "thread_id": "x" * 49}, headers=ALICE).status_code == 422


# ---------- 租户隔离 ----------

def test_same_thread_id_isolated_across_tenants(api, client):
    m, sink = api
    client.post("/chat", json={"message": "hi", "thread_id": "t1"}, headers=ALICE)
    client.post("/chat", json={"message": "hi", "thread_id": "t1"}, headers=BOB)
    a = sink["configs"][-2]["configurable"]["thread_id"]
    b = sink["configs"][-1]["configurable"]["thread_id"]
    assert a == f"{principal_of('key_alice')}:t1"
    assert b == f"{principal_of('key_bob')}:t1"
    assert a != b                              # 同一个 thread_id，两个租户互不影响


def test_conversation_write_records_owner(api, client):
    m, sink = api
    client.post("/chat", json={"message": "hi", "thread_id": "t1"}, headers=ALICE)
    inserts = [q for q in sink["queries"] if q[0].startswith("INSERT")]
    assert inserts
    assert all(q[1][0] == principal_of("key_alice") for q in inserts)
