import pymysql
import redis

# ===== MySQL 测试 =====
print("=== MySQL ===")
conn = pymysql.connect(
    host="localhost",
    port=3306,
    user="root",
    password="root123",
    database="agent",
    charset="utf8mb4"
)
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
    cur.execute(
        "INSERT INTO conversations (thread_id, role, content) VALUES (%s, %s, %s)",
        ("user_001", "human", "MCP 是什么？")
    )
    conn.commit()
    cur.execute("SELECT id, thread_id, role, content FROM conversations WHERE thread_id = %s", ("user_001",))
    for row in cur.fetchall():
        print(row)
conn.close()

# ===== Redis 测试 =====
print("\n=== Redis ===")
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

r.set("test_key", "hello")
print("get:", r.get("test_key"))

r.set("cache:abc", "缓存内容", ex=60)
print("set with ex:", r.get("cache:abc"))

r.delete("counter:user_001")
r.incr("counter:user_001")
r.incr("counter:user_001")
r.expire("counter:user_001", 60)
print("incr:", r.get("counter:user_001"))
print("ttl:", r.ttl("counter:user_001"))