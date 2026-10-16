# 十一、MySQL 展示层 + Redis 预留（工程化）

### 为什么还需要 MySQL

| 问题 | 解法 |
|---|---|
| 用户要查看历史对话 | MySQL 展示层，给 /history 接口用 |
| Agent 的记忆 | 交给 LangGraph checkpointer，不经过 MySQL |

**角色划分**：
- **checkpointer（SQLite）**：Agent 的记忆，唯一真相源
- **MySQL**：只做展示层，给 `/history` 接口用
- **Redis**：暂时不用，预留做限流

### MySQL 对话历史（展示层）

```sql
CREATE TABLE conversations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    owner VARCHAR(32) NOT NULL,            -- 租户 id（API key 派生），用于隔离
    thread_id VARCHAR(64) NOT NULL,
    role VARCHAR(20),
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_owner_thread (owner, thread_id)
);
```

**数据流**：
```text
用户提问
    ↓
调 Agent（checkpointer 自动加载历史）
    ↓
写 MySQL 展示层（失败不影响主流程）
```

**关键设计**：
- MySQL 写入包 try/except，失败只打日志
- `/history` 接口加 `limit` 上限保护（最大 100）
- `/history` **需要鉴权**，且查询带 `owner = 当前租户` 条件——只返回自己租户的记录（防越权 IDOR）

### Redis 预留说明

**当前没用**。历史版本用过 Redis 缓存，但实测发现：
- 缓存 key 用 `context_version` 做版本号
- `context_version` = MySQL 消息数，每轮 +2
- 同一问题第二次必然 miss，**命中率结构性为 0**
- 是负优化，已删除

**下一步计划**：
- 用于限流（`incr` + `expire`）
- 或用于分布式缓存（多实例共享）

**为什么保留依赖**：
- `requirements.txt` 和 `docker-compose.db.yml` 里还留着
- 将来加限流时不用重新配置

### 完整数据流（当前版本）

```text
用户提问（带 thread_id）
    ↓
调 Agent（checkpointer 自动加载历史）
    ↓
写 MySQL 展示层（失败不影响主流程）
    ↓
返回答案
```

### 面试答题要点

| 问题 | 答案 |
|---|---|
| 对话历史怎么持久化？ | LangGraph checkpointer + AsyncSqliteSaver |
| MySQL 用来做什么？ | 展示层，给 /history 接口用 |
| 为什么删掉 Redis 缓存？ | 命中率结构性为 0，是负优化 |
| 为什么不删 Redis 依赖？ | 预留做限流（incr + expire） |
| 多实例部署怎么共享状态？ | checkpointer 换 PostgresSaver；限流用 Redis |

### 可优化方向

- Redis 加密码和连接池，生产环境必须
- 用连接池代替每次新建 MySQL 连接（性能优化）
- MySQL 写入改异步（aiomysql）
