# 21. Docker 容器重建后，LangGraph 的记忆会丢吗？

**会，如果不做持久化。**

`AsyncSqliteSaver` 默认把 `checkpoints.db` 写到容器内部，`docker compose down` 一销毁就全没了。README 里说"服务重启不丢历史"，对**进程重启**成立，对**容器重建**不成立——这是一个很容易忽略的边界。

**修法**：把 checkpoints 目录挂载到宿主机。

```python
# api_agent.py：用 DATA_DIR 环境变量决定路径
data_dir = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent / "data"))
data_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = str(data_dir / "checkpoints.db")
async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
    ...
```

```yaml
# docker-compose.yml：加 volume + 环境变量
environment:
  - DATA_DIR=/app/rag_api/data
volumes:
  - ./data:/app/rag_api/data
```

**验证方法**：
1. 问一个问题
2. `docker compose down`（完全销毁容器）
3. `docker compose up -d`
4. 用代词问（"它和普通相似度有什么区别？"）→ 能理解"它"指上一轮的 MMR ✅

**宿主机 `data/checkpoints.db` 大小会实时变化**，证明写入生效。

**为什么不直接挂载 `./checkpoints.db:/app/rag_api/checkpoints.db`**：如果宿主机文件不存在，Docker 会**创建一个同名目录**挂进去，导致 SQLite 打不开。挂目录比挂文件稳。

**面试话术**：
> "Docker 化的一个坑是 checkpoints 持久化。`AsyncSqliteSaver` 默认把 DB 写在容器内部，`docker compose down` 就全丢了。我的做法是用 `DATA_DIR` 环境变量决定路径，compose 里把 `./data` 挂到 `/app/rag_api/data`。本地开发 fallback 到 `rag_api/data/`。验证过：`docker compose down` 后 `up -d`，用代词问，Agent 还能回忆起上一轮。"
