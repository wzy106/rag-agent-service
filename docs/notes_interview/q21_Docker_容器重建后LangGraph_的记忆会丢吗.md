# 21. Docker 容器重建后，LangGraph 的记忆会丢吗？

**会，如果不做持久化。**

`AsyncSqliteSaver` 默认把 `checkpoints.db` 写到容器内部，`docker compose down` 一销毁就全没了。README 里说"服务重启不丢历史"，对**进程重启**成立，对**容器重建**不成立。

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
4. 用代词问 → 能理解"它"指上一轮 MMR ✅

**为什么不直接挂载 `./checkpoints.db:/app/rag_api/checkpoints.db`**：如果宿主机文件不存在，Docker 会**创建一个同名目录**挂进去，导致 SQLite 打不开。挂目录比挂文件稳。
