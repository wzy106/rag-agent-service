# 故事 27：checkpoints.db 无 volume，容器重建丢记忆（2026-10-08）

### 问题
README 宣传"服务重启不丢历史"，但**只在进程重启场景下成立**。`docker compose down` 后：
- 容器销毁
- `rag_api/checkpoints.db` 随之消失
- 用户的历史对话全丢

**这是一个很容易忽略的边界**：restart ≠ down。

### 分析
原 `docker-compose.yml` 只挂了一个 volume：
```yaml
volumes:
  - ./hf_cache:/root/.cache/huggingface   # 只挂了 HF 缓存
  # 缺 checkpoints volume
```

`api_agent.py` 里的路径：
```python
checkpoint_path = str(Path(__file__).resolve().parent / "checkpoints.db")
```
**落在源码目录里，宿主机看不到。**

### 方案

**① 宿主机建目录：**
```cmd
mkdir D:\Projects\Agent\data
```

**② 改 `api_agent.py`：**
```python
# checkpointer 持久化路径：优先用 DATA_DIR（容器里由 compose 设定），
# 否则用 rag_api/data/（本地开发）
data_dir = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent / "data"))
data_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = str(data_dir / "checkpoints.db")
async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
    ...
```

**③ 改 `docker-compose.yml`：**
```yaml
environment:
  - DATA_DIR=/app/rag_api/data
volumes:
  - ./hf_cache:/root/.cache/huggingface
  - ./data:/app/rag_api/data       # 新增
```

**④ 把旧的 db 挪到新位置：**
```cmd
move rag_api\checkpoints.db data\checkpoints.db
```

**⑤ `.gitignore` 加 `data/`：**
```
data/
```

### 验证
```cmd
:: 1. 问一个问题
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"MMR 是什么？\", \"thread_id\": \"p0b_test\"}"

:: 2. 完全销毁容器
docker compose down

:: 3. 重建
docker compose up -d

:: 4. 用代词问
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"它和普通相似度有什么区别？\", \"thread_id\": \"p0b_test\"}"
```

**第 4 步能理解"它"= MMR → 记忆持久化生效。**

**宿主机 `data/checkpoints.db` 大小会实时变化，证明写入生效。**

### 为什么挂目录，不挂文件
**不要**用这种写法：
```yaml
- ./checkpoints.db:/app/rag_api/checkpoints.db      # ❌ 危险
```

**原因**：如果宿主机 `./checkpoints.db` 不存在，Docker **会创建一个同名目录**挂进去，导致 SQLite 打不开。挂目录比挂文件稳。

### 收获
**Docker 持久化有三个层次，缺一不可**：
1. **代码层**：用环境变量（`DATA_DIR`）决定路径，不用硬编码
2. **compose 层**：volume 挂载到宿主机
3. **git 层**：`.gitignore` 忽略运行时文件（`data/`）

**"服务重启不丢"这句话要拆开看**：
- `docker compose restart` → 只重启进程，容器不销毁，DB 还在容器里 ✅
- `docker compose down` → 销毁容器，DB 一起消失 ❌
- `docker compose down && up` → 除非有 volume，否则记忆全丢

**面试时能讲**：
> "Docker 化的一个坑是 checkpoints 持久化。`AsyncSqliteSaver` 默认把 DB 写在容器内部，`docker compose down` 就全丢了。我的做法是用 `DATA_DIR` 环境变量决定路径，compose 里把 `./data` 挂到 `/app/rag_api/data`。本地开发 fallback 到 `rag_api/data/`。验证过：`docker compose down` 后 `up -d`，用代词问，Agent 还能回忆起上一轮。"

---
