# 十四、Docker 部署完整方案（面试重点）

### 14.1 最终 Dockerfile

```dockerfile
# ---------- Stage 1: 构建依赖 ----------
FROM python:3.12-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 阿里云源 + PyTorch CPU 源
RUN pip install --no-cache-dir --prefix=/install \
        -r requirements.txt \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --trusted-host mirrors.aliyun.com \
        --trusted-host download.pytorch.org \
        --timeout 300 \
        --retries 10

# ---------- Stage 2: 运行时 ----------
FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /install /usr/local

ENV HF_HOME=/root/.cache/huggingface
ENV HF_ENDPOINT=https://hf-mirror.com

COPY docs/ ./docs/
COPY rag_core/ ./rag_core/
COPY faiss_index/ ./faiss_index/
COPY mcp_test/mcp_rag_server.py ./mcp_test/mcp_rag_server.py
COPY rag_api/api_agent.py ./rag_api/api_agent.py

EXPOSE 8000
CMD ["uvicorn", "rag_api.api_agent:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 14.2 最终 docker-compose.yml

```yaml
services:
  mysql:
    image: mysql:8
    container_name: agent-mysql
    ports:
      - "3306:3306"
    environment:
      MYSQL_ROOT_PASSWORD: root123
      MYSQL_DATABASE: agent
    volumes:
      - mysql_data:/var/lib/mysql
    networks:
      - agent-net

  agent-api:
    build: .
    container_name: agent-api
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - MYSQL_HOST=mysql
      - MYSQL_PORT=3306
      - MYSQL_USER=root
      - MYSQL_DATABASE=agent
      - HF_ENDPOINT=https://hf-mirror.com
      - HF_HOME=/root/.cache/huggingface
      - HF_HUB_OFFLINE=1
      - TRANSFORMERS_OFFLINE=1
      - DATA_DIR=/app/rag_api/data
    volumes:
      - ./hf_cache:/root/.cache/huggingface
      - ./data:/app/rag_api/data
    depends_on:
      - mysql
    networks:
      - agent-net
    restart: unless-stopped

volumes:
  mysql_data:

networks:
  agent-net:
    driver: bridge
```

### 14.3 requirements.txt 关键两行

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
```

**原理**：PyTorch 官方 CPU 源里 torch 版本是 `2.9.0+cpu`，按 PEP 440 规则 `2.9.0+cpu > 2.9.0`，pip 自动选 CPU 版。

### 14.4 部署踩坑时间线

**2026-10-08（Docker 部署问题闭环）**

| 时间 | 问题 | 动作 |
|---|---|---|
| 19:00 | `unpigz corrupted`，构建失败 | 查 C 盘 → 0 字节 |
| 19:10 | 清理 C 盘（Temp + SoftwareDistribution + HF cache） | Free 10 GB |
| 19:20 | 项目搬到 D:\Projects\Agent | Move-Item 被锁 → robocopy 绕过 |
| 19:30 | Docker 数据迁移 | Settings → `D:\Docker` → C 盘 65 GB |
| 19:50 | 重建 Dockerfile（多阶段 + 阿里云源 + CPU torch） | 构建成功，2.42 GB |
| 20:00 | 启动容器 | `docker compose up -d` ✅ |
| 20:10 | 首次 `/chat` 报 `FileNotFoundError: snapshots` | MCP Server 找不到模型缓存 |
| 20:20 | 宿主机 `snapshot_download` 到 `hf_cache/` | 192 MB 下完 |
| 20:30 | 重启容器 | `Application startup complete.` ✅ |
| 20:35 | 端到端测试（chat / stream / 多轮记忆） | 全过 ✅ |
| 21:00 | 发现 `docker compose down` 后记忆全丢 | checkpoints 无 volume |
| 21:10 | 改 `DATA_DIR` + compose 加 volume | 跨容器重建记忆保留 ✅ |

**2026-10-09（评估可复现 + 抽公共模块）**

| 时间 | 问题 | 动作 |
|---|---|---|
| 16:50 | 抽 `rag_core/search.py` 公共模块 | 服务端 import 改公共模块 ✅ |
| 17:00 | Dockerfile 加 `COPY rag_core/` | 容器重建成功 ✅ |
| 17:06 | `mcp_rag_server.py` 加 sys.path hack | 本地/容器都能 import ✅ |
| 17:10 | 评估脚本重写（加随机基线） | 跑出 84.1% ✅ |
| 17:15 | 删掉 `eval/hybrid_search_test.py` | 分叉实现清理 ✅ |
| 17:20 | Git 提交 + 推送 | `99af642` ✅ |

**2026-10-10（阈值标定）**

| 时间 | 问题 | 动作 |
|---|---|---|
| 19:20 | 构造 25 条负样本 | `eval/negatives.json` ✅ |
| 19:30 | 写标定脚本 | `eval/calibrate_distance.py` ✅ |
| 19:56 | 画两张图 | `eval/output/*.png` ✅ |
| 20:00 | 分析：1.1 偏松，F1 最大 1.06，P≥0.95 最高召回 1.00 | 决策选 1.00 ✅ |
| 20:10 | 改 `MAX_DISTANCE = 1.0` | 幻觉率 9.1% → 3.3% ✅ |
| 20:20 | 重跑评估，混合检索召回 100% → 96.7% | 有意取舍 ✅ |
| 20:30 | Git 提交 | `466e671` ✅ |

**2026-10-11（Rerank）**

| 时间 | 问题 | 动作 |
|---|---|---|
| 13:58 | 恢复 Docker 环境 | 容器都 Up ✅ |
| 14:00 | 下载 bge-reranker-base（3.36 GB） | 2 分 23 秒下完 ✅ |
| 14:10 | 改 `rag_core/search.py` 加 `rerank` + `hybrid_search_with_rerank` | import OK ✅ |
| 14:15 | 改 `mcp_rag_server.py` 用新 pipeline | 本地验证 OK ✅ |
| 14:20 | 改 `eval_retrieval.py` 加 Rerank 策略 | 5 策略对比 ✅ |
| 14:25 | 跑评估：精确率 0.86 → 0.90，MRR 0.94 → 0.97 | ✅ |
| 14:30 | 容器端到端测试 | ✅ |
| 14:35 | Git 提交 | `37d47f1` ✅ |

### 14.5 面试话术（Docker 部署）

> "我的项目用 Docker 多阶段构建 + CPU 版 torch，镜像从 3.51 GB 优化到 2.42 GB。Docker 数据落在 D 盘避免占满 C 盘。MCP Server 需要两个模型：bge-small-zh-v1.5（embedding，192 MB）和 bge-reranker-base（rerank，3.36 GB），我都下到宿主机 `hf_cache/` 然后 volume 挂载进容器。checkpoints.db 也挂到宿主机 `./data/`，这样 `docker compose down` 后容器重建，Agent 的记忆也不会丢。检索逻辑抽到 `rag_core/search.py`，服务端和评估脚本共用同一份代码。降级阈值 MAX_DISTANCE 通过正负样本距离分布标定为 1.0，幻觉率从 9.1% 降到 3.3%。检索用两阶段 pipeline：混合检索召回 Top-20 + Rerank 精排 Top-3，精确率从 0.86 提到 0.90。docker compose up -d 一条命令就能起完整服务：MySQL + Agent API。"
