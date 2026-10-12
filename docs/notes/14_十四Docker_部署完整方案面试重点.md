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
| 19:10 | 清理 C 盘 | Free 10 GB |
| 19:20 | 项目搬到 D:\Projects\Agent | robocopy 绕过文件锁 |
| 19:30 | Docker 数据迁移 | Settings → `D:\Docker` → C 盘 65 GB |
| 19:50 | 重建 Dockerfile | 构建成功，2.42 GB |
| 20:00 | 启动容器 | ✅ |
| 20:10 | 首次 `/chat` 报 `FileNotFoundError` | MCP Server 找不到模型缓存 |
| 20:20 | 宿主机 `snapshot_download` | 192 MB 下完 |
| 20:30 | 重启容器 | ✅ |
| 21:00 | 发现 `docker compose down` 后记忆全丢 | checkpoints 无 volume |
| 21:10 | 改 `DATA_DIR` + compose 加 volume | ✅ |

**2026-10-09**：抽公共模块 + 加随机基线

**2026-10-10**：正负样本标定 MAX_DISTANCE 1.1→1.0

**2026-10-11**：Rerank 精排 + 语料 2→96 篇 + 测试集 30→249 条

**2026-10-12**：阈值重标定 MAX_DISTANCE 1.0→0.96；notes/07 拆成 25 个独立面试题文件，语料同步为 98 篇 / 576 chunk，RAG 评估更严格

### 14.5 面试话术（Docker 部署）

> “我的项目用 Docker 多阶段构建 + CPU 版 torch，镜像从 3.51 GB 优化到 2.42 GB。Docker 数据落在 D 盘避免占满 C 盘。MCP Server 需要两个模型：bge-small-zh-v1.5（embedding，192 MB）和 bge-reranker-base（rerank，3.36 GB），我都下到宿主机 `hf_cache/` 然后 volume 挂载进容器。checkpoints.db 也挂到宿主机 `./data/`，这样 `docker compose down` 后容器重建，Agent 的记忆也不会丢。检索逻辑抽到 `rag_core/search.py`，服务端和评估脚本共用同一份代码。降级阈值 MAX_DISTANCE 通过正负样本距离分布标定为 0.96，P=0.990、R=1.000、F1=0.995，幻觉率约 1.0%。检索用两阶段 pipeline：混合检索召回 Top-20 + Rerank 精排 Top-3，召回率从 54.6% 提到 60.6%。当前知识库 98 篇文档 / 576 chunk，评估集 249 条，随机基线 3.1%。docker compose up -d 一条命令就能起完整服务：MySQL + Agent API。”
