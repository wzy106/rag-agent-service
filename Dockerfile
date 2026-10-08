# ---------- Stage 1: 构建依赖 ----------
FROM python:3.12-slim AS builder
WORKDIR /app

# 安装编译工具（只在 builder 阶段存在）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖清单
COPY requirements.txt .

# 阿里云源 + PyTorch CPU 源，装到 /install 前缀
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

# 从 builder 拷贝已装好的依赖，不带走编译工具链
COPY --from=builder /install /usr/local

# 模型缓存走 volume 挂载，不再在构建期下载
ENV HF_HOME=/root/.cache/huggingface
ENV HF_ENDPOINT=https://hf-mirror.com

# 复制知识库文档（MCP Server 启动时从这里加载）
COPY docs/ ./docs/

# 复制 FAISS 索引（避免容器内重新 embedding）
COPY faiss_index/ ./faiss_index/

# 复制 MCP Server 代码
COPY mcp_test/mcp_rag_server.py ./mcp_test/mcp_rag_server.py

# 复制 FastAPI 服务代码
COPY rag_api/api_agent.py ./rag_api/api_agent.py

EXPOSE 8000

CMD ["uvicorn", "rag_api.api_agent:app", "--host", "0.0.0.0", "--port", "8000"]