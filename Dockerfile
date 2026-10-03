# 基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（sentence-transformers 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖清单，先装依赖（利用 Docker 缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 用镜像源下载 embedding 模型
ENV HF_ENDPOINT=https://hf-mirror.com
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# 复制项目文件
COPY mcp_test/mcp_rag_server.py ./mcp_test/mcp_rag_server.py
COPY rag_api/api_agent.py ./rag_api/api_agent.py

# 暴露端口
EXPOSE 8000

# 启动
CMD ["uvicorn", "rag_api.api_agent:app", "--host", "0.0.0.0", "--port", "8000"]