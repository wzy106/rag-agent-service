# 故事 7：Docker 构建慢

### 问题
第一次 `docker compose up --build` 花了 **19 分钟**。

### 分析
Dockerfile 把所有步骤写在一起，每次构建都要重新装依赖、下载模型。

### 方案
利用 Docker 层缓存，把不常变的部分放前面：
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .                              # 先复制依赖清单
RUN pip install -r requirements.txt                  # 装依赖（这层缓存）
ENV HF_ENDPOINT=https://hf-mirror.com
RUN python -c "...SentenceTransformer(...)"          # 下载模型（这层缓存）
COPY mcp_test/mcp_rag_server.py ./mcp_test/          # 再复制代码
COPY rag_api/api_agent.py ./rag_api/
```

### 为什么这么选
- **层缓存**：代码改了不影响依赖层，第二次构建几秒
- **镜像可复现**：模型和依赖都打包进镜像，换机器一致

### 收获
Dockerfile 的指令顺序影响构建效率。把变化频率低的放前面，充分利用层缓存。

### 更新（2026-10-08）
本次构建又踩了新坑——把模型下载放在构建期会导致网络不稳定时构建失败。**最终方案**：模型下载移出构建期，改由运行时 volume 挂载 `hf_cache`。详见故事 26。

---
