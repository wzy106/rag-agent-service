# 6. 为什么要用 Docker？

- 环境一致性：换机器不用担心依赖问题
- 隔离：不污染宿主机环境
- 部署简单：一条 `docker compose up -d`
- 可复现：镜像=代码+环境+依赖

**追问：你的 Docker 镜像多大？怎么优化的？**

> "从 3.51 GB 优化到 2.42 GB。三个关键点：
> ① **多阶段构建**：builder 阶段装 build-essential（编译器 300MB+），运行时不带，只 `COPY --from=builder /install /usr/local`；
> ② **CPU 版 torch**：原来默认装 CUDA 版，会拉 triton 248MB + cudnn 500MB + nccl 200MB + nvshmem 100MB，换成 `--extra-index-url https://download.pytorch.org/whl/cpu` 后，bge-small 这种小模型在 CPU 上毫秒级，根本用不上 GPU；
> ③ **模型下载移出构建期**：原来 Dockerfile 里有一行 `RUN python -c "...下载 embedding 模型..."`，构建耗时 33 分钟且容易网络超时。改成运行时 volume 挂载 `hf_cache`，构建期不下载。"
