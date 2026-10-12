# 18. 为什么 pip 突然开始超时？

**触发点**：我改 Dockerfile 时，pip 命令加了 `--prefix=/install`，Docker 层缓存失效，必须重新下载所有依赖。

**原来为什么"不超时"**：之前的构建命中旧缓存层，pip 实际没走网络。

**超时原因**：torch 默认拉 CUDA 全家桶，光 triton 就 248 MB，加上 cudnn (500 MB) + nccl (200 MB) + nvshmem (100 MB)，总量 3~5 GB。清华源大包容易超时。

**解法**：
- 换阿里云源（比清华稳定）
- 加 `--timeout 300 --retries 10`
- **根本解法**：requirements.txt 加 `--extra-index-url https://download.pytorch.org/whl/cpu`，用 CPU 版 torch，下载量从 5 GB → 600 MB
