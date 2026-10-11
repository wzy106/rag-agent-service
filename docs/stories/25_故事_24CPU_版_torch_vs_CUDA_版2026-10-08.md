# 故事 24：CPU 版 torch vs CUDA 版（2026-10-08）

### 问题
用默认的 `pip install torch` 会拉**CUDA 全家桶**：

| 包 | 大小 |
|---|---|
| torch | ~900 MB |
| triton | 248 MB |
| nvidia-cudnn-cu13 | ~500 MB |
| nvidia-nccl-cu13 | ~200 MB |
| nvidia-nvshmem-cu13 | ~100 MB |
| nvidia-cusparselt-cu13 | ~100 MB |
| **合计** | **3~5 GB** |

**下载 5 GB 在镜像源上很容易超时。**

### 分析
- 本项目只用 **bge-small-zh-v1.5**（24 MB 小模型）做 embedding
- RAG 检索走 FAISS（CPU）
- LLM 推理走 DeepSeek API（云端，不在容器里跑）
- **容器里唯一用 torch 的地方就是 embedding，CPU 毫秒级**
- **装 CUDA 全家桶是杀鸡用牛刀**

### 方案
在 `requirements.txt` 顶部加两行：
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
```

**原理**：PyTorch 官方 CPU 源里 torch 版本是 `2.x.x+cpu`。按 PEP 440 版本规则，`2.9.0+cpu > 2.9.0`，**pip 会自动选 CPU 版**。

### 效果对比

| 项目 | CUDA 版 | CPU 版 |
|---|---|---|
| torch | 900 MB + 一堆 CUDA 依赖 | **126 MB** |
| triton | 248 MB | **不装** |
| nvidia-cudnn-cu13 | ~500 MB | **不装** |
| nvidia-nccl-cu13 | ~200 MB | **不装** |
| nvidia-nvshmem-cu13 | ~100 MB | **不装** |
| **总下载量** | **3~5 GB** | **~600 MB** |
| **镜像大小** | 3.51 GB | **2.42 GB** |

### 为什么这么选
- **性能需求**：本项目用不到 GPU，CPU 版完全够用
- **下载稳定**：600 MB 在 300 秒超时内稳定完成
- **镜像小**：少了 1.1 GB
- **构建快**：从 33 分钟降到 5~10 分钟

### 收获
**装依赖前先问："这个项目真的需要 GPU 吗？"** 很多 RAG / Agent 项目只是用 embedding 模型，CPU 完全够用。**默认装 CUDA 版 = 拉 5 GB 垃圾进镜像。**

**面试时能讲**：
> "我把 torch 从默认的 CUDA 版换成 CPU 版，下载量从 5 GB 砍到 600 MB，镜像从 3.5 GB 瘦到 2.4 GB。因为我们项目只用 bge-small 做 embedding，24 MB 的小模型在 CPU 上毫秒级，根本不需要 GPU。改法是在 requirements.txt 顶部加一行 `--extra-index-url https://download.pytorch.org/whl/cpu`。"

---
