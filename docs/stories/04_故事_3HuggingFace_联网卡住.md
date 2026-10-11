# 故事 3：HuggingFace 联网卡住

### 问题
跑 RAG 时，加载本地 embedding 模型卡住：
```text
[WinError 10060] 由于连接方在一段时间后没有正确答复...
Retrying in 1s [Retry 1/5].
```

### 分析
`SentenceTransformer` 默认会去 huggingface.co 检查模型更新。国内网络连不上，重试 5 次后崩溃。

### 方案
设两个环境变量：
```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```
强制离线，只用本地缓存。

### 为什么这么选
- **本地已有模型**：之前下载过 `bge-small-zh-v1.5`，不需要联网
- **零成本**：两行环境变量，不用改其他代码
- **稳定**：开发阶段完全离线，不受网络影响

### 收获
第三方库的默认行为（自动联网检查）在国内网络环境下需要显式关闭，环境变量是最轻量的方式。

---
