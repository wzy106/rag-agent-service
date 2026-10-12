# 22. 为什么要让 HF_HOME 三级探测？

**问题**：MCP Server 是 `api_agent.py` 用 stdio 拉起的**子进程**，`MultiServerMCPClient` 起子进程时环境变量可能不完整传递，导致容器里 `HF_HOME` 为空，代码 fallback 到项目内 `hf_cache/`（容器里不存在），最终 `FileNotFoundError`。

**解法**：

```python
def _resolve_hf_home() -> str:
    if os.environ.get("HF_HOME"):           # 1. 显式环境变量
        return os.environ["HF_HOME"]
    container_default = Path("/root/.cache/huggingface")
    if container_default.exists():          # 2. 容器默认挂载点
        return str(container_default)
    return str(Path(__file__).resolve().parent.parent / "hf_cache")   # 3. 本地开发
```

**为什么不硬编码**：硬编码 `Path.home()/.cache/huggingface/.../snapshots` 依赖精确的目录结构，换环境必崩，且不易发现。

**为什么不只用环境变量**：子进程可能拿不到，本地开发时也常常忘了设。
