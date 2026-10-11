# 23. 为什么要抽公共 `hybrid_search` 模块？

**问题**：评估脚本和服务端各有一份 `hybrid_search` 实现，参数不一样（评估脚本缺距离降级），所以"评估跑的数字"描述的不是"部署的系统"。

**修法**：抽到 `rag_core/search.py`，服务端和评估脚本都 `from rag_core.search import hybrid_search`。

```python
# rag_core/search.py
def hybrid_search(query, vectorstore, bm25, all_docs, k=3, ...):
    ...
```

**Dockerfile 也要加 COPY**：
```dockerfile
COPY rag_core/ ./rag_core/
```

**`mcp_rag_server.py` 加 sys.path hack**（脚本在 `mcp_test/` 子目录，找不到项目根的 `rag_core`）：
```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_core.search import hybrid_search, ...
```

**收益**：
- 评估结果真实反映部署行为
- 改一处，两端生效（比如调 `MAX_DISTANCE` 只需改一处）
- 面试报的每个数字都能被复现
- Dockerfile 加了 COPY，容器内也能 import

**面试话术**：
> "我之前评估脚本和服务端各写了一份 hybrid_search，参数还不一样——评估那份缺距离降级。发现后把检索逻辑抽到 rag_core/search.py，两边都 import，Dockerfile 也加了 COPY。这样评估脚本跑 python eval_retrieval.py 得到的数字就是部署系统的真实表现。"
