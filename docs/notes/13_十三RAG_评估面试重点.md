# 十三、RAG 评估（面试重点）

### 评估方法

构建 30 个测试项的测试集，每个标注期望命中的文档。
问题故意不包含文档关键词，测试语义检索能力。
包含单文档问题和多文档问题（一个 query 对应多个相关文档）。

### 评估脚本核心

```python
def evaluate_full(search_fn, name, k=3):
    hit = 0
    precision_sum = 0
    mrr_sum = 0
    diversity_sum = 0

    for item in test_set:
        results = search_fn(item["question"], k)
        sources = [doc.metadata["source"] for doc in results]
        unique_sources = set(sources)

        # 召回率
        if any(exp in sources for exp in item["expected_sources"]):
            hit += 1

        # 精确率
        if sources:
            relevant = sum(1 for s in sources if s in item["expected_sources"])
            precision_sum += relevant / len(sources)

        # MRR
        for rank, s in enumerate(sources, 1):
            if s in item["expected_sources"]:
                mrr_sum += 1 / rank
                break

        # 多样性
        diversity_sum += len(unique_sources)

    n = len(test_set)
    return {
        "recall": hit / n,
        "precision": precision_sum / n,
        "mrr": mrr_sum / n,
        "diversity": diversity_sum / n,
    }
```

### 随机基线（组合公式精确计算）

```python
from math import comb
from collections import Counter

def random_baseline(test_set, all_docs, k=3):
    """
    从全部 chunk 里随机抽 k 个，算 source 级召回率期望。
    P(至少命中一个期望 source) = 1 - C(N - n_expected, k) / C(N, k)
    """
    n_total = len(all_docs)
    source_counts = Counter(doc.metadata["source"] for doc in all_docs)

    total = 0.0
    for item in test_set:
        n_expected = sum(source_counts.get(s, 0) for s in item["expected_sources"])
        prob = 1 - comb(n_total - n_expected, k) / comb(n_total, k)
        total += prob
    return total / len(test_set)
```

### 评估结果（当前 2 个文档 / 102 chunk）

| 策略 | 召回率 | 精确率 | MRR | 平均多样性 |
|---|---|---|---|---|
| 普通相似度 | 100% | 0.87 | 0.96 | 1.37 |
| MMR | 100% | 0.83 | 0.97 | 1.43 |
| MMR + 按来源去重 | 100% | 0.77 | 0.97 | **1.57** |
| 混合检索（带降级 1.0） | 96.7% | 0.86 | 0.94 | 1.30 |
| **混合检索 + Rerank** | 96.7% | **0.90** | **0.97** | 1.17 |
| 随机基线 | **84.1%** | — | — | — |

> **可复现**：`python eval/eval_retrieval.py` 一条命令跑出上面所有数字（含随机基线 + Rerank）。评估脚本从 `rag_core.search` import 检索函数，和服务端部署的是同一份代码。

### 关键发现

**1. Rerank 是精确率的最后一块拼图**

- 混合检索精确率 0.86 → Rerank 后 0.90
- MRR 0.94 → 0.97
- **但召回率不变**（Rerank 只重排，不召回）

**2. Rerank 收益被语料规模压制**

- 只有 2 文档 / 102 chunk，判别空间有限
- 扩到 500+ chunk 才能发挥 Reranker 全部威力
- **主动承认这个局限**，比夸张提升更可信

**3. 精确率和多样性呈明显 trade-off**

- 相似度/混合检索：精确率高，多样性低
- MMR + 去重：多样性最高（1.57），精确率最低（0.77）
- Rerank 进一步压低多样性（1.17）——只选最相关的

**4. 选哪个策略取决于业务**

| 场景 | 推荐策略 |
|---|---|
| 追求精确（如事实查询） | **混合检索 + Rerank** |
| 需要多角度（如分析任务） | MMR + 去重 |
| 平衡 | MMR |

### 早期评估（5 个文档 / 24 chunk）

| 策略 | 召回率 | 平均多样性 |
|---|---|---|
| 普通相似度 | 29/30 = 97% | 2.00 |
| MMR | 28/30 = 93% | 2.50 |
| MMR + 按来源去重 | — | 3.00 |

**核心结论**：MMR 用 4% 的召回率换取 25% 的多样性提升。

### 失败案例分析

**案例 1**："怎么让回答一个字一个字显示出来？"
- 期望：`deployment.md`
- 实际：`langchain.md`、`rag.md`
- 原因：口语化表达和文档术语有语义鸿沟

**案例 2**："多 Agent 系统怎么保证消息不丢？"
- 期望：`langgraph.md`、`deployment.md`
- 实际：`langchain.md`、`rag.md`
- 原因：MMR 强制 chunk 级多样性时挤掉了最相关的文档

### 已知不足

**评估层面**：
- **文档级召回而非 chunk 级**：用 `source`（文件名）判断命中，只有两三篇文档，命中太容易，含金量不高。生产环境应该标注 chunk id 或关键句。
- **30 条样本太少**：3% 的差异（97% vs 93%）是一条样本的差别，统计上无意义。应该扩到 200 条以上。
- **没有端到端指标**：检索对了不等于回答对了。应该加 LLM-as-judge 或人工标注的 answer correctness。
- **随机基线 84.1%**：语料太小（2 文档/102 chunk），天花板效应明显，判别空间只有 15.9 个点。**这也是 Rerank 提升有限的原因。**

**架构层面**：
| 位置 | 问题 | 生产级解法 |
|---|---|---|
| `/chat` 和 `/chat/stream` | pymysql 是**同步客户端**，在 async 路径上会阻塞事件循环 | 用 `aiomysql` |
| `save_conversation` | 每次请求新建连接，没有连接池 | 用连接池（如 `DBUtils`） |
| ~~`MAX_DISTANCE = 1.1`~~ | ~~阈值校准不够系统，是拍脑袋定的~~ | ✅ 2026-10-10 已修复：标定为 1.0（P=0.967, R=0.967） |
| 流式状态机 | 用三个布尔变量表达状态，边界情况可能出错 | 改用 `metadata["langgraph_node"]` 做来源判断 + 显式状态机 |
| MCP Server 用了私有 API | `vectorstore.docstore._dict` 是内部 API，版本升级可能坏 | 构建索引时自己维护 chunks 列表并 pickle 落盘 |
| ~~Docker 部署~~ | ~~Dockerfile 缺 COPY docs/，compose 缺 network，host 写死 localhost~~ | ✅ 2026-10-08 已修复 |
| ~~HF 缓存路径硬编码~~ | ~~`Path.home()/.cache/huggingface/...` 依赖精确路径~~ | ✅ 2026-10-08 已修复：三级探测 |
| ~~checkpoints 无 volume~~ | ~~`docker compose down` 后记忆全丢~~ | ✅ 2026-10-08 已修复：挂载到 `./data/` |
| ~~评估代码分叉~~ | ~~评估脚本和服务端各有一份 hybrid_search，参数不一致~~ | ✅ 2026-10-09 已修复：抽到 `rag_core/search.py` |
| **Rerank 收益被语料规模压制** | **只有 2 文档 102 chunk，判别空间有限** | **扩语料到 500+ chunk** |

### 改进方案

| 方案 | 原理 | 状态 |
|---|---|---|
| 混合检索 | 向量 + BM25，加权合并 | ✅ 已做 |
| 降级策略 | 距离阈值拦截无关问题 | ✅ 已做 |
| 多指标评估 | 召回率 + 精确率 + MRR + 多样性 | ✅ 已做 |
| 随机基线 | 组合公式精确计算 | ✅ 已做 |
| 友好错误提示 | friendly_error 映射（异常类型） | ✅ 已做 |
| 删掉无效缓存 | 命中率结构性为 0 | ✅ 已做 |
| Docker 多阶段构建 | 3.51 GB → 2.42 GB | ✅ 已做 |
| CPU 版 torch | 下载量 5 GB → 600 MB | ✅ 已做 |
| 模型缓存 volume 挂载 | 重建不重下 | ✅ 已做 |
| checkpoints volume 挂载 | 容器重建不丢记忆 | ✅ 已做 |
| HF_HOME 三级探测 | 本地/容器都能跑 | ✅ 已做 |
| 抽公共检索模块 | rag_core/search.py | ✅ 已做 |
| 阈值标定 | 正负样本距离分布 + PR 曲线 | ✅ 已做（MAX_DISTANCE 1.1→1.0） |
| **Rerank 精排** | **bge-reranker-base 对 Top-20 重排** | ✅ 已做（精确率 0.86→0.90） |
| Chunk 级评估 | 标注 chunk id，不只标文档 | ⏳ 待做 |
| 扩充测试集 | 200 条以上 + 置信区间 | ⏳ 待做 |
| 扩语料 | 500+ chunk，让 Rerank 发挥全部威力 | ⏳ 待做 |
| 端到端评估 | LLM-as-judge 或人工标注 | ⏳ 待做 |

### 面试答题模板

> "我构建了 30 个测试项的 RAG 评估集，故意让问题不含文档关键词。用了四个指标 + 随机基线：召回率、精确率、MRR、多样性。随机基线用组合公式精确算，从 102 个 chunk 里随机抽 3 个，命中期望 source 的概率是 84.1%。检索 pipeline 是两阶段：混合检索召回 Top-20（向量 + BM25 + 加权 RRF），再用 bge-reranker-base 交叉编码器精排 Top-3。Rerank 把精确率从 0.86 提到 0.90，MRR 从 0.94 提到 0.97。降级阈值 MAX_DISTANCE 也是标定出来的——正样本 30 条 + 负样本 25 条，扫 101 个候选点选 P/R 平衡点 1.0，幻觉率从 9.1% 降到 3.3%。所有数字都能用 `python eval/eval_retrieval.py` 和 `python eval/calibrate_distance.py` 复现。诚实的局限：Rerank 收益被语料规模压制（只有 2 文档 102 chunk），下一步扩到 500+ chunk 才能发挥全部威力。"
