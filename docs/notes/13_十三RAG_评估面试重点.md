# 十三、RAG 评估（面试重点）

### 评估方法

构建 **249 条测试项**（覆盖 83 篇文档），测试集用脚本自动生成：
1. 遍历所有 chunk，按 source 分组
2. 每篇文档挑 2 个最长 chunk
3. 调 DeepSeek API 生成 3 个问题
4. 标注 `expected_sources = 该文档的 source`

**关键设计**：
- 问题口语化、角度多样、能被对应文档回答
- 13 篇极短文档（chunk < 2）被跳过
- 平均每篇 3 题

### 测试集自动生成脚本

`scripts/generate_testset.py`：

```python
async def generate_questions(source: str, chunks: list, sem: asyncio.Semaphore):
    async with sem:
        # 挑两个最长 chunk
        sorted_chunks = sorted(chunks, key=lambda c: len(c.page_content), reverse=True)
        picked = sorted_chunks[:CHUNKS_PER_DOC]
        if len(picked) < 2:
            return None

        prompt = PROMPT_TEMPLATE.format(
            chunk1=picked[0].page_content[:800],
            chunk2=picked[1].page_content[:800],
        )
        resp = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        content = resp.choices[0].message.content.strip()
        # 去掉可能的 ```json ``` 包裹
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        questions = json.loads(content)
        return {"source": source, "questions": questions[:QUESTIONS_PER_DOC]}
```

**成本**：约 ¥0.2（98 篇 × 1 次 API 调用）
**时间**：并发 5，约 1~2 分钟

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

### 评估结果（98 篇文档 / 583 chunk / 249 测试项）

| 策略 | 召回率 | 精确率 | MRR | 平均多样性 |
|---|---|---|---|---|
| 普通相似度 | 52.2% | 0.24 | 0.40 | 2.63 |
| MMR | 41.0% | 0.16 | 0.35 | 2.79 |
| MMR + 按来源去重 | 44.2% | 0.15 | 0.36 | **3.00** |
| 混合检索（带降级 0.96） | 54.6% | 0.24 | 0.45 | 2.60 |
| **混合检索 + Rerank** | **60.6%** | **0.27** | **0.49** | 2.60 |
| 随机基线 | **3.1%** | — | — | — |

> **可复现**：`python eval/eval_retrieval.py` 一条命令跑出上面所有数字。评估脚本从 `rag_core.search` import 检索函数，和服务端部署的是同一份代码。

### 关键发现

**1. 随机基线从 84.1% 降到 3.1%——判别空间 57.5 个点**

- 旧语料：2 篇文档 / 102 chunk / 30 测试项 → 随机基线 84.1%，判别空间 15.9 个点
- 新语料：98 篇文档 / 583 chunk / 249 测试项 → 随机基线 3.1%，判别空间 **57.5 个点**
- **旧数字全 100% 是“天花板效应”**——测试集太简单，看不出策略差异

**2. Rerank 是精确率的最后一块拼图**

- 混合检索召回率 54.6% → Rerank 后 **60.6%**（+6.0%）
- MRR 0.45 → **0.49**（+0.04）
- **主要提升召回率**——Rerank 把 Top-20 里被埋没的相关 chunk 顶上来

**3. 混合检索优于纯相似度**

- 54.6% vs 52.2%（**+2.4%**）
- 说明 BM25 补的关键词匹配确实有用

**4. MMR 在“精确匹配”任务上吃亏**

- 召回率只有 41.0%（低于相似度 52.2%）
- 但多样性较高（2.79）
- **多样性导向的代价**——选多角度的结果，但可能漏掉最相关的

**5. 精确率和多样性呈明显 trade-off**

| 策略 | 精确率 | 多样性 |
|---|---|---|
| 混合检索 + Rerank | 0.27 | 2.60 |
| MMR + 去重 | 0.15 | **3.00** |

**6. 选哪个策略取决于业务**

| 场景 | 推荐策略 |
|---|---|
| 追求精确（如事实查询） | **混合检索 + Rerank** |
| 需要多角度（如分析任务） | MMR + 去重 |
| 平衡 | MMR |

### 失败案例分析

**案例 1**：口语化表达
- 用户问：“怎么让回答一个字一个字显示出来？”
- 期望文档：`readme/04_启动.md`
- 实际命中：`notes_interview/q05_流式输出怎么实现的.md`
- 原因：口语化表达和文档术语有语义鸿沟

**案例 2**：跨主题问题
- 用户问：“多 Agent 系统怎么保证消息不丢？”
- 期望文档：`stories/18_故事_17双源记忆打架架构重构.md`
- 实际命中：`notes_interview/q12_Agent_多轮对话怎么实现.md`
- 原因：语义相近但主题不同

### 已知不足

**评估层面**：
- **文档级召回而非 chunk 级**：用 `source`（相对路径）判断命中，同一文档的多个 chunk 算命中。粒度还可以更细
- **source 级评估可能高估效果**：估计高估 8~14 个百分点；后续需要 chunk 级标注或人工复核
- **chunk 级标注成本高**：需要人工标注每个问题的“正确 chunk id”，耗时大
- **没有端到端指标**：检索对了不等于回答对了。应该加 LLM-as-judge 或人工标注的 answer correctness

**架构层面**：
| 位置 | 问题 | 生产级解法 |
|---|---|---|
| `/chat` 和 `/chat/stream` | pymysql 是**同步客户端**，在 async 路径上会阻塞事件循环 | 用 `aiomysql` |
| `save_conversation` | 每次请求新建连接，没有连接池 | 用连接池（如 `DBUtils`） |
| 流式状态机 | 用三个布尔变量表达状态，边界情况可能出错 | 改用 `metadata["langgraph_node"]` + 显式状态机 |
| MCP Server 用了私有 API | `vectorstore.docstore._dict` 是内部 API，版本升级可能坏 | 构建索引时自己维护 chunks 列表并 pickle 落盘 |
| **Reranker 性能** | CPU 精排 20 候选约 2~7 秒，249 条约 10~30 分钟 | 更小模型 / GPU / ONNX 量化 |
| **chunk 级评估** | 现在是文档级 | 标注 chunk id |

### 改进方案

| 方案 | 原理 | 状态 |
|---|---|---|
| 混合检索 | 向量 + BM25，加权合并 | ✅ 已做 |
| 降级策略 | 距离阈值拦截无关问题 | ✅ 已做 |
| 多指标评估 | 召回率 + 精确率 + MRR + 多样性 | ✅ 已做 |
| 随机基线 | 组合公式精确计算 | ✅ 已做 |
| 友好错误提示 | friendly_error 映射 | ✅ 已做 |
| 删掉无效缓存 | 命中率结构性为 0 | ✅ 已做 |
| Docker 多阶段构建 | 3.51 GB → 2.42 GB | ✅ 已做 |
| CPU 版 torch | 下载量 5 GB → 600 MB | ✅ 已做 |
| 模型缓存 volume 挂载 | 重建不重下 | ✅ 已做 |
| checkpoints volume 挂载 | 容器重建不丢记忆 | ✅ 已做 |
| HF_HOME 三级探测 | 本地/容器都能跑 | ✅ 已做 |
| 抽公共检索模块 | rag_core/search.py | ✅ 已做 |
| 阈值标定 | 正负样本距离分布 + PR 曲线 | ✅ 已做（MAX_DISTANCE 1.0→0.96） |
| Rerank 精排 | bge-reranker-base 对 Top-20 重排 | ✅ 已做（召回率 +6.0%） |
| **语料扩充** | 2 篇 → 98 篇 | ✅ 已做 |
| **测试集自动生成** | 30 条 → 249 条 | ✅ 已做 |
| Chunk 级评估 | 标注 chunk id，不只标文档 | ⏳ 待做 |
| 端到端评估 | LLM-as-judge 或人工标注 | ⏳ 待做 |

### 面试答题模板

> “我构建了 249 个测试项的 RAG 评估集，故意让问题不含文档关键词。测试集是通过脚本自动生成的——遍历 98 篇文档的 chunk，每篇挑 2 个最长 chunk，调 DeepSeek API 生成 3 个问题。用了四个指标 + 随机基线：召回率、精确率、MRR、多样性。随机基线用组合公式精确算，从 583 个 chunk 里随机抽 3 个，命中期望 source 的概率是 3.1%。检索 pipeline 是两阶段：混合检索召回 Top-20（向量 + BM25 + 加权 RRF），再用 bge-reranker-base 交叉编码器精排 Top-3。Rerank 把召回率从 54.6% 提到 60.6%，MRR 从 0.45 提到 0.49。降级阈值 MAX_DISTANCE 也是标定出来的——正样本 100 条 + 负样本 25 条，扫 101 个候选点，最终取 0.96，P=0.990、R=1.000、F1=0.995。所有数字都能用 `python eval/eval_retrieval.py` 复现。**这里有个真实的迭代故事**：我第一版语料只有 2 篇文档、测试集 30 条，随机基线 84.1%——天花板效应让 Rerank 提升看起来只有 +0.04 精确率。**扩语料到 98 篇、测试集到 249 条后**，随机基线降到 3.1%，Rerank 的真实增益才显现出来，召回率 +6.0%、MRR +0.04。”
