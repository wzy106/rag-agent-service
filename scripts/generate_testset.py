"""
自动生成 RAG 测试集。

策略：
1. 遍历所有 chunk，按 source 分组
2. 每篇文档挑 2 个最长 chunk（避免选到目录、标题之类的空壳）
3. 调 DeepSeek API 生成 3 个问题（覆盖这 2 个 chunk 的内容）
4. 标注 expected_sources = 该文档的 source

输出：eval/test_set_v2.json

用法：
    cd D:/Projects/Agent
    python scripts/generate_testset.py

成本估算：约 ¥0.2（96 篇 × 1 次 API 调用）
时间估算：串行约 5 分钟；可调整 CONCURRENCY 加速
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from collections import defaultdict

# ===== 路径 =====
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from openai import AsyncOpenAI
from rag_core.search import load_vectorstore, LocalEmbeddings, DEFAULT_MODEL


# ===== 配置 =====
CONCURRENCY = 5          # 并发数（太大会触发限流）
CHUNKS_PER_DOC = 2       # 每篇文档挑几个 chunk
QUESTIONS_PER_DOC = 3    # 每篇文档生成几个问题
OUTPUT_FILE = _PROJECT_ROOT / "eval" / "test_set_v2.json"

# DeepSeek client（从 .env 读）
client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

PROMPT_TEMPLATE = """你是一个 RAG 测试集生成助手。

下面是一篇技术文档的两个片段。请基于这些内容生成 3 个不同的中文问题，要求：

1. **问题不能直接复制片段中的原文措辞**，要用自然的提问方式
2. **问题要能被这些片段回答**（不要问片段里没有的信息）
3. **3 个问题覆盖不同角度**（不要是同一问题的不同问法）
4. 每个问题 8~30 个字
5. 用口语化的表达，像真实用户会问的

片段 1：
{chunk1}

片段 2：
{chunk2}

请只输出一个 JSON 数组，格式如下（不要任何其他文字）：
["问题1", "问题2", "问题3"]
"""


async def generate_questions(source: str, chunks: list, sem: asyncio.Semaphore) -> dict | None:
    """为单篇文档生成问题。"""
    async with sem:
        # 挑两个最长 chunk
        sorted_chunks = sorted(chunks, key=lambda c: len(c.page_content), reverse=True)
        picked = sorted_chunks[:CHUNKS_PER_DOC]

        if len(picked) < 2:
            print(f"  [跳过] {source}（chunk 太少）")
            return None

        prompt = PROMPT_TEMPLATE.format(
            chunk1=picked[0].page_content[:800],
            chunk2=picked[1].page_content[:800],
        )

        try:
            resp = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )
            content = resp.choices[0].message.content.strip()

            # 容错：去掉可能的 ```json ``` 包裹
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            questions = json.loads(content)

            if not isinstance(questions, list):
                print(f"  [失败] {source}（返回不是数组）")
                return None

            return {
                "source": source,
                "questions": questions[:QUESTIONS_PER_DOC],
            }

        except Exception as e:
            print(f"  [失败] {source}: {type(e).__name__}: {e}")
            return None


async def main():
    print("加载索引...")
    embedding = LocalEmbeddings(DEFAULT_MODEL)
    vectorstore = load_vectorstore(embedding=embedding)
    all_docs = list(vectorstore.docstore._dict.values())

    # 按 source 分组
    by_source = defaultdict(list)
    for doc in all_docs:
        by_source[doc.metadata["source"]].append(doc)

    print(f"共 {len(all_docs)} 个 chunk，{len(by_source)} 篇文档\n")

    # 并发生成
    print(f"开始生成问题（并发 {CONCURRENCY}）...")
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [generate_questions(src, chunks, sem) for src, chunks in by_source.items()]
    results = await asyncio.gather(*tasks)

    # 组装测试集
    test_set = []
    for r in results:
        if r is None:
            continue
        for q in r["questions"]:
            test_set.append({
                "question": q,
                "expected_sources": [r["source"]],
            })

    # 保存
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(test_set, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[完成] 生成 {len(test_set)} 条测试项")
    print(f"[保存] {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())