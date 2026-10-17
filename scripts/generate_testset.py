"""Generate a test set with evidence chunk ids.

v3 schema per item:
{
  "question": "...",
  "expected_sources": ["langchain/01_....md"],
  "evidence_chunk_ids": ["<chunk_id>", ...],
  "evidence_coverage_mode": "any" | "all",
  "generator_version": "v3-evidence"
}

The two longest chunks per document are used, same as v2. The model must say
which chunk(s) contain the answer, so chunk-level recall has a clean label.

Usage:
    python scripts/generate_testset.py --output eval/test_set_v3.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from openai import AsyncOpenAI

from rag_core.search import DEFAULT_MODEL, LocalEmbeddings, load_vectorstore

PROMPT_TEMPLATE = """你是一个 RAG 测试集生成助手。

下面是一篇技术文档的两个片段（编号 0 和 1）。请基于这些内容生成 {n} 个不同的中文问题，要求：

1. 问题不能直接复制片段中的原文措辞，要用自然的提问方式
2. 问题要能被这些片段回答（不要问片段里没有的信息）
3. {n} 个问题覆盖不同角度
4. 每个问题 8~30 个字
5. 用口语化的表达，像真实用户会问的
6. 每条问题必须标注 evidence：真正包含答案的片段编号。
   - 只需要一个片段就能回答：evidence 写 [0] 或 [1]
   - 必须同时用到两个片段：evidence 写 [0, 1]

片段 0：
{chunk0}

片段 1：
{chunk1}

只输出一个 JSON 对象，不要任何其他文字：
{{"questions": [{{"question": "问题1", "evidence": [0]}}, {{"question": "问题2", "evidence": [1]}}]}}
"""


def parse_json_object(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    return json.loads(content)


async def generate_for_source(
    source: str,
    chunks: list,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    n_questions: int,
    chunks_per_doc: int,
) -> dict | None:
    async with sem:
        picked = sorted(chunks, key=lambda c: len(c.page_content), reverse=True)[:chunks_per_doc]
        if len(picked) < chunks_per_doc:
            print(f"  [跳过] {source}（chunk 太少）")
            return None

        prompt = PROMPT_TEMPLATE.format(
            n=n_questions,
            chunk0=picked[0].page_content[:800],
            chunk1=picked[1].page_content[:800],
        )
        try:
            resp = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800,
            )
            parsed = parse_json_object(resp.choices[0].message.content or "")
            questions = parsed.get("questions", [])
            if not isinstance(questions, list):
                raise ValueError("questions is not a list")

            out = []
            for item in questions[:n_questions]:
                if not isinstance(item, dict):
                    continue
                question = str(item.get("question", "")).strip()
                evidence = item.get("evidence", [])
                if isinstance(evidence, int):
                    evidence = [evidence]
                evidence = [i for i in evidence if isinstance(i, int) and 0 <= i < chunks_per_doc]
                if not question or not evidence:
                    continue
                out.append(
                    {
                        "question": question,
                        "evidence_chunk_ids": [picked[i].metadata["chunk_id"] for i in sorted(set(evidence))],
                        "evidence_coverage_mode": "all" if len(set(evidence)) > 1 else "any",
                    }
                )
            if not out:
                raise ValueError("no valid questions")
            return {"source": source, "items": out}
        except Exception as exc:
            print(f"  [失败] {source}: {type(exc).__name__}: {exc}")
            return None


def load_fallback(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = defaultdict(list)
    for item in data:
        for source in item.get("expected_sources", []):
            out[source].append(item["question"])
    return out


async def main_async(args) -> int:
    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )

    print("加载索引...")
    embedding = LocalEmbeddings(DEFAULT_MODEL)
    vectorstore = load_vectorstore(embedding=embedding)
    all_docs = list(vectorstore.docstore._dict.values())
    by_source = defaultdict(list)
    for doc in all_docs:
        by_source[doc.metadata["source"]].append(doc)
    print(f"共 {len(all_docs)} 个 chunk，{len(by_source)} 篇文档\n")

    sources = sorted(by_source)
    if args.limit:
        sources = sources[: args.limit]

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        generate_for_source(
            source, by_source[source], client, sem, args.questions_per_doc, args.chunks_per_doc
        )
        for source in sources
    ]
    results = await asyncio.gather(*tasks)

    fallback = load_fallback(Path(args.fallback_v2))
    test_set = []
    failed_sources = []
    for source, result in zip(sources, results):
        if result is None:
            failed_sources.append(source)
            for question in fallback.get(source, []):
                picked = sorted(by_source[source], key=lambda c: len(c.page_content), reverse=True)[
                    : args.chunks_per_doc
                ]
                test_set.append(
                    {
                        "question": question,
                        "expected_sources": [source],
                        "evidence_chunk_ids": [c.metadata["chunk_id"] for c in picked],
                        "evidence_coverage_mode": "all",
                        "generator_version": "v3-fallback-v2",
                    }
                )
            continue
        for item in result["items"]:
            test_set.append(
                {
                    "question": item["question"],
                    "expected_sources": [source],
                    "evidence_chunk_ids": item["evidence_chunk_ids"],
                    "evidence_coverage_mode": item["evidence_coverage_mode"],
                    "generator_version": "v3-evidence",
                }
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(test_set, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 生成 {len(test_set)} 条测试项")
    print(f"[保存] {out_path}")
    print(f"[失败文档] {len(failed_sources)}")
    if failed_sources:
        print("  " + ", ".join(failed_sources[:10]))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval/test_set_v3.json")
    parser.add_argument("--fallback-v2", default="eval/test_set_v2.json")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--chunks-per-doc", type=int, default=2)
    parser.add_argument("--questions-per-doc", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
