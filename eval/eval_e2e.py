"""End-to-end RAG evaluation: retrieval + generation + LLM-as-judge.

This evaluates the retrieval+generation chain, not the MCP agent loop. The
direct chain is used so reruns are deterministic and tool-selection variance
does not contaminate the RAG quality signal.

Usage:
    python eval/eval_e2e.py \
        --test-set eval/test_set_v3_gold.json \
        --negatives eval/negatives.json eval/negatives_hard.json \
        --answerable 80 --negative 20 --seed 42 --concurrency 3 \
        --jsonl eval/output/e2e.jsonl --summary eval/output/e2e_summary.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from openai import AsyncOpenAI

from rag_core.eval_metrics import recall_any, wilson_ci
from rag_core.search import DEFAULT_MODEL, LocalEmbeddings, build_bm25, hybrid_search_with_rerank, load_vectorstore

REFUSAL = "根据现有资料无法回答。"

GENERATE_PROMPT = """你是一个只依据给定资料回答问题的助手。

资料：
{context}

问题：{question}

要求：
1. 只使用资料中的信息回答；资料不足以回答时，只输出“根据现有资料无法回答。”
2. 不要编造，不要使用资料以外的知识。
3. 每个结论后用 [编号] 标注来源。
4. 回答简洁，控制在 300 字以内。
"""

JUDGE_PROMPT = """你是严格的 RAG 评审。给定问题、检索资料、模型回答，判断回答质量。只输出 JSON。

问题：
{question}

检索资料：
{context}

模型回答：
{answer}

请输出：
{{
  "context_sufficient": true/false,
  "faithful": true/false,
  "relevant": true/false,
  "refusal_appropriate": true/false,
  "unsupported_claims": ["..."],
  "reason": "一句话理由"
}}

判定标准：
- context_sufficient：资料是否足以回答问题。
- faithful：回答中的每个事实性陈述是否都能在资料中找到依据；引用不存在的来源算不忠实。
- relevant：回答是否直接解决用户问题；若资料不足而正确拒答，relevant 记为 true。
- refusal_appropriate：拒答是否恰当。资料不足时拒答为 true；资料足够却拒答为 false；资料不足却编造为 false。
- unsupported_claims：没有资料依据的具体陈述。
"""


def load_questions(paths: list[str]) -> list[str]:
    out = []
    for path_str in paths:
        data = json.loads(Path(path_str).read_text(encoding="utf-8"))
        questions = data if isinstance(data, list) else data["questions"]
        out.extend(q if isinstance(q, str) else q["question"] for q in questions)
    return out


def stratified_sample(items: list[dict], n: int, rng: random.Random) -> list[dict]:
    groups = defaultdict(list)
    for item in items:
        source = (item.get("expected_sources") or ["unknown"])[0]
        groups[source.split("/")[0]].append(item)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    keys = sorted(groups)
    while len(selected) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(selected) < n:
                selected.append(groups[k].pop())
    return selected


def is_effective_refusal(row: dict) -> bool:
    """A refusal is effective only if the judge says the context was insufficient.

    A partial answer that starts with "资料不足" but still contains grounded
    content is not an abstention; it is an answer with a caveat.
    """
    if row.get("generation_error") or not (row.get("answer") or "").strip():
        return True
    judge = row.get("judge")
    if not judge:
        return False
    return judge.get("context_sufficient") is False and judge.get("refusal_appropriate") is True


def item_id_for(kind: str, question: str) -> str:
    return hashlib.sha1(f"{kind}:{question}".encode("utf-8")).hexdigest()[:16]


def normalize_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "是"}:
            return True
        if lowered in {"false", "no", "0", "否"}:
            return False
    return None


def extract_json_object(content: str) -> str:
    start = content.find("{")
    if start < 0:
        raise ValueError("no JSON object in judge response")
    depth = 0
    for i in range(start, len(content)):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return content[start : i + 1]
    raise ValueError("unbalanced JSON object in judge response")


def parse_judge(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    data = json.loads(extract_json_object(content))
    out = {}
    for key in ["context_sufficient", "faithful", "relevant", "refusal_appropriate"]:
        value = normalize_bool(data.get(key))
        if value is None:
            raise ValueError(f"invalid judge field {key}: {data.get(key)!r}")
        out[key] = value
    claims = data.get("unsupported_claims") or []
    out["unsupported_claims"] = claims if isinstance(claims, list) else [str(claims)]
    out["reason"] = str(data.get("reason", ""))
    return out


async def call_with_retry(coro_factory, attempts=3):
    last = None
    for i in range(attempts):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - API errors vary by client version
            last = exc
            if i < attempts - 1:
                await asyncio.sleep(2**i)
    raise last


async def generate_answer(client: AsyncOpenAI, model: str, question: str, context: str) -> tuple[str, dict]:
    if not context.strip():
        return REFUSAL, {"prompt_tokens": 0, "completion_tokens": 0}
    prompt = GENERATE_PROMPT.format(context=context, question=question)
    resp = await call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800,
        )
    )
    usage = getattr(resp, "usage", None)
    return resp.choices[0].message.content or "", {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }


async def judge_answer(client: AsyncOpenAI, model: str, question: str, context: str, answer: str) -> tuple[dict, dict]:
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    resp = await call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=600,
        )
    )
    usage = getattr(resp, "usage", None)
    parsed = parse_judge(resp.choices[0].message.content or "")
    return parsed, {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }


def build_context(results) -> tuple[str, list[str], list[str]]:
    lines = []
    ids = []
    sources = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        ids.append(chunk_id)
        sources.append(source)
        lines.append(f"[{i}] {source}#{chunk_id}\n{doc.page_content}")
    return "\n\n".join(lines), ids, sources


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def rate(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"value": None, "hits": 0, "n": 0, "ci_low": 0.0, "ci_high": 0.0}
    hits = sum(1 for v in values if v)
    lo, hi = wilson_ci(hits, len(values))
    return {"value": hits / len(values), "hits": hits, "n": len(values), "ci_low": lo, "ci_high": hi}


def p95(values):
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, int(0.95 * (len(values) - 1)))
    return values[idx]


async def main_async(args) -> int:
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("缺少 DEEPSEEK_API_KEY", file=sys.stderr)
        return 1

    test_set = json.loads(Path(args.test_set).read_text(encoding="utf-8"))
    negatives = load_questions(args.negatives)
    rng = random.Random(args.seed)
    answerable = stratified_sample(test_set, args.answerable, rng)
    rng.shuffle(negatives)
    negatives = negatives[: args.negative]

    answer_client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )
    judge_client = AsyncOpenAI(
        api_key=os.getenv("JUDGE_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("JUDGE_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL"),
    )
    judge_model = os.getenv("JUDGE_MODEL", "deepseek-chat")
    answer_model = os.getenv("ANSWER_MODEL", "deepseek-chat")

    print("加载 embedding 和 FAISS 索引...")
    embedding = LocalEmbeddings(DEFAULT_MODEL)
    vectorstore = load_vectorstore(embedding=embedding)
    bm25, all_docs = build_bm25(vectorstore)
    print(f"索引：{len(all_docs)} chunk")
    print(f"抽样：answerable={len(answerable)}, negative={len(negatives)}")

    items = []
    for i, item in enumerate(answerable):
        items.append({"kind": "answerable", "index": i, **item})
    for i, question in enumerate(negatives):
        items.append({"kind": "negative", "index": i, "question": question})
    for item in items:
        item["item_id"] = item_id_for(item["kind"], item["question"])

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    jsonl_path = Path(args.jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                row["abstained"] = is_effective_refusal(row)
                done[row["item_id"]] = row
            except Exception:
                continue
    wanted = {item["item_id"] for item in items}
    completed_rows = [done[i] for i in wanted if i in done]
    pending = [item for item in items if item["item_id"] not in done]
    print(f"已完成 {len(completed_rows)} 条，续跑剩余 {len(pending)} 条")

    # Retrieval is CPU-bound; do it sequentially, then run LLM calls concurrently.
    for item in pending:
        t0 = time.time()
        results = await asyncio.to_thread(
            hybrid_search_with_rerank,
            item["question"],
            vectorstore=vectorstore,
            bm25=bm25,
            all_docs=all_docs,
            final_k=3,
        )
        item["retrieval_ms"] = (time.time() - t0) * 1000
        context, ids, sources = build_context(results)
        item["context"] = context
        item["retrieved_ids"] = ids
        item["retrieved_sources"] = sources
        item["retrieval_hit"] = None
        if item["kind"] == "answerable":
            gold_chunks = item.get("gold_chunk_ids") or []
            gold_sources = item.get("gold_sources") or item.get("expected_sources") or []
            item["retrieval_hit"] = bool(
                recall_any(ids, gold_chunks) if gold_chunks else recall_any(sources, gold_sources)
            )

    async def process(item):
        item_id = item["item_id"]
        async with sem:
            t_gen = time.time()
            generation_error = None
            try:
                answer, usage = await generate_answer(
                    answer_client, answer_model, item["question"], item["context"]
                )
            except Exception as exc:  # noqa: BLE001
                answer = ""
                usage = {"prompt_tokens": 0, "completion_tokens": 0}
                generation_error = f"{type(exc).__name__}: {exc}"
            generation_ms = (time.time() - t_gen) * 1000
            judge = None
            judge_usage = {"prompt_tokens": 0, "completion_tokens": 0}
            judge_error = None
            judge_ms = 0.0
            if generation_error:
                pass
            elif item["kind"] == "negative" and not item["context"].strip():
                judge = {
                    "context_sufficient": False,
                    "faithful": None,
                    "relevant": None,
                    "refusal_appropriate": True,
                    "unsupported_claims": [],
                    "reason": "retrieval returned empty; fixed refusal",
                    "source": "rule",
                }
            else:
                t_judge = time.time()
                try:
                    judge, judge_usage = await judge_answer(
                        judge_client, judge_model, item["question"], item["context"], answer
                    )
                    judge["source"] = "llm"
                except Exception as exc:  # noqa: BLE001
                    judge_error = f"{type(exc).__name__}: {exc}"
                judge_ms = (time.time() - t_judge) * 1000
            row = {
                "item_id": item_id,
                "kind": item["kind"],
                "index": item["index"],
                "question": item["question"],
                "expected_sources": item.get("expected_sources", []),
                "gold_sources": item.get("gold_sources", []),
                "gold_chunk_ids": item.get("gold_chunk_ids", []),
                "retrieved_ids": item["retrieved_ids"],
                "retrieved_sources": item["retrieved_sources"],
                "retrieval_hit": item["retrieval_hit"],
                "retrieval_ms": item["retrieval_ms"],
                "answer": answer,
                "abstained": False,
                "judge": judge,
                "judge_error": judge_error,
                "generation_error": generation_error,
                "usage": {
                    "answer": usage,
                    "judge": judge_usage,
                },
                "generation_ms": generation_ms,
                "judge_ms": judge_ms,
            }
            row["abstained"] = is_effective_refusal(row)
            async with lock:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return row

    pending_rows = await asyncio.gather(*(process(item) for item in pending))
    rows = completed_rows + pending_rows

    answerable_rows = [r for r in rows if r["kind"] == "answerable"]
    negative_rows = [r for r in rows if r["kind"] == "negative"]
    answered = [r for r in answerable_rows if r["judge"] and not r["abstained"]]

    summary = {
        "config": {
            "test_set": args.test_set,
            "negatives": args.negatives,
            "answerable": len(answerable_rows),
            "negative": len(negative_rows),
            "seed": args.seed,
            "chunks": len(all_docs),
            "answer_model": answer_model,
            "judge_model": judge_model,
        },
        "answerable": {
            "n": len(answerable_rows),
            "answered": len(answered),
            "abstained": sum(1 for r in answerable_rows if r["abstained"]),
            "faithful": rate([r["judge"]["faithful"] for r in answered if r["judge"]]),
            "relevant": rate([r["judge"]["relevant"] for r in answered if r["judge"]]),
            "unsupported_claim_rate": rate(
                [bool(r["judge"]["unsupported_claims"]) for r in answered if r["judge"]]
            ),
        },
        "negative": {
            "n": len(negative_rows),
            "refusal_correct": rate(
                [r["judge"]["refusal_appropriate"] for r in negative_rows if r["judge"]]
            ),
        },
        "judge_errors": sum(1 for r in rows if r["judge_error"]),
        "generation_errors": sum(1 for r in rows if r.get("generation_error")),
        "latency_ms": {
            "retrieval_p50": statistics.median([r["retrieval_ms"] for r in rows]) if rows else None,
            "retrieval_p95": p95([r["retrieval_ms"] for r in rows]),
            "generation_p50": statistics.median([r["generation_ms"] for r in rows]) if rows else None,
            "generation_p95": p95([r["generation_ms"] for r in rows]),
            "judge_p50": statistics.median([r["judge_ms"] for r in rows]) if rows else None,
            "judge_p95": p95([r["judge_ms"] for r in rows]),
        },
        "token_usage": {
            "answer_prompt": sum(r["usage"]["answer"]["prompt_tokens"] for r in rows),
            "answer_completion": sum(r["usage"]["answer"]["completion_tokens"] for r in rows),
            "judge_prompt": sum(r["usage"]["judge"]["prompt_tokens"] for r in rows),
            "judge_completion": sum(r["usage"]["judge"]["completion_tokens"] for r in rows),
        },
    }

    # 2x2: retrieval hit x faithful (answered answerable only)
    matrix = Counter()
    for r in answered:
        if r["judge"] is None:
            continue
        matrix[(bool(r["retrieval_hit"]), bool(r["judge"]["faithful"]))] += 1
    summary["retrieval_faithfulness_matrix"] = {
        f"retrieval_hit={hit},faithful={faithful}": count
        for (hit, faithful), count in sorted(matrix.items())
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("端到端评估")
    print("=" * 70)
    print(f"answerable n={summary['answerable']['n']}, answered={summary['answerable']['answered']}, "
          f"abstained={summary['answerable']['abstained']}")
    for key in ["faithful", "relevant"]:
        m = summary["answerable"][key]
        print(f"{key}: {m['value']:.1%} ({m['hits']}/{m['n']}) "
              f"95% CI {m['ci_low']:.1%}~{m['ci_high']:.1%}" if m["value"] is not None else f"{key}: n/a")
    m = summary["negative"]["refusal_correct"]
    print(f"negative refusal accuracy: {m['value']:.1%} ({m['hits']}/{m['n']}) "
          f"95% CI {m['ci_low']:.1%}~{m['ci_high']:.1%}" if m["value"] is not None else "negative refusal: n/a")
    print(f"generation errors: {summary['generation_errors']}; judge errors: {summary['judge_errors']}")
    print("2x2 retrieval_hit x faithful:")
    for k, v in summary["retrieval_faithfulness_matrix"].items():
        print(f"  {k}: {v}")
    print(f"latency: retrieval p50={summary['latency_ms']['retrieval_p50']:.0f}ms "
          f"p95={summary['latency_ms']['retrieval_p95']:.0f}ms; "
          f"generation p50={summary['latency_ms']['generation_p50']:.0f}ms "
          f"p95={summary['latency_ms']['generation_p95']:.0f}ms; "
          f"judge p50={summary['latency_ms']['judge_p50']:.0f}ms "
          f"p95={summary['latency_ms']['judge_p95']:.0f}ms")
    print(f"summary: {summary_path}")
    print(f"jsonl: {jsonl_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="eval/test_set_v3_gold.json")
    parser.add_argument("--negatives", nargs="+", default=["eval/negatives.json", "eval/negatives_hard.json"])
    parser.add_argument("--answerable", type=int, default=80)
    parser.add_argument("--negative", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--jsonl", default="eval/output/e2e.jsonl")
    parser.add_argument("--summary", default="eval/output/e2e_summary.json")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
