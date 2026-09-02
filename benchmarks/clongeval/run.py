"""
CLongEval Benchmark Runner
==========================

Chinese long-conversation memory evaluation benchmark.
Based on CLongEval (OpenLMLab) LCvMem sub-task.

Flow:
    1. Load CLongEval JSONL dataset
    2. Group entries by conversation
    3. For each conversation:
        a. Parse daily segments, ingest via Mem0
        b. For each question:
            - Search Mem0 -> retrieved memories
            - Generate answer (answerer model)
            - Judge answer vs ground truth (judge model)
    4. Compute metrics
    5. Write unified result JSON

Usage:
    python -m benchmarks.clongeval.run --project-name test --conversations 0
    python -m benchmarks.clongeval.run --project-name full --answerer-model qwen-plus --provider qwen
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from benchmarks.common.llm_client import LLMClient
from benchmarks.common.mem0_client import Mem0Client, format_search_results
from benchmarks.common.utils import (
    Checkpoint,
    GracefulShutdown,
    IngestionCheckpoint,
    cutoff_label,
    save_result_json,
    setup_logging,
)

from .prompts import (
    CATEGORIES_TO_EVALUATE,
    CATEGORY_NAMES,
    JUDGE_SYSTEM_PROMPT,
    get_answer_generation_prompt,
    get_judge_prompt,
)

load_dotenv(override=True)

# ===============================================================================
# CONSTANTS
# ===============================================================================

DEFAULT_DATASET_DIR = "datasets/clongeval"
DEFAULT_DATASET_FILE = "small.jsonl"
CHUNK_SIZE = 10  # lines per ingestion chunk


# ===============================================================================
# DATASET LOADING
# ===============================================================================


def load_dataset(data_path: str) -> list[dict]:
    """Load CLongEval JSONL dataset."""
    entries = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


# ===============================================================================
# CONVERSATION PARSING
# ===============================================================================


def parse_date_cn(date_str: str) -> datetime | None:
    """Parse Chinese date: '2023年04月27日'."""
    for fmt in ("%Y年%m月%d日", "%Y年%m月%d日"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def cn_date_to_epoch(date_str: str) -> int | None:
    parsed = parse_date_cn(date_str)
    if parsed:
        return int(parsed.timestamp())
    return None


def extract_daily_segments(context: str) -> list[tuple[str, str]]:
    """Extract daily conversation segments from CLongEval context.

    Returns [(date_str, conversation_text), ...]
    """
    clean_ctx = context
    if "请记住以上全部对话记录" in clean_ctx:
        clean_ctx = clean_ctx[: clean_ctx.index("请记住以上全部对话记录")]

    date_pattern = r"以下是(\d{4}年\d{2}月\d{2}日)的对话记录："
    segments = re.split(date_pattern, clean_ctx)

    results = []
    for i in range(1, len(segments), 2):
        date_str = segments[i]
        content = segments[i + 1] if i + 1 < len(segments) else ""
        if content.strip():
            results.append((date_str, content.strip()))

    return results


def group_by_conversation(entries: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group entries by conversation using context prefix hash."""
    groups = defaultdict(list)
    for i, entry in enumerate(entries):
        ctx = entry.get("context", "")
        conv_text = ctx.split("请记住以上全部对话记录")[0] if "请记住以上全部对话记录" in ctx else ctx
        conv_hash = hashlib.md5(conv_text[:500].encode()).hexdigest()[:8]
        entry["_original_idx"] = i
        groups[conv_hash].append(entry)

    result = []
    for h, items in groups.items():
        items.sort(key=lambda x: x["_original_idx"])
        result.append((h, items))
    return result


def clean_dialog_text(text: str) -> str:
    """Clean dialog text, remove quote wrapping."""
    text = text.strip()
    if text.startswith("\u201c") and text.endswith("\u201d"):
        text = text[1:-1]
    elif text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text.strip()


def segment_to_chunks(text: str, date_str: str) -> list[list[dict]]:
    """Convert a daily segment into message chunks for ingestion."""
    clean_text = clean_dialog_text(text)
    date_header = f"[对话日期: {date_str}]\n"
    lines = clean_text.split("\n")

    chunks = []
    for i in range(0, len(lines), CHUNK_SIZE):
        batch = "\n".join(lines[i : i + CHUNK_SIZE])
        if batch.strip():
            chunks.append([{"role": "user", "content": date_header + batch}])
    return chunks


def categorize_question(question: str, gold_answer: str) -> int:
    """Best-effort category assignment for CLongEval questions."""
    if any(kw in question for kw in ["几", "多少", "多少天", "多久"]):
        return 3  # temporal/numerical
    if any(kw in question for kw in ["和", "与", "分别", "都", "哪些"]):
        return 2  # multi-hop
    if any(kw in question for kw in ["谁", "什么", "哪", "哪里"]):
        return 1  # single-hop
    return 4  # conversation-understanding


# ===============================================================================
# INGESTION
# ===============================================================================


async def ingest_conversation(
    conv_idx: int,
    conv_hash: str,
    context: str,
    mem0: Mem0Client,
    logger: Any,
    run_id: str,
    project_name: str,
    output_dir: str,
    shutdown: GracefulShutdown,
) -> tuple[bool, str]:
    """Ingest all daily segments of a CLongEval conversation into Mem0."""
    user_id = f"clong_{conv_idx}_{run_id}"
    daily_segments = extract_daily_segments(context)

    logger.info(
        "Ingesting conversation %d: %d daily segments, user_id=%s",
        conv_idx, len(daily_segments), user_id,
    )

    pbar = tqdm(total=len(daily_segments), desc=f"Ingest conv {conv_idx}", leave=True)

    for seg_idx, (date_str, seg_text) in enumerate(daily_segments):
        if shutdown.requested:
            pbar.close()
            return True, user_id

        seg_epoch = cn_date_to_epoch(date_str)
        chunks = segment_to_chunks(seg_text, date_str)

        for chunk in chunks:
            try:
                await mem0.add(chunk, user_id, timestamp=seg_epoch)
            except Exception as e:
                logger.warning("Ingest failed conv %d seg %d: %s", conv_idx, seg_idx, e)

        pbar.update(1)

    pbar.close()
    return True, user_id


# ===============================================================================
# SEARCH + ANSWER + JUDGE
# ===============================================================================


async def process_question(
    entry: dict,
    qa_idx: int,
    conv_idx: int,
    user_id: str,
    mem0: Mem0Client,
    answerer: LLMClient,
    judge_llm: LLMClient,
    cutoffs: list[int],
    top_k: int,
    reference_date: str,
    logger: Any,
) -> dict[str, Any]:
    """Process a single CLongEval question: search + answer + judge."""
    question = entry["query"]
    gold_answer = str(entry["answer"])
    category = categorize_question(question, gold_answer)

    # --- Search ---
    start = time.monotonic()
    search_results = await mem0.search(question, user_id, top_k=top_k)
    search_latency = (time.monotonic() - start) * 1000

    formatted, _ = format_search_results(search_results)

    result: dict[str, Any] = {
        "question_id": f"conv{conv_idx}_q{qa_idx}",
        "conversation_idx": conv_idx,
        "category": category,
        "category_name": CATEGORY_NAMES.get(category, "unknown"),
        "question": question,
        "ground_truth_answer": gold_answer,
        "user_id": user_id,
        "reference_date": reference_date,
        "retrieval": {
            "search_query": question,
            "search_results": formatted,
            "search_latency_ms": round(search_latency, 1),
            "total_results": len(formatted),
        },
    }

    # --- Answer + Judge at each cutoff ---
    cutoff_results: dict[str, dict] = {}

    for c in cutoffs:
        sliced = formatted[:c]
        label = cutoff_label(c)

        # Generate answer
        gen_prompt = get_answer_generation_prompt(question, sliced, reference_date=reference_date)
        generated_answer = await answerer.generate(system="", user=gen_prompt)

        # Judge
        judge_prompt = get_judge_prompt(question, gold_answer, generated_answer)
        raw = await judge_llm.generate_structured(
            system=JUDGE_SYSTEM_PROMPT,
            user=judge_prompt,
        )

        if isinstance(raw, dict):
            label_val = raw.get("label", "").upper()
            correct = label_val == "CORRECT"
        else:
            correct = False

        cutoff_results[label] = {
            "judgment": "CORRECT" if correct else "WRONG",
            "score": 1.0 if correct else 0.0,
            "generated_answer": generated_answer,
            "memories_evaluated": len(sliced),
            "reason": raw.get("reasoning", "") if isinstance(raw, dict) else "",
        }

    result["cutoff_results"] = cutoff_results
    return result


# ===============================================================================
# METRICS
# ===============================================================================


def compute_metrics(evaluations: list[dict], cutoffs: list[int]) -> dict:
    """Compute per-category and overall metrics at each cutoff."""
    metrics_by_cutoff = {}
    for c in cutoffs:
        label = cutoff_label(c)
        total = len(evaluations)
        scores = [e.get("cutoff_results", {}).get(label, {}).get("score", 0.0) for e in evaluations]
        correct = sum(1 for s in scores if s >= 0.5)

        by_category: dict[str, list] = defaultdict(list)
        for e in evaluations:
            cat_name = e.get("category_name", "unknown")
            by_category[cat_name].append(e.get("cutoff_results", {}).get(label, {}).get("score", 0.0))

        cat_metrics = {}
        for cat_name in sorted(by_category):
            cat_scores = by_category[cat_name]
            cat_correct = sum(1 for s in cat_scores if s >= 0.5)
            cat_metrics[cat_name] = {
                "total": len(cat_scores),
                "correct": cat_correct,
                "accuracy": cat_correct / len(cat_scores) * 100 if cat_scores else 0.0,
            }

        metrics_by_cutoff[label] = {
            "overall": {
                "total": total,
                "correct": correct,
                "accuracy": correct / total * 100 if total else 0.0,
            },
            "by_category": cat_metrics,
        }
    return metrics_by_cutoff


def display_results(metrics_by_cutoff: dict, cutoffs: list[int]) -> None:
    """Print metrics to console."""
    for c in cutoffs:
        label = cutoff_label(c)
        m = metrics_by_cutoff.get(label, {})
        overall = m.get("overall", {})
        print(f"\n--- {label} ---")
        print(f"  Overall: {overall.get('correct', 0)}/{overall.get('total', 0)} "
              f"({overall.get('accuracy', 0):.1f}%)")
        for cat_name, cm in sorted(m.get("by_category", {}).items()):
            print(f"  {cat_name}: {cm['correct']}/{cm['total']} ({cm['accuracy']:.1f}%)")


# ===============================================================================
# CLI
# ===============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CLongEval benchmark: ingest + search + answer + judge",
    )
    parser.add_argument("--project-name", required=True, help="Name for this eval run")
    parser.add_argument("--answerer-model", default="qwen-plus", help="Model for answer generation")
    parser.add_argument("--judge-model", default="qwen-plus", help="Model for judging")
    parser.add_argument("--provider", default="qwen", help="LLM provider (openai, qwen, zhipu, moonshot, anthropic, azure)")
    parser.add_argument("--judge-provider", default=None, help="Judge provider (defaults to --provider)")
    parser.add_argument("--conversations", default="0", help="Comma-separated conversation indices")
    parser.add_argument("--top-k", type=int, default=200, help="Number of search results to retrieve")
    parser.add_argument("--top-k-cutoffs", default="10,20,50,200", help="Comma-separated cutoffs")
    parser.add_argument("--max-workers", type=int, default=10, help="Max parallel workers")
    parser.add_argument("--output-dir", default="results/clongeval", help="Output directory")
    parser.add_argument("--dataset-path", default=None, help="Path to CLongEval JSONL file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    parser.add_argument("--max-questions", type=int, default=None, help="Max questions per conversation")
    parser.add_argument("--rpm", type=int, default=200, help="Requests per minute for LLM")
    parser.add_argument("--backend", default="oss", choices=["oss", "cloud"],
                        help="Mem0 backend: 'oss' or 'cloud'")
    parser.add_argument("--mem0-host", default=None, help="Mem0 server URL")
    parser.add_argument("--mem0-api-key", default=None, help="Mem0 API key (cloud only)")
    return parser.parse_args()


# ===============================================================================
# MAIN
# ===============================================================================


async def async_main() -> None:
    args = parse_args()
    logger = setup_logging("clongeval", debug=args.debug)

    cutoffs = [int(c) for c in args.top_k_cutoffs.split(",")]
    conv_indices = [int(c) for c in args.conversations.split(",")]

    run_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join(args.output_dir, f"predicted_{args.project_name}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"CLongEval Benchmark | project={args.project_name} run_id={run_id}")
    print(f"  Answerer: {args.answerer_model} ({args.provider})")
    print(f"  Judge: {args.judge_model} ({args.judge_provider or args.provider})")
    print(f"  Conversations: {args.conversations}")

    # Load dataset
    dataset_path = args.dataset_path or os.path.join(DEFAULT_DATASET_DIR, DEFAULT_DATASET_FILE)
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Please download CLongEval data from https://github.com/OpenLMLab/CLongEval")
        print(f"and place {DEFAULT_DATASET_FILE} in {DEFAULT_DATASET_DIR}/")
        return

    entries = load_dataset(dataset_path)
    conversations = group_by_conversation(entries)
    print(f"  Dataset: {len(entries)} entries, {len(conversations)} conversations")

    # Select conversations
    selected = [(h, items) for i, (h, items) in enumerate(conversations) if i in conv_indices]
    print(f"  Selected: {len(selected)} conversations\n")

    # Init LLM clients
    answerer = LLMClient(model=args.answerer_model, provider=args.provider, rpm=args.rpm)
    judge_provider = args.judge_provider or args.provider
    judge_llm = LLMClient(model=args.judge_model, provider=judge_provider, rpm=args.rpm)

    # Init Mem0
    mem0 = Mem0Client(mode=args.backend, host=args.mem0_host, api_key=args.mem0_api_key, rpm=args.rpm)
    shutdown = GracefulShutdown()

    all_evaluations: list[dict] = []

    for conv_idx, (conv_hash, conv_entries) in enumerate(selected):
        context = conv_entries[0]["context"]
        daily_segments = extract_daily_segments(context)
        ref_date = daily_segments[-1][0] if daily_segments else "2023年05月"

        # Ingest
        success, user_id = await ingest_conversation(
            conv_indices[conv_idx] if conv_idx < len(conv_indices) else conv_idx,
            conv_hash, context, mem0, logger, run_id, args.project_name, output_dir, shutdown,
        )

        # Answer questions
        qa_list = conv_entries
        if args.max_questions:
            qa_list = qa_list[: args.max_questions]

        print(f"\n[Conv {conv_indices[conv_idx]}] {len(qa_list)} questions")

        sem = asyncio.Semaphore(args.max_workers)

        async def process_one(qa_idx: int, entry: dict) -> dict:
            async with sem:
                return await process_question(
                    entry, qa_idx, conv_indices[conv_idx], user_id,
                    mem0, answerer, judge_llm, cutoffs, args.top_k, ref_date, logger,
                )

        results = await asyncio.gather(*[process_one(i, qa) for i, qa in enumerate(qa_list)])

        conv_correct = sum(
            1 for r in results
            if r.get("cutoff_results", {}).get(cutoff_label(cutoffs[-1]), {}).get("score", 0) >= 0.5
        )
        print(f"  Result: {conv_correct}/{len(results)}")

        all_evaluations.extend(results)

        # Save per-question results
        for r in results:
            path = os.path.join(output_dir, f"{r['question_id']}.json")
            save_result_json(path, r)

    # Compute and display metrics
    metrics = compute_metrics(all_evaluations, cutoffs)
    display_results(metrics, cutoffs)

    # Save unified result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unified_path = os.path.join(args.output_dir, f"clongeval_results_{timestamp}.json")
    save_result_json(unified_path, {
        "metadata": {
            "benchmark": "clongeval",
            "project_name": args.project_name,
            "run_id": run_id,
            "timestamp": timestamp,
            "answerer_model": args.answerer_model,
            "judge_model": args.judge_model,
            "provider": args.provider,
            "top_k": args.top_k,
            "top_k_cutoffs": [cutoff_label(c) for c in cutoffs],
            "total_questions": len(all_evaluations),
        },
        "metrics_by_cutoff": metrics,
        "evaluations": all_evaluations,
    })

    print(f"\nResults saved to: {unified_path}")
    print(f"Total questions evaluated: {len(all_evaluations)}")

    await mem0.close()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
