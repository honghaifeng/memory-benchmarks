"""Analyze CLongEval result JSONs and produce a summary for the README.

Usage:
    python3 scripts/analyze_clongeval.py results/clongeval/*.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def classify_error(entry: dict, cutoff: str) -> str:
    """Classify an INCORRECT evaluation into an error bucket."""
    q = entry.get("question", "")
    gt = entry.get("ground_truth_answer", "")
    cr = entry.get("cutoff_results", {}).get(cutoff, {})
    answer = (cr.get("generated_answer") or "").strip()
    reason = (cr.get("reason") or "").lower()
    retrieval = entry.get("retrieval", {})
    results = retrieval.get("search_results", [])

    # 1. Retrieval failure: relevant memory not in top-k
    if not results:
        return "检索失败(无记忆返回)"
    gt_terms = [t for t in re.split(r"[\s，。；、：""''《》]+", gt) if len(t) >= 2]
    hit = any(any(t in (m.get("memory") or "") for t in gt_terms) for m in results)
    if not hit:
        return "检索失败(相关记忆未命中)"

    # 2. Fabrication / hallucination
    if "无法确定" in answer or "没有找到" in answer or "未找到" in answer or "不确定" in answer:
        return "诚实拒答(未记录)"
    if answer and gt and gt not in answer and not any(t in answer for t in gt_terms):
        return "幻觉(答案与记忆不符)"

    # 3. Temporal reasoning
    if any(k in q for k in ["日期", "几号", "哪天", "什么时候", "多久", "几月", "去年", "今年", "昨天", "前天", "星期"]):
        return "时间推理错误"

    # 4. Multi-hop
    if entry.get("category_name") == "multi-hop":
        return "多跳推理错误"

    # 5. Partial / other
    if "部分" in reason or "不完整" in reason or "遗漏" in reason:
        return "答案不完整"

    return "其他"


def analyze(path: Path) -> dict:
    d = load_results(path)
    meta = d.get("metadata", {})
    metrics = d.get("metrics_by_cutoff", {})

    summary = {
        "file": path.name,
        "project": meta.get("project_name", path.stem),
        "answerer": meta.get("answerer_model", "?"),
        "judge": meta.get("judge_model", "?"),
        "provider": meta.get("provider", "?"),
        "metrics": metrics,
    }

    # Per-cutoff error analysis
    evals = d.get("evaluations", [])
    for cutoff in ["top_10", "top_20", "top_50", "top_200"]:
        if cutoff not in metrics:
            continue
        incorrect = [e for e in evals if e.get("cutoff_results", {}).get(cutoff, {}).get("judgment") == "INCORRECT"]
        buckets = Counter(classify_error(e, cutoff) for e in incorrect)
        cat_counter = Counter(e.get("category_name", "?") for e in incorrect)
        summary[f"errors_{cutoff}"] = {
            "total": len(incorrect),
            "buckets": dict(buckets),
            "by_category": dict(cat_counter),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    for f in sorted(args.files):
        s = analyze(f)
        print("=" * 70)
        print(f"Project: {s['project']}  |  Answerer: {s['answerer']} ({s['provider']})")
        for cutoff, m in s["metrics"].items():
            overall = m.get("overall", {})
            cats = m.get("by_category", {})
            sh = cats.get("single-hop", {}).get("accuracy")
            mh = cats.get("multi-hop", {}).get("accuracy")
            print(f"  {cutoff}: overall={overall.get('accuracy')}% ({overall.get('correct')}/{overall.get('total')})"
                  f" single-hop={sh}% multi-hop={mh}%")
        for cutoff in ["top_10", "top_20", "top_50", "top_200"]:
            err = s.get(f"errors_{cutoff}")
            if err and err["total"]:
                print(f"  {cutoff} 错误分析 ({err['total']} 条):")
                for bucket, n in err["buckets"].items():
                    print(f"    - {bucket}: {n}")
                print(f"    按类别: {err['by_category']}")


if __name__ == "__main__":
    main()
