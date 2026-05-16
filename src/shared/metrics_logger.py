import json
import os
from datetime import datetime

RESULTS_FILE = "results/benchmark_results.json"


def log_result(question: str, result: dict, ground_truth: str | None = None) -> None:
    os.makedirs("results", exist_ok=True)

    rows = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            rows = json.load(f)

    rows.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "pipeline": result.get("pipeline", "unknown"),
        "answer": result.get("answer", ""),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "latency_seconds": result.get("latency_seconds", 0),
        "cost_usd": result.get("cost_usd", 0),
        "ground_truth": ground_truth,
    })

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def get_summary() -> dict:
    if not os.path.exists(RESULTS_FILE):
        return {}

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        rows = json.load(f)

    by_pipeline = {}

    for row in rows:
        name = row["pipeline"]
        by_pipeline.setdefault(name, []).append(row)

    summary = {}

    for name, items in by_pipeline.items():
        count = len(items)
        summary[name] = {
            "query_count": count,
            "avg_total_tokens": round(sum(x["total_tokens"] for x in items) / count, 1),
            "avg_latency_seconds": round(sum(x["latency_seconds"] for x in items) / count, 3),
            "avg_cost_usd": round(sum(x["cost_usd"] for x in items) / count, 8),
        }

    return summary