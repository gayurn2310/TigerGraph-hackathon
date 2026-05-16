import json
import os
import sys
import time

MAX_QUESTIONS = 1
SLEEP_SECONDS = 20

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm_only.pipeline import run_query as run_llm_only
from src.basic_rag.pipeline import run_query as run_basic_rag
from src.shared.metrics_logger import log_result, get_summary

QUESTIONS_FILE = "data/test_questions.json"

PIPELINES = [
    ("LLM-Only", run_llm_only),
    ("Basic RAG", run_basic_rag),
]


def main() -> None:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    questions = questions[:MAX_QUESTIONS]

    print(
        f"Running benchmark on {len(questions)} question(s) "
        f"across {len(PIPELINES)} pipeline(s)"
    )

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        ground_truth = item.get("correct_answer")

        print(f"\n[{index}/{len(questions)}] {question}")

        for pipeline_name, pipeline_fn in PIPELINES:
            print(f"\n  Pipeline: {pipeline_name}")

            try:
                result = pipeline_fn(question)
                log_result(question, result, ground_truth=ground_truth)

                print(f"  Answer: {result['answer'][:160]}...")
                print(
                    f"  Tokens: {result['total_tokens']} | "
                    f"Latency: {result['latency_seconds']}s | "
                    f"Cost: ${result['cost_usd']:.8f}"
                )

            except Exception as exc:
                print(f"  ERROR: {exc}")

            time.sleep(SLEEP_SECONDS)

    print("\nSummary:")
    print(get_summary())


if __name__ == "__main__":
    main()