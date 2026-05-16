import json
import os
import sys
import time

MAX_QUESTIONS = 3

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm_only.pipeline import run_query
from src.shared.metrics_logger import log_result, get_summary

QUESTIONS_FILE = "data/test_questions.json"


def main() -> None:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    questions = questions[:MAX_QUESTIONS]

    print(f"Running LLM-only benchmark on {len(questions)} questions")

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        ground_truth = item.get("correct_answer")

        print(f"\n[{index}/{len(questions)}] {question}")

        try:
            result = run_query(question)
            log_result(question, result, ground_truth=ground_truth)

            print(f"Answer: {result['answer'][:160]}...")
                       print(
                f"Tokens: {result['total_tokens']} | "
                f"Latency: {result['latency_seconds']}s | "
                f"Cost: ${result['cost_usd']:.8f}"
            )

        except Exception as exc:
            print(f"ERROR: {exc}")

        time.sleep(15)

    print("\nSummary:")
    print(get_summary())


if __name__ == "__main__":
    main()