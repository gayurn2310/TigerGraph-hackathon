from src.shared.llm_client import call_llm

SYSTEM_PROMPT = (
    "You are a careful research assistant. "
    "Answer clearly and concisely. "
    "If you are unsure, say so instead of inventing details."
)


def run_query(question: str) -> dict:
    prompt = (
        "Answer the following question using your general knowledge only.\n\n"
        f"Question: {question}"
    )

    result = call_llm(prompt, system_prompt=SYSTEM_PROMPT)
    result["pipeline"] = "LLM-Only"
    return result


if __name__ == "__main__":
    response = run_query("What is Alzheimer's disease?")
    print(response)