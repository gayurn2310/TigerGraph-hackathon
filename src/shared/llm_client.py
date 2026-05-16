import time
from google import genai
from google.genai import types

from src.shared.config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    COST_PER_INPUT_TOKEN,
    COST_PER_OUTPUT_TOKEN,
)


def call_llm(prompt: str, system_prompt: str = "") -> dict:
    start = time.time()

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=0.1,
            max_output_tokens=256,
        ),
    )

    latency = time.time() - start
    usage = response.usage_metadata

    prompt_tokens = usage.prompt_token_count or 0
    completion_tokens = usage.candidates_token_count or 0
    total_tokens = usage.total_token_count or prompt_tokens + completion_tokens

    cost_usd = (
        prompt_tokens * COST_PER_INPUT_TOKEN
        + completion_tokens * COST_PER_OUTPUT_TOKEN
    )

    return {
        "answer": response.text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": round(latency, 3),
        "cost_usd": round(cost_usd, 8),
    }