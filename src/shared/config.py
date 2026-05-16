import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

COST_PER_INPUT_TOKEN = float(os.getenv("COST_PER_INPUT_TOKEN", "0.000000075"))
COST_PER_OUTPUT_TOKEN = float(os.getenv("COST_PER_OUTPUT_TOKEN", "0.0000003"))