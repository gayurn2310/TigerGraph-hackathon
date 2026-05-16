import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

COST_PER_INPUT_TOKEN = float(os.getenv("COST_PER_INPUT_TOKEN", "0.000000075"))
COST_PER_OUTPUT_TOKEN = float(os.getenv("COST_PER_OUTPUT_TOKEN", "0.0000003"))

CHUNKS_FILE = "data/chunks/chunks.json"
CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION = "alzheimers_chunks"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_TOP_K = 3