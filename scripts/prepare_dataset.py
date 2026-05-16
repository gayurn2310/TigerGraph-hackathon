"""
prepare_dataset.py — Chunk raw .txt files into chunks.json for Basic RAG + GraphRAG
Place this file at: graphrag-hackathon/scripts/prepare_dataset.py

Run AFTER fetch_pubmed.py:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --chunk-size 512 --overlap 64

What this script does:
    1. Reads every .txt file from data/raw/
    2. Cleans and normalises the text
    3. Splits into overlapping chunks
    4. Saves data/chunks/chunks.json
    5. Prints statistics (chunk count, token estimate)

chunks.json format:
[
  {
    "chunk_id":   "PMC1234567_0",
    "pmc_id":     "PMC1234567",
    "text":       "...",
    "char_count": 2048,
    "est_tokens": 512,
    "chunk_index": 0,
    "total_chunks": 4
  },
  ...
]
"""

import json
import re
import argparse
import logging
import html
import unicodedata
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
CHUNKS_DIR   = PROJECT_ROOT / "data" / "chunks"
CHUNKS_FILE  = CHUNKS_DIR / "chunks.json"

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise raw PubMed text:
    - Collapse multiple spaces / newlines
    - Decode XML/HTML entities
    - Preserve meaningful Unicode such as APOE ε4
    - Keep sentence structure intact (important for BERTScore later)
    """
    text = html.unescape(text)
    # Remove control characters while preserving printable Unicode.
    text = "".join(
        " " if unicodedata.category(ch).startswith("C") and ch not in "\n\t" else ch
        for ch in text
    )
    # Collapse runs of whitespace (but preserve paragraph breaks)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text  = "\n".join(lines)
    return text.strip()


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks by word count.

    Why word-based (not character-based)?
    - More consistent token estimates
    - Avoids splitting mid-word

    Args:
        text:       Cleaned plain text
        chunk_size: Target words per chunk (default 400 ≈ 512 tokens)
        overlap:    Words shared between adjacent chunks (default 50)
    """
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start  = 0
    step   = chunk_size - overlap  # how far to advance each iteration

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += step

    return chunks


# ── Main ───────────────────────────────────────────────────────────────────────

def main(chunk_size: int, overlap: int, min_chunk_words: int):
    txt_files = sorted(RAW_DIR.glob("PMC*.txt"))

    if not txt_files:
        log.error(f"No .txt files found in {RAW_DIR}. Run fetch_pubmed.py first.")
        return

    log.info("=" * 60)
    log.info("Dataset Preparation — Chunking raw PubMed text")
    log.info(f"  Source files  : {len(txt_files)}")
    log.info(f"  Chunk size    : {chunk_size} words (~{chunk_size * 1.3:.0f} tokens)")
    log.info(f"  Overlap       : {overlap} words")
    log.info("=" * 60)

    all_chunks     = []
    total_tokens   = 0
    skipped_files  = 0

    for txt_path in txt_files:
        pmc_id = txt_path.stem  # e.g. "PMC1234567"

        raw = txt_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw)

        if len(cleaned.split()) < min_chunk_words:
            log.debug(f"  Skipping {pmc_id} — too short ({len(cleaned.split())} words)")
            skipped_files += 1
            continue

        chunks = chunk_text(cleaned, chunk_size, overlap)
        total_chunks = len(chunks)

        for idx, chunk_text_str in enumerate(chunks):
            word_count  = len(chunk_text_str.split())
            est_tokens  = int(word_count * 1.3)  # word → token ratio for medical English
            total_tokens += est_tokens

            all_chunks.append({
                "chunk_id":     f"{pmc_id}_{idx}",
                "pmc_id":       pmc_id,
                "text":         chunk_text_str,
                "char_count":   len(chunk_text_str),
                "word_count":   word_count,
                "est_tokens":   est_tokens,
                "chunk_index":  idx,
                "total_chunks": total_chunks,
            })

    # ── Save chunks.json ──
    output = {
        "generated_at":     datetime.utcnow().isoformat(),
        "source_files":     len(txt_files),
        "skipped_files":    skipped_files,
        "total_chunks":     len(all_chunks),
        "total_est_tokens": total_tokens,
        "chunk_size_words": chunk_size,
        "overlap_words":    overlap,
        "chunks":           all_chunks,
    }

    CHUNKS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # ── Summary ──
    log.info("\n" + "=" * 60)
    log.info("DONE")
    log.info(f"  Source files    : {len(txt_files)}")
    log.info(f"  Skipped files   : {skipped_files}")
    log.info(f"  Total chunks    : {len(all_chunks):,}")
    log.info(f"  Total ~tokens   : {total_tokens:,}")
    log.info(f"  Saved to        : {CHUNKS_FILE}")

    # Sanity checks
    if len(all_chunks) == 0:
        log.error("  ✗  No chunks produced. Check your raw data directory.")
    elif total_tokens < 2_000_000:
        log.warning(
            f"  ⚠  Only ~{total_tokens:,} tokens. "
            f"Need 2M minimum. Fetch more papers with fetch_pubmed.py."
        )
    else:
        log.info(f"  ✓  Round 1 token threshold met")

    log.info("\nNext step:")
    log.info("  python src/basic_rag/ingest.py   ← embed chunks into ChromaDB")
    log.info("=" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk PubMed .txt files into chunks.json")
    parser.add_argument(
        "--chunk-size", type=int, default=400,
        help="Words per chunk (default: 400 ≈ 512 tokens). "
             "Smaller = more precise retrieval. Larger = more context per chunk."
    )
    parser.add_argument(
        "--overlap", type=int, default=50,
        help="Overlapping words between adjacent chunks (default: 50). "
             "Prevents cutting sentences at boundaries."
    )
    parser.add_argument(
        "--min-chunk-words", type=int, default=100,
        help="Skip files with fewer than this many words (default: 100)"
    )
    args = parser.parse_args()
    main(args.chunk_size, args.overlap, args.min_chunk_words)
