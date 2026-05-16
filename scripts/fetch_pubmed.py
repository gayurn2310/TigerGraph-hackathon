"""
fetch_pubmed.py — Download Alzheimer's full-text papers from PubMed Central
Place this file at: graphrag-hackathon/scripts/fetch_pubmed.py

Usage:
    python scripts/fetch_pubmed.py
    python scripts/fetch_pubmed.py --max-papers 5000 --batch-size 100

Requirements:
    pip install biopython requests tqdm python-dotenv

What this script does:
    1. Searches PubMed Central for open-access Alzheimer's papers
    2. Downloads full-text XML for each paper
    3. Extracts clean plain text (title + abstract + body)
    4. Saves one .txt file per paper into data/raw/
    5. Saves a manifest.json with metadata for every paper
    6. Prints a token estimate at the end

Folder output:
    data/
    ├── raw/
    │   ├── PMC1234567.txt
    │   ├── PMC1234568.txt
    │   └── ...
    └── manifest.json
"""

import os
import re
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Biopython for Entrez API
try:
    from Bio import Entrez, Medline
except ImportError:
    raise ImportError("Run: pip install biopython")

# ── Configuration ──────────────────────────────────────────────────────────────

load_dotenv()  # loads .env from project root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# NCBI requires a valid email — set in .env or hardcode for testing
Entrez.email = os.getenv("NCBI_EMAIL", "your@email.com")
Entrez.api_key = os.getenv("NCBI_API_KEY", "")  # optional but raises rate limit 3→10 req/s

# Paths — relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
MANIFEST     = PROJECT_ROOT / "data" / "manifest.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# Search terms — feel free to expand later for Parkinson's, MS, ALS, etc.
SEARCH_QUERIES = [
    "Alzheimer's disease[Title/Abstract]",
    "amyloid beta Alzheimer[Title/Abstract]",
    "tau protein neurodegeneration Alzheimer[Title/Abstract]",
    "APOE4 Alzheimer risk[Title/Abstract]",
    "Alzheimer clinical trial treatment[Title/Abstract]",
    "dementia Alzheimer pathology[Title/Abstract]",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def search_pmc(query: str, max_results: int) -> list[str]:
    """Return a list of PMC IDs for the given query."""
    log.info(f"Searching PMC: {query!r} (max {max_results})")
    handle = Entrez.esearch(
        db="pmc",
        term=query + " AND open access[filter]",  # open-access only = free full text
        retmax=max_results,
        usehistory="y",
    )
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]
    log.info(f"  → Found {len(ids)} articles")
    return ids


def fetch_full_text_xml(pmc_id: str) -> str | None:
    """Fetch the full-text XML for one PMC article. Returns raw XML string or None."""
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pmc&id={pmc_id}&rettype=full&retmode=xml"
    )
    if Entrez.api_key:
        url += f"&api_key={Entrez.api_key}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        log.warning(f"  Failed to fetch PMC{pmc_id}: {e}")
        return None


def xml_to_plain_text(xml: str, pmc_id: str) -> dict | None:
    """
    Extract plain text from PMC XML.
    Returns dict with keys: pmc_id, title, abstract, body, full_text
    Uses simple regex — no heavy XML parser needed for our purposes.
    """
    def strip_tags(text: str) -> str:
        """Remove all XML/HTML tags and clean whitespace."""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_between(tag: str, xml: str) -> str:
        """Pull content between opening and closing tag (first match)."""
        pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
        match = re.search(pattern, xml, re.DOTALL | re.IGNORECASE)
        return strip_tags(match.group(1)) if match else ""

    title    = extract_between("article-title", xml)
    abstract = extract_between("abstract", xml)

    # Body = everything inside <body> tags
    body_match = re.search(r"<body[^>]*>(.*?)</body>", xml, re.DOTALL | re.IGNORECASE)
    body = strip_tags(body_match.group(1)) if body_match else ""

    # Skip papers with no useful text
    if not abstract and not body:
        return None

    full_text = "\n\n".join(filter(None, [title, abstract, body]))

    return {
        "pmc_id":    pmc_id,
        "title":     title,
        "abstract":  abstract,
        "body":      body[:5000] + "..." if len(body) > 5000 else body,  # manifest preview only
        "full_text": full_text,
        "char_count": len(full_text),
        # rough token estimate: 1 token ≈ 4 chars for English medical text
        "est_tokens": len(full_text) // 4,
    }


def estimate_tokens(text: str) -> int:
    return len(text) // 4


# ── Main ───────────────────────────────────────────────────────────────────────

def main(max_papers: int, batch_size: int, delay: float):
    log.info("=" * 60)
    log.info("PubMed Central — Alzheimer's Dataset Collector")
    log.info("=" * 60)

    # ── Step 1: Collect all PMC IDs across all queries ──
    all_ids: set[str] = set()
    per_query = max_papers // len(SEARCH_QUERIES)

    for query in SEARCH_QUERIES:
        ids = search_pmc(query, per_query)
        all_ids.update(ids)
        time.sleep(0.4)  # be polite to NCBI

    all_ids = list(all_ids)
    log.info(f"\nTotal unique articles found: {len(all_ids)}")
    log.info(f"Target: {max_papers} papers\n")

    # ── Step 2: Skip already-downloaded files ──
    existing = {f.stem for f in RAW_DIR.glob("PMC*.txt")}
    to_fetch  = [i for i in all_ids if f"PMC{i}" not in existing]
    log.info(f"Already downloaded: {len(existing)} | Remaining: {len(to_fetch)}")

    # ── Step 3: Download + extract in batches ──
    manifest   = []
    total_tokens = 0
    failed     = []

    for i, pmc_id in enumerate(tqdm(to_fetch, desc="Downloading papers")):
        xml = fetch_full_text_xml(pmc_id)

        if xml is None:
            failed.append(pmc_id)
            continue

        parsed = xml_to_plain_text(xml, pmc_id)

        if parsed is None:
            log.debug(f"  Skipped PMC{pmc_id} — no usable text")
            continue

        # Save plain text file
        out_path = RAW_DIR / f"PMC{pmc_id}.txt"
        out_path.write_text(parsed["full_text"], encoding="utf-8")

        # Accumulate manifest entry (without full_text to keep manifest small)
        manifest.append({
            "pmc_id":     parsed["pmc_id"],
            "title":      parsed["title"],
            "char_count": parsed["char_count"],
            "est_tokens": parsed["est_tokens"],
            "file":       str(out_path.relative_to(PROJECT_ROOT)),
            "fetched_at": datetime.utcnow().isoformat(),
        })

        total_tokens += parsed["est_tokens"]

        # Rate limiting — NCBI allows 10 req/s with API key, 3 req/s without
        rate_delay = delay if Entrez.api_key else max(delay, 0.35)
        time.sleep(rate_delay)

        # Save manifest every batch_size papers (crash recovery)
        if (i + 1) % batch_size == 0:
            _save_manifest(manifest, total_tokens, failed)
            log.info(f"  Checkpoint saved — {i+1} papers, ~{total_tokens:,} tokens so far")

    # ── Step 4: Final manifest save ──
    _save_manifest(manifest, total_tokens, failed)

    # ── Step 5: Summary ──
    log.info("\n" + "=" * 60)
    log.info("DONE")
    log.info(f"  Papers downloaded : {len(manifest)}")
    log.info(f"  Failed / skipped  : {len(failed)}")
    log.info(f"  Total ~tokens     : {total_tokens:,}")
    log.info(f"  Files saved to    : {RAW_DIR}")
    log.info(f"  Manifest saved to : {MANIFEST}")

    if total_tokens < 2_000_000:
        needed = 2_000_000 - total_tokens
        log.warning(
            f"\n  ⚠  You have ~{total_tokens:,} tokens. "
            f"Need ~{needed:,} more for Round 1 minimum (2M tokens). "
            f"Re-run with --max-papers {max_papers * 2}"
        )
    else:
        log.info(f"\n  ✓  Round 1 threshold met (2M tokens minimum)")

    log.info("=" * 60)


def _save_manifest(manifest: list, total_tokens: int, failed: list):
    data = {
        "generated_at":   datetime.utcnow().isoformat(),
        "total_papers":   len(manifest),
        "total_est_tokens": total_tokens,
        "failed_ids":     failed,
        "papers":         manifest,
    }
    MANIFEST.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Alzheimer's papers from PubMed Central")
    parser.add_argument(
        "--max-papers", type=int, default=3000,
        help="Max papers to fetch across all queries (default: 3000 ≈ 2M+ tokens)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Save manifest every N papers as a checkpoint (default: 100)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.15,
        help="Seconds between API calls (default: 0.15 with API key)"
    )
    args = parser.parse_args()
    main(args.max_papers, args.batch_size, args.delay)