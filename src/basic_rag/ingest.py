import json

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

from src.shared.config import (
    CHUNKS_FILE,
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
)


def load_chunks() -> list[dict]:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]

    if not chunks:
        raise ValueError("No chunks found in chunks file.")

    sample = chunks[0]
    required = {"chunk_id", "pmc_id", "text"}
    missing = required - set(sample)

    if missing:
        raise ValueError(f"Chunk schema missing fields: {missing}")

    return chunks


def ingest() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks):,} chunks")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(CHROMA_COLLECTION)
        print(f"Deleted existing collection: {CHROMA_COLLECTION}")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100

    for start in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        batch = chunks[start:start + batch_size]

        collection.add(
            ids=[item["chunk_id"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[
                {
                    "pmc_id": item.get("pmc_id", ""),
                    "chunk_index": item.get("chunk_index", 0),
                    "word_count": item.get("word_count", 0),
                    "est_tokens": item.get("est_tokens", 0),
                }
                for item in batch
            ],
        )

    print(f"Stored {collection.count():,} chunks in {CHROMA_DIR}/")


if __name__ == "__main__":
    ingest()