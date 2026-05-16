import chromadb
from chromadb.utils import embedding_functions

from src.shared.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    RAG_TOP_K,
)
from src.shared.llm_client import call_llm


embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_collection(
    name=CHROMA_COLLECTION,
    embedding_function=embedding_fn,
)

SYSTEM_PROMPT = (
    "You are a careful biomedical research assistant. "
    "Answer using only the provided context chunks. "
    "If the answer is not supported by the context, say so clearly."
)


def retrieve(question: str, top_k: int = RAG_TOP_K) -> list[dict]:
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    rows = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        rows.append({
            "text": document,
            "metadata": metadata,
            "distance": distance,
            "similarity": 1 - distance,
        })

    return rows


def run_query(question: str) -> dict:
    rows = retrieve(question)

    context_parts = []

    for index, row in enumerate(rows, start=1):
        metadata = row["metadata"]
        chunk_text = row["text"][:1200]

        context_parts.append(
            f"[Chunk {index} | PMC: {metadata.get('pmc_id')}]\n"
            f"{chunk_text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer using only the context:"
    )

    result = call_llm(prompt, system_prompt=SYSTEM_PROMPT)
    result["pipeline"] = "Basic RAG"
    result["sources"] = [row["metadata"].get("pmc_id") for row in rows]

    return result


if __name__ == "__main__":
    question = "What is the role of APOE4 in Alzheimer's disease risk?"
    result = run_query(question)

    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Tokens: {result['total_tokens']}")
    print(f"Latency: {result['latency_seconds']}s")
    print(f"Cost: ${result['cost_usd']:.8f}")