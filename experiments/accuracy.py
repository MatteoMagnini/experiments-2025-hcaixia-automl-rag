"""Retrieval-accuracy evaluation.

Each question was generated from a document (the `id` column) and each chunk
comes from a document (the `document_id` column in the lookup table). A retrieval
is counted correct if at least one retrieved chunk comes from the question's
source document.
"""
import pandas as pd
from langchain_core.documents import Document

from qdrant import CHUNK_LOOKUP_FILE_NAME, CHUNK_DOCUMENTS_LOOKUP_FILE_NAME


def trim_batch_results(batch_results: list[list[Document]], limit: int) -> list[list[Document]]:
    return [list(result)[:limit] for result in batch_results]


def extract_document_id(result: Document, chunk_lookup: dict[str, int], chunk_document_lookup: dict[int, int]) -> int | None:
    metadata = result.metadata or {}
    document_id = metadata.get("document_id")
    if pd.notna(document_id):
        return int(document_id)

    chunk_id = metadata.get("chunk_id")
    if pd.notna(chunk_id):
        return chunk_document_lookup.get(int(chunk_id))

    if result.page_content not in chunk_lookup:
        return None

    return chunk_document_lookup.get(chunk_lookup[result.page_content])


def compute_accuracy(
    training_questions: pd.DataFrame,
    results: list[list[Document]],
    chunk_lookup: dict[str, int],
    chunk_document_lookup: dict[int, int],
) -> float:
    correct_retrievals = 0
    for i, question in training_questions.iterrows():
        question_document_id = int(question["id"])
        retrieved_documents = {
            document_id
            for result in results[i]
            if (document_id := extract_document_id(result, chunk_lookup, chunk_document_lookup)) is not None
        }
        if question_document_id in retrieved_documents:
            correct_retrievals += 1

    return correct_retrievals / len(training_questions)


def embeddings_cache_ready(embeddings_path) -> bool:
    required_files = [
        embeddings_path / CHUNK_LOOKUP_FILE_NAME,
        embeddings_path / CHUNK_DOCUMENTS_LOOKUP_FILE_NAME,
    ]
    return embeddings_path.exists() and all(path.exists() for path in required_files)
