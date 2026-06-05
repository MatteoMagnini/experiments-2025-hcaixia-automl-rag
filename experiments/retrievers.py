"""Retriever construction and the per-embeddings-path caches.

Builds the base/ensemble/bm25/mmr retrievers for a trial. The chunk lookup
tables and the BM25 index are read/built once per embeddings_path and reused
across the many SMAC trials that share the same embedder/chunking, instead of
re-reading and re-indexing on every single trial.
"""
from pathlib import Path

import pandas as pd
from ConfigSpace import Configuration
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from qdrant import CHUNK_LOOKUP_FILE_NAME, CHUNK_DOCUMENTS_LOOKUP_FILE_NAME


_chunk_lookup_cache: dict[str, tuple[dict[str, int], dict[int, int]]] = {}
_bm25_retriever_cache: dict[str, BM25Retriever] = {}


def invalidate_embeddings_path_cache(embeddings_path: Path) -> None:
    key = str(embeddings_path)
    _chunk_lookup_cache.pop(key, None)
    _bm25_retriever_cache.pop(key, None)


def load_chunk_lookup_mappings(embeddings_path: Path) -> tuple[dict[str, int], dict[int, int]]:
    key = str(embeddings_path)
    if key not in _chunk_lookup_cache:
        chunk_lookup = pd.read_csv(embeddings_path / CHUNK_LOOKUP_FILE_NAME)
        chunk_document_lookup = pd.read_csv(embeddings_path / CHUNK_DOCUMENTS_LOOKUP_FILE_NAME)
        _chunk_lookup_cache[key] = (
            {row["chunk"]: int(row["id"]) for _, row in chunk_lookup.iterrows()},
            {int(row["chunk_id"]): int(row["document_id"]) for _, row in chunk_document_lookup.iterrows()},
        )
    return _chunk_lookup_cache[key]


def build_bm25_documents(embeddings_path: Path) -> list[Document]:
    chunk_lookup = pd.read_csv(embeddings_path / CHUNK_LOOKUP_FILE_NAME)
    _, chunk_document_lookup = load_chunk_lookup_mappings(embeddings_path)
    return [
        Document(
            page_content=row["chunk"],
            metadata={
                "chunk_id": int(row["id"]),
                "document_id": chunk_document_lookup[int(row["id"])],
            },
        )
        for _, row in chunk_lookup.iterrows()
    ]


def build_bm25_retriever(embeddings_path: Path, number_of_docs: int) -> BM25Retriever:
    key = str(embeddings_path)
    if key not in _bm25_retriever_cache:
        _bm25_retriever_cache[key] = BM25Retriever.from_documents(build_bm25_documents(embeddings_path))
    retriever = _bm25_retriever_cache[key]
    retriever.k = number_of_docs
    return retriever


def build_vector_retriever(vectorstore: QdrantVectorStore, number_of_docs: int):
    return vectorstore.as_retriever(search_kwargs={"k": number_of_docs})


def build_mmr_retriever(
    vectorstore: QdrantVectorStore,
    number_of_docs: int,
    fetch_k: int,
    lambda_mult: float,
):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": number_of_docs,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )


def build_retriever(config: Configuration, vectorstore: QdrantVectorStore, embeddings_path: Path):
    number_of_docs = int(config["num_docs"])
    retriever_type = str(config["retriever"])

    match retriever_type:
        case "base":
            return build_vector_retriever(vectorstore, number_of_docs)
        case "ensemble":
            return EnsembleRetriever(
                retrievers=[
                    build_vector_retriever(vectorstore, number_of_docs),
                    build_bm25_retriever(embeddings_path, number_of_docs),
                ],
                weights=[0.5, 0.5],
            )
        case "bm25_only":
            return build_bm25_retriever(embeddings_path, number_of_docs)
        case "mmr":
            return build_mmr_retriever(
                vectorstore,
                number_of_docs,
                fetch_k=int(config["mmr_fetch_k"]),
                lambda_mult=float(config["mmr_lambda_mult"]),
            )
        case _:
            raise ValueError(f"Unknown retriever type {retriever_type}")
