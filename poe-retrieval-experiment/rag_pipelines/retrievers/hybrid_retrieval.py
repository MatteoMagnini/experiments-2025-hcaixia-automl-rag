import logging
import time
from typing import Any

import httpx
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from retrievers.environment import Environment

logger = logging.getLogger(__name__)

QDRANT_RETRIES = 5
QDRANT_RETRY_DELAY = 2


# =========================================================
# EMBEDDINGS + RERANK
# =========================================================

def _get_embeddings(texts: list[str], embedder: str) -> list[list[float]]:
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{Environment.model_server_url}/embeddings",
            json={"input": texts, "model": embedder},
        )
        if resp.status_code != 200:
            raise RuntimeError(resp.text)

    return [item["embeddings"] for item in resp.json()["data"]]


def _rerank(query: str, documents: list[str]) -> list[float]:
    if not documents:
        return []

    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{Environment.model_server_url}/rerank",
            json={
                "query": query,
                "documents": documents,
                "reranker": Environment.reranking_model,
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(resp.text)

    results = resp.json()["results"]
    return [r["score"] for r in sorted(results, key=lambda x: x["index"])]

# =========================================================
# CLIENT
# =========================================================

def build_client() -> QdrantClient:
    return QdrantClient(
        url=Environment.qdrant_url,
        port=Environment.qdrant_port,
        timeout=60,
        headers={"Authorization": Environment.qdrant_authorization},
    )



# =========================================================
# QDRANT RETRY
# =========================================================

def qdrant_retry(func, *args, **kwargs):
    last_exc = None

    for attempt in range(1, QDRANT_RETRIES + 1):
        try:
            return func(*args, **kwargs)

        except (
            ResponseHandlingException,
            httpx.RemoteProtocolError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.ConnectError,
            httpx.ReadError,
        ) as exc:

            last_exc = exc

            if attempt == QDRANT_RETRIES:
                logger.exception(
                    "Qdrant request failed permanently after %s attempts",
                    QDRANT_RETRIES,
                )
                raise

            sleep_time = QDRANT_RETRY_DELAY * attempt

            logger.warning(
                "Qdrant request failed (%s/%s): %s. Retrying in %ss...",
                attempt,
                QDRANT_RETRIES,
                exc,
                sleep_time,
            )

            time.sleep(sleep_time)

    raise last_exc





# =========================================================
# RRF FUSION
# =========================================================

def calculate_rrf(
    ranked_lists: list[list[dict[str, Any]]],
    id_key: str = "id",
    k: int = 60,
) -> list[dict[str, Any]]:

    all_ids = set()
    rank_maps = []

    for lst in ranked_lists:
        rank_map = {item[id_key]: idx + 1 for idx, item in enumerate(lst)}
        all_ids |= set(rank_map.keys())
        rank_maps.append(rank_map)

    fused = []

    for doc_id in all_ids:
        rrf_score = sum(
            1.0 / (k + rm[doc_id])
            for rm in rank_maps
            if doc_id in rm
        )

        doc = next(
            item for lst in ranked_lists for item in lst if item[id_key] == doc_id
        )

        fused.append({**doc, "rrf_score": rrf_score})

    return sorted(fused, key=lambda x: x["rrf_score"], reverse=True)


# =========================================================
# HELPERS
# =========================================================

def _get_source(doc_id: Any, points_by_id: dict) -> Any:
    pt = points_by_id.get(doc_id)
    return pt.payload.get("source") if pt else None


# =========================================================
# RETRIEVERS
# =========================================================

def dense_retrieve(client, query_vector, query, collection, k, with_payload=True):
    resp = qdrant_retry(
        client.query_points,
        collection_name=collection,
        query=query_vector,
        using="dense_vector",
        limit=k * 5,
        with_payload=with_payload,
    )

    points_by_id = {pt.id: pt for pt in resp.points}

    docs = [
        {
            "id": str(pt.id),
            "query": query,
            "file_name": _get_source(pt.id, points_by_id),
            "text": pt.payload.get("text", ""),
            "cosine_similarity": float(pt.score),
            "retriever": "dense",
        }
        for pt in resp.points
    ]

    scores = _rerank(query, [d["text"] for d in docs])

    reranked = sorted(
        [{**d, "cross_encoder_score": s} for d, s in zip(docs, scores)],
        key=lambda x: x["cross_encoder_score"],
        reverse=True,
    )

    return reranked[:k]


def bm25_retrieve(client, query, collection, sparse_model, k, with_payload=True):
    resp = qdrant_retry(
        client.query_points,
        collection_name=collection,
        query=models.Document(text=query, model=sparse_model),
        using="bm25_sparse_vector",
        limit=k * 5,
        with_payload=with_payload,
    )

    points_by_id = {pt.id: pt for pt in resp.points}

    docs = [
        {
            "id": str(pt.id),
            "query": query,
            "file_name": _get_source(pt.id, points_by_id),
            "text": pt.payload.get("text", ""),
            "BM25_score": float(pt.score),
            "retriever": "bm25",
        }
        for pt in resp.points
    ]

    scores = _rerank(query, [d["text"] for d in docs])

    reranked = sorted(
        [{**d, "cross_encoder_score": s} for d, s in zip(docs, scores)],
        key=lambda x: x["cross_encoder_score"],
        reverse=True,
    )

    return reranked[:k]


def mmr_retrieve(
    client,
    query_vector,
    query,
    collection,
    k,
    mmr_lambda,
    mmr_candidates,
    with_payload=True,
):
    resp = qdrant_retry(
        client.query_points,
        collection_name=collection,
        query=models.NearestQuery(
            nearest=query_vector,
            mmr=models.Mmr(
                diversity=mmr_lambda,
                candidates_limit=mmr_candidates,
            ),
        ),
        using="dense_vector",
        limit=k,
        with_payload=with_payload,
    )

    points_by_id = {pt.id: pt for pt in resp.points}

    docs = [
        {
            "id": str(pt.id),
            "query": query,
            "file_name": _get_source(pt.id, points_by_id),
            "text": pt.payload.get("text", ""),
            "MMR_score": float(pt.score),
            "retriever": "mmr",
        }
        for pt in resp.points
    ]

    scores = _rerank(query, [d["text"] for d in docs])

    reranked = sorted(
        [{**d, "cross_encoder_score": s} for d, s in zip(docs, scores)],
        key=lambda x: x["cross_encoder_score"],
        reverse=True,
    )

    return reranked[:k]


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_retrieve(
    query: str,
    k: int,
    collection: str,
    embedder: str,
    sparse_model: str = "Qdrant/bm25",
    strategy: str = "bm25",
    mmr_lambda: float = 0.5,
    mmr_candidates: int = 100,
    with_payload: bool = True,
) -> list[dict[str, Any]]:

    client = build_client()
    query_vector = _get_embeddings([query], embedder)[0]

    if strategy == "bm25":
        return bm25_retrieve(
            client, query, collection, sparse_model, k, with_payload
        )

    elif strategy == "dense":
        return dense_retrieve(
            client, query_vector, query, collection, k, with_payload
        )

    elif strategy == "mmr":
        return mmr_retrieve(
            client,
            query_vector,
            query,
            collection,
            k,
            mmr_lambda,
            mmr_candidates,
            with_payload,
        )

    elif strategy == "ensemble":
        dense = dense_retrieve(
            client, query_vector, query, collection, k, with_payload
        )

        bm25 = bm25_retrieve(
            client, query, collection, sparse_model, k, with_payload
        )

        return calculate_rrf([dense, bm25])

    elif strategy == "double_dense":
        dense = dense_retrieve(
            client, query_vector, query, collection, k, with_payload
        )

        mmr = mmr_retrieve(
            client,
            query_vector,
            query,
            collection,
            k,
            mmr_lambda,
            mmr_candidates,
            with_payload,
        )

        return calculate_rrf([dense, mmr])
    
    elif strategy == "all":
        dense = dense_retrieve(
            client, query_vector, query, collection, k, with_payload
        )

        bm25 = bm25_retrieve(
            client, query, collection, sparse_model, k, with_payload
        )

        mmr = mmr_retrieve(
            client,
            query_vector,
            query,
            collection,
            k,
            mmr_lambda,
            mmr_candidates,
            with_payload,
        )

        return calculate_rrf([dense, mmr, bm25])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")