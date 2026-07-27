"""
Retrieval benchmark runner (final clean version)

- Parallel retrieval over queries
- One file per (strategy, collection, dataset)
- Incremental updates per query
- Thread-safe writes
"""

import os
import json
import argparse
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from rag_pipelines.retrievers.hybrid_retrieval import run_retrieve


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

QUERY_CSV = "/data/lsanna/experiments-2025-hcaixia-automl-rag/data/test.csv"
OUTPUT_DIR = "outputs"
RETRIEVAL_DIR = os.path.join(OUTPUT_DIR, "retrieval")

os.makedirs(RETRIEVAL_DIR, exist_ok=True)

WRITE_LOCK = threading.Lock()

DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2-moe"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_RERANKING_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

DEFAULT_K = 20


# ---------------------------------------------------------------------------
# IO HELPERS
# ---------------------------------------------------------------------------

def retrieval_path(strategy: str, collection: str, dataset: str) -> str:
    safe_collection = collection.replace("/", "_")
    safe_dataset = dataset.replace("/", "_")

    return os.path.join(
        RETRIEVAL_DIR,
        f"{strategy}__{safe_collection}__{safe_dataset}.json"
    )


def load_store(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "strategy": None,
        "collection": None,
        "dataset": None,
        "results": {}
    }


def update_store(path: str, file_name: str, query: str, docs: list, meta: dict):
    with WRITE_LOCK:
        store = load_store(path)

        store["strategy"] = meta["strategy"]
        store["collection"] = meta["collection"]
        store["dataset"] = meta["dataset"]

        store["results"][file_name] = {
            "query": query,
            "docs": docs,
            "meta": meta,
        }

        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)

        os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# WORKER
# ---------------------------------------------------------------------------

def retrieve_single(row, args, store_path):
    query = row["question"].strip()
    file_name = row["file_name"].strip()

    docs = run_retrieve(
        query=query,
        k=args.k,
        collection=args.collection,
        embedder=args.embedding_model,
        sparse_model=args.sparse_model,
        strategy=args.strategy,
        mmr_lambda=args.mmr_lambda,
        mmr_candidates=args.mmr_candidates,
    )

    meta = {
        "strategy": args.strategy,
        "collection": args.collection,
        "dataset": args.dataset_name,
        "embedding_model": args.embedding_model,
        "sparse_model": args.sparse_model,
        "reranking_model": args.reranking_model,
        "mmr_lambda": args.mmr_lambda,
        "mmr_candidates": args.mmr_candidates,
    }

    update_store(store_path, file_name, query, docs, meta)

    return file_name, False


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------

def run(args):
    queries = pd.read_csv(args.query_csv).to_dict(orient="records")

    if args.test_mode:
        print("\n🧪 TEST MODE ACTIVE → 5 queries\n")
        queries = queries[:5]

    store_path = retrieval_path(
        args.strategy,
        args.collection,
        args.dataset_name
    )

    print("=" * 60)
    print(f"STRATEGY   = {args.strategy}")
    print(f"COLLECTION = {args.collection}")
    print(f"DATASET    = {args.dataset_name}")
    print(f"OUTPUT     = {store_path}")
    print(f"QUERIES    = {len(queries)}")
    print("=" * 60)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(retrieve_single, row, args, store_path)
            for row in queries
        ]

        for i, future in enumerate(as_completed(futures)):
            file_name, _ = future.result()
            print(f"[{i}] done → {file_name}")

    print("\nRetrieval complete.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["bm25", "mmr", "ensemble", "dense", "double_dense"],
    )

    parser.add_argument("--collection", type=str, required=True)
    parser.add_argument("--query_csv", type=str, default=QUERY_CSV)

    parser.add_argument("--dataset_name", type=str, default="UNKNOWN")

    parser.add_argument("--embedding_model", type=str,
                        default=DEFAULT_EMBEDDING_MODEL)

    parser.add_argument("--sparse_model", type=str,
                        default=DEFAULT_SPARSE_MODEL)

    parser.add_argument("--reranking_model", type=str,
                        default=DEFAULT_RERANKING_MODEL)

    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--mmr_lambda", type=float)
    parser.add_argument("--mmr_candidates", type=int)

    parser.add_argument("--test_mode", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run(args)