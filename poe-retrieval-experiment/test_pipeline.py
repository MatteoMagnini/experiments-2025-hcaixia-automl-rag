"""
Minimal working test — runs the full pipeline without CLI.

Run:
    python test_experiment.py

Requires env vars:
    OPENROUTER_API_KEY
    RAG_PIPELINES_URL
    RAG_PIPELINES_API_KEY
"""

from __future__ import annotations

import os
import sys

from rag_prompts import HYDOC_PROMPT, format_rag_prompt
from llm_client import query_openrouter, extract_text
from retriever import run_retrieval_pipeline
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QUERY        = "Posso prendere la tachipirina in gravidanza?"
MODEL        = "meta-llama/llama-3.3-70b-instruct"
TEMPERATURE  = 0.9
COLLECTION   = "chatfaq_gravidanza"
EMBEDDING    = "mixedbread-ai/mxbai-embed-large-v1"
SPARSE       = "Qdrant/bm25"
RERANKING    = "jinaai/jina-reranker-v2-base-multilingual"
K            = 3
RETRIEVERS_K = 10


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

openrouter_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_key:
    sys.exit("ERROR: OPENROUTER_API_KEY not set.")


# ---------------------------------------------------------------------------
# Step 1 — Retrieval
# ---------------------------------------------------------------------------

print("[ 1/3 ] Retrieval...")
top_docs = run_retrieval_pipeline(
    query=QUERY,
    openrouter_api_key=openrouter_key,
    llm_model=MODEL,
    hydoc_prompt=HYDOC_PROMPT,
    collection=COLLECTION,
    embedding_model=EMBEDDING,
    sparse_model=SPARSE,
    reranking_model=RERANKING,
    k=K,
    retrievers_k=RETRIEVERS_K,
    llm_temperature=TEMPERATURE,
)
assert isinstance(top_docs, list), "Expected a list of docs"
assert len(top_docs) <= K,        "Got more docs than requested k"
print(f"       OK — {len(top_docs)} docs retrieved")


# ---------------------------------------------------------------------------
# Step 2 — HyDE answer
# ---------------------------------------------------------------------------

print("[ 2/3 ] HyDE answer...")
hydoc_response, status = query_openrouter(
    api_key=openrouter_key,
    model_name=MODEL,
    sys_prompt=HYDOC_PROMPT.format(query=QUERY),
    user_prompt=QUERY,
    temperature=TEMPERATURE,
)
assert status == 200, f"HyDE call failed (HTTP {status}): {hydoc_response}"
hydoc_text = extract_text(hydoc_response)
assert hydoc_text, "HyDE returned empty text"
print(f"       OK — {len(hydoc_text)} chars")


# ---------------------------------------------------------------------------
# Step 3 — Final grounded answer
# ---------------------------------------------------------------------------

print("[ 3/3 ] Final answer...")
rag_response, status = query_openrouter(
    api_key=openrouter_key,
    model_name=MODEL,
    sys_prompt=format_rag_prompt(
        query=QUERY,
        hydoc=hydoc_text,
        docs=[doc.get("text", "") for doc in top_docs],
    ),
    user_prompt=QUERY,
    temperature=TEMPERATURE,
)
assert status == 200, f"RAG call failed (HTTP {status}): {rag_response}"
final_answer = extract_text(rag_response)
assert final_answer, "RAG returned empty answer"
print(f"       OK — {len(final_answer)} chars")


# ---------------------------------------------------------------------------
# Print result
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"QUERY:  {QUERY}")
print(f"{'='*60}")
print(final_answer)
print(f"{'='*60}\n")

print("All steps passed.")