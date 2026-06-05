"""Single-trial runner: evaluate one SMAC configuration end-to-end.

Builds/loads the embeddings for the config's embedder+chunking, retrieves over
the training questions, scores retrieval accuracy and (optionally) the
generative metrics, records the result row and returns the SMAC cost vector. A
corrupted local Qdrant cache is rebuilt once before giving up.
"""
import os
import shutil
import traceback
from pathlib import Path
from time import time

import pandas as pd
from ConfigSpace import Configuration
from langchain_qdrant import QdrantVectorStore

from data import PATH as DATA_PATH, TRAINING_FILE_NAME
from qdrant import PATH as QDRANT_PATH, DATABASE_NAME_FAQ
from qdrant.__main__ import main as create_embeddings
from utils import DEFAULT_PROVIDER, build_embeddings, get_embeddings_path

from .accuracy import compute_accuracy, embeddings_cache_ready, trim_batch_results
from .constants import DEFAULT_EVAL_SAMPLE_SIZE, FAILURE_ACCURACY_COST
from .evaluation import run_alternative_evaluation
from .results_recording import (
    build_configuration_payload,
    build_result_payload,
    cache_experiment_row,
    extract_smac_costs,
)
from .retrievers import (
    build_retriever,
    invalidate_embeddings_path_cache,
    load_chunk_lookup_mappings,
)


TRAINING_FILE = DATA_PATH / TRAINING_FILE_NAME


def is_recoverable_qdrant_error(exc: Exception) -> bool:

    message = str(exc).lower()
    recoverable_markers = (
        "already accessed",          # local storage lock held by another client
        "storage folder",
        "no such file or directory",
        "collection",                # missing/corrupted collection metadata
        "corrupt",
        "database disk image is malformed",
    )
    return any(marker in message for marker in recoverable_markers)


def rebuild_embeddings_cache(embeddings_path: Path, chunk_token_length: int, overlap_percentage: float, embedder: str, provider: str) -> None:
    invalidate_embeddings_path_cache(embeddings_path)
    if embeddings_path.exists():
        shutil.rmtree(embeddings_path, ignore_errors=True)
    create_embeddings(chunk_token_length, overlap_percentage, embedder, provider=provider)


def run_experiment(config: Configuration, seed: int = 0, budget: int = 100, provider: str = DEFAULT_PROVIDER, gen_model: str = None, pregenerated_gold_answers: list[str] = None, eval_sample_size: int = DEFAULT_EVAL_SAMPLE_SIZE) -> dict[str, float]:
    number_of_docs = int(config["num_docs"])
    configuration = build_configuration_payload(config)
    chunk_token_length = int(config["chunk_token_length"])
    overlap_percentage = float(config["overlap_percentage"])
    embedder = str(config["embedder"])
    overlap = int(chunk_token_length * overlap_percentage)
    embeddings_path = get_embeddings_path(
        QDRANT_PATH / DATABASE_NAME_FAQ,
        provider,
        embedder,
        chunk_token_length,
        overlap,
    )

    for attempt in range(2):
        # Check if the embeddings are already available
        vectorstore = None
        try:
            print(f"Running experiment with config: {config}")
            if embeddings_cache_ready(embeddings_path):
                print(f"Skipping embeddings creation for {config['embedder']} with chunk token length {chunk_token_length} and overlap {overlap}")
            else:
                create_embeddings(chunk_token_length, overlap_percentage, embedder, provider=provider)

            training_questions = pd.read_csv(TRAINING_FILE).reset_index(drop=True)
            embeddings = build_embeddings(embedder, provider)

            qdrant_url = os.environ.get("QDRANT_URL")
            qdrant_api_key = os.environ.get("QDRANT_API_KEY")

            if qdrant_url:
                vectorstore = QdrantVectorStore.from_existing_collection(
                    embedding=embeddings,
                    collection_name=DATABASE_NAME_FAQ,
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                )
            else:
                vectorstore = QdrantVectorStore.from_existing_collection(
                    embedding=embeddings,
                    collection_name=DATABASE_NAME_FAQ,
                    path=str(embeddings_path),
                )

            ## Retriever part
            retriever = build_retriever(config, vectorstore, embeddings_path)

            ## Retrieve
            results: list[list] = []
            print("Retrieving...")
            batch_size = 128
            questions = training_questions["question"]
            start_time = time()
            for j in range(0, len(questions), batch_size):
                print(f"Retrieving batch {j} to {j + batch_size}")
                batch_questions = list(questions[j:j + batch_size])
                batch_results = trim_batch_results(retriever.batch(batch_questions), number_of_docs)
                results.extend(batch_results)
            end_time = time()
            retrieval_time = end_time - start_time
            print(f"Retrieval time: {retrieval_time}")

            ## Evaluate
            chunk_lookup, chunk_document_lookup = load_chunk_lookup_mappings(embeddings_path)
            accuracy = compute_accuracy(training_questions, results, chunk_lookup, chunk_document_lookup)
            print(f"Accuracy: {accuracy}")

            # Evaluation
            alt_results = None

            if eval_sample_size > 0:
                try:
                    n_eval = min(eval_sample_size, len(training_questions))
                    eval_questions = training_questions.sample(n=n_eval, random_state=42)
                    # Positional indices into `results` (index is a RangeIndex after reset_index above).
                    eval_indices = eval_questions.index.tolist()

                    eval_user_inputs = [training_questions.iloc[idx]["question"] for idx in eval_indices]
                    eval_retrieved_contexts = [
                        [chunk.page_content for chunk in results[idx]] for idx in eval_indices
                    ]

                    # Get gen_model from SMAC configuration or overrides
                    chosen_gen_model = config.get("gen_model") if config.get("gen_model") else gen_model

                    print("Running alternative metrics evaluation (ROUGE, BERTScore, Embedding Similarity)...")
                    alt_results = run_alternative_evaluation(
                        eval_user_inputs,
                        eval_retrieved_contexts,
                        provider,
                        embeddings_model=embeddings,
                        gen_model=chosen_gen_model,
                        pregenerated_gold_answers=pregenerated_gold_answers,
                    )
                except Exception as e:
                    print(f"Failed to prepare or run evaluation: {e}")
                    print(traceback.format_exc())

            result = build_result_payload(
                config,
                number_of_docs,
                1 - accuracy,
                alt_metrics=alt_results,
            )
            cache_experiment_row(configuration | result)

            return extract_smac_costs(result)
        except Exception as exc:
            if attempt == 0 and is_recoverable_qdrant_error(exc):
                print(f"Recovering corrupted Qdrant cache at {embeddings_path}: {exc}")
                rebuild_embeddings_cache(embeddings_path, chunk_token_length, overlap_percentage, embedder, provider)
                continue

            print(f"Experiment failed for config: {config}")
            print(f"{type(exc).__name__}: {exc}")
            print(traceback.format_exc())
            failure_result = build_result_payload(config, number_of_docs, FAILURE_ACCURACY_COST)
            cache_experiment_row(configuration | failure_result | {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})

            return extract_smac_costs(failure_result)
        finally:
            # Release the local Qdrant storage lock so the next trial can reopen
            # the same path without a spurious "already accessed" error.
            if vectorstore is not None:
                try:
                    vectorstore.client.close()
                except Exception:
                    pass
