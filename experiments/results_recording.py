"""Per-trial result shaping and persistence.

Turns a config + raw metrics into the SMAC cost vector and the self-describing
result row, and appends that row to the on-disk results cache (and the in-memory
ResultSingleton).
"""
import pandas as pd
from ConfigSpace import Configuration

from results.cache import PATH as CACHE_PATH
from utils import ResultSingleton

from .constants import FAILURE_ACCURACY_COST, SMAC_OBJECTIVES


def extract_smac_costs(result: dict[str, float]) -> dict[str, float]:
    """Pull the SMAC objective values out of a full result payload.

    SMAC cannot handle NaN/None costs, so any unavailable metric (e.g. BERTScore
    skipped while offline) is reported as the worst-case cost instead.
    """
    costs = {}
    for key in SMAC_OBJECTIVES:
        value = result.get(key)
        costs[key] = FAILURE_ACCURACY_COST if value is None or pd.isna(value) else float(value)
    return costs


def normalize_optional_config_value(value):
    return value if value is not None else ""


def build_result_payload(
    config: Configuration,
    number_of_docs: int,
    accuracy_cost: float,
    alt_metrics: dict[str, float] = None,
) -> dict[str, float]:
    payload = {
        "1 - accuracy": float(accuracy_cost),
        "number of documents": float(number_of_docs),
    }

    # Define a default set of keys for alternative metrics to guarantee shape consistency
    alt_keys = [
        "rouge1_gold", "rouge2_gold", "rougeL_gold",
        "bert_f1_gold", "bert_precision_gold", "bert_recall_gold", "emb_sim_gold",
        "rouge1_query", "rouge2_query", "rougeL_query", "bert_f1_query", "emb_sim_query",
        "rouge1_context", "rouge2_context", "rougeL_context", "bert_f1_context", "emb_sim_context"
    ]

    for k in alt_keys:
        val = alt_metrics.get(k) if alt_metrics else None
        # A skipped/failed metric stays NaN instead of being faked as 0.0, so it
        # is never confused with a genuinely poor score downstream.
        if val is None:
            payload[f"1 - {k}"] = float("nan")
            payload[k] = float("nan")
        else:
            payload[f"1 - {k}"] = float(1.0 - val)
            payload[k] = float(val)

    return payload


def build_configuration_payload(config: Configuration) -> dict[str, object]:
    return {
        "chunk_token_length": int(config["chunk_token_length"]),
        "overlap_percentage": float(config["overlap_percentage"]),
        "retriever": str(config["retriever"]),
        "embedder": str(config["embedder"]),
        "mmr_fetch_k": normalize_optional_config_value(config.get("mmr_fetch_k")),
        "mmr_lambda_mult": normalize_optional_config_value(config.get("mmr_lambda_mult")),
        "gen_model": normalize_optional_config_value(config.get("gen_model")),
    }


def reset_results_cache() -> None:
    cache_file = CACHE_PATH / "results.csv"
    if cache_file.exists():
        cache_file.unlink()


def cache_experiment_row(row: dict[str, object]) -> None:
    ResultSingleton().append(row)
    cache_file = CACHE_PATH / "results.csv"
    # Write the header the first time so the CSV is self-describing instead of a
    # headerless dump.
    write_header = not cache_file.exists() or cache_file.stat().st_size == 0
    pd.DataFrame([row]).to_csv(cache_file, mode="a", header=write_header, index=False)
