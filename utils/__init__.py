import os

import numpy as np
import pandas as pd
from pathlib import Path
from results import PATH as RESULT_PATH
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ConfigSpace import Configuration
from smac.facade.abstract_facade import AbstractFacade


PATH = Path(__file__).parents[0]
OLLAMA_URL = "clusters.almaai.unibo.it"
OLLAMA_PORT = 11434
HUGGINGFACE_NAME_MAP = {
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
}


class HuggingFaceEmbeddingAdapter:
    def __init__(self, model_name: str, trust_remote_code: bool = False):
        self.embedding_model = HuggingFaceEmbedding(model_name=model_name, trust_remote_code=trust_remote_code, embed_batch_size=1024)

    def embed_documents(self, documents: list) -> list:
        return self.embedding_model.get_text_embedding_batch(documents)

    def embed_query(self, query: str) -> list:
        return self.embedding_model.get_text_embedding_batch([query])[0]


class ResultSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResultSingleton, cls).__new__(cls)
            cls._instance.results = []  # List of dictionaries with the results.

        return cls._instance

    def append(self, result: dict[int: dict[str: float]]) -> None:
        self.results.append(result)

    def save_results(self, name: str, path: PATH = RESULT_PATH) -> None:
        # Convert to DataFrame
        # Every dictionary has the same keys
        df = pd.DataFrame(self.results)
        df.to_csv(path / f"{name}_results.csv", index=False)

    def check_if_results_exist(self, name: str) -> bool:
        return (RESULT_PATH / f"{name}_results.csv").exists()


def read_results_and_incumbents(columns: list[str]) -> (pd.DataFrame, pd.DataFrame):
    return (
        pd.read_csv(RESULT_PATH / "results.csv", usecols=columns, index_col=False),
        pd.read_csv(RESULT_PATH / "incumbents.csv", usecols=columns, index_col=False),
    )


def get_pareto_front(smac: AbstractFacade) -> tuple[list[Configuration], list[list[float]]]:
    """Returns the Pareto front of the runhistory.

    Returns
    -------
    configs : list[Configuration]
        The configs of the Pareto front.
    costs : list[list[float]]
        The costs from the configs of the Pareto front.
    """

    # Get costs from runhistory first
    average_costs = []
    configs = smac.runhistory.get_configs()
    for config in configs:
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each
        # configuration
        # Luckily, SMAC already does this for us
        average_cost = smac.runhistory.average_cost(config)
        average_costs += [average_cost]

    # Let's work with a numpy array
    costs = np.vstack(average_costs)

    is_efficient = np.arange(costs.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(
            costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(
            nondominated_point_mask[:next_point_index]) + 1

    return [configs[i] for i in is_efficient], [average_costs[i] for i in is_efficient]
