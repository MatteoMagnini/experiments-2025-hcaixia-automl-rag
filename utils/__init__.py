import numpy as np
import pandas as pd
from pathlib import Path
from results import PATH as RESULT_PATH
from smac import HyperparameterOptimizationFacade as HPOFacade
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ConfigSpace import Configuration
import matplotlib.pyplot as plt


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


class ResultSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResultSingleton, cls).__new__(cls)
            cls._instance.results = []  # List of dictionaries with the results.

        return cls._instance

    def append(self, result: dict[int: dict[str: float]]) -> None:
        self.results.append(result)

    def save_results(self, name: str) -> None:
        # Convert to DataFrame
        # Every dictionary has the same keys
        df = pd.DataFrame(self.results)
        df.to_csv(RESULT_PATH / f"{name}_results.csv", index=False)

    def check_if_results_exist(self, name: str) -> bool:
        return (RESULT_PATH / f"{name}_results.csv").exists()


def plot_pareto(smac: HPOFacade, incumbents: list[Configuration]) -> None:
    """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
    average_costs = []
    average_pareto_costs = []
    for config in smac.runhistory.get_configs():
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
        average_cost = smac.runhistory.average_cost(config)

        if config in incumbents:
            average_pareto_costs += [average_cost]
        else:
            average_costs += [average_cost]

    # Let's work with a numpy array
    if len(average_costs) == 1:
        costs = np.array(average_costs)
    else:
        costs = np.vstack(average_costs)
    pareto_costs = np.vstack(average_pareto_costs)
    pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them

    costs_x, costs_y = costs[:, 0], costs[:, 1]
    pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]

    plt.scatter(costs_x, costs_y, marker="x", label="Configuration")
    plt.scatter(pareto_costs_x, pareto_costs_y, marker="x", c="r", label="Incumbent")
    plt.step(
        [pareto_costs_x[0]] + pareto_costs_x.tolist() + [np.max(costs_x)],  # We add bounds
        [np.max(costs_y)] + pareto_costs_y.tolist() + [np.min(pareto_costs_y)],  # We add bounds
        where="post",
        linestyle=":",
    )

    plt.title("Pareto-Front")
    plt.xlabel(smac.scenario.objectives[0])
    plt.ylabel(smac.scenario.objectives[1])
    plt.legend()
    plt.show()
