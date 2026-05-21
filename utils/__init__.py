import numpy as np
import pandas as pd
from pathlib import Path
from ollama import ResponseError
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from results import PATH as RESULT_PATH
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ConfigSpace import Configuration
from smac.facade.abstract_facade import AbstractFacade


PATH = Path(__file__).parents[0]
OLLAMA_URL = "localhost"
OLLAMA_PORT = 11434
DEFAULT_PROVIDER = "ollama"
OLLAMA_EMBEDDERS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "embeddinggemma",
    "nomic-embed-text-v2-moe",
    "qwen3-embedding:0.6b",
    "qwen3-embedding:4b",
    "ibm/granite-embedding:278m",
    "ibm/granite-embedding:125m",
    "ibm/granite-embedding:107m",
    "ibm/granite-embedding:30m",

]
HUGGINGFACE_NAME_MAP = {
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "bert-base-cased": "google-bert/bert-base-cased",
    "bert-base-italian-xxl-cased": "dbmdz/bert-base-italian-xxl-cased",
    "biobert-base-cased": "dmis-lab/biobert-base-cased-v1.1"
}
CHUNKING_DIRECTORY_BY_PROVIDER = {
    "huggingface": "hf_tokens_v1",
    "ollama": "char_chunks_v2",
}
MIN_OLLAMA_EMBED_WORDS = 32
OLLAMA_RETRY_SHRINK_FACTOR = 0.75


def _coerce_texts(documents: list[str | Document]) -> list[str]:
    return [document.page_content if isinstance(document, Document) else document for document in documents]


def _truncate_to_word_limit(text: str, word_limit: int) -> str:
    words = text.split()
    if len(words) <= word_limit:
        return text

    truncated = " ".join(words[:word_limit]).strip()
    return truncated if truncated else text


def _is_ollama_context_error(exc: Exception) -> bool:
    return isinstance(exc, ResponseError) and "context length" in str(exc).lower()


class HuggingFaceEmbeddingAdapter:
    def __init__(self, model_name: str, trust_remote_code: bool = False):
        self.embedding_model = HuggingFaceEmbedding(model_name=model_name, trust_remote_code=trust_remote_code, embed_batch_size=1024)

    def embed_documents(self, documents: list) -> list:
        return self.embedding_model.get_text_embedding_batch(_coerce_texts(documents))

    def embed_query(self, query: str) -> list:
        return self.embedding_model.get_text_embedding_batch([query])[0]


class OllamaEmbeddingAdapter:
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.max_words_per_input: int | None = None
        self.embedding_model = OllamaEmbeddings(model=model_name, base_url=base_url)

    def _truncate_texts(self, texts: list[str], word_limit: int | None) -> list[str]:
        if word_limit is None:
            return texts

        return [_truncate_to_word_limit(text, word_limit) for text in texts]

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        original_texts = texts
        current_limit = self.max_words_per_input
        current_texts = self._truncate_texts(original_texts, current_limit)

        while True:
            try:
                embeddings = self.embedding_model.embed_documents(current_texts)
                if current_limit is not None:
                    self.max_words_per_input = current_limit
                return embeddings
            except Exception as exc:
                if not _is_ollama_context_error(exc):
                    raise

                longest_text = max((len(text.split()) for text in original_texts), default=0)
                base_limit = current_limit or longest_text
                next_limit = max(int(base_limit * OLLAMA_RETRY_SHRINK_FACTOR), MIN_OLLAMA_EMBED_WORDS)
                if next_limit >= base_limit:
                    raise

                current_limit = next_limit
                current_texts = self._truncate_texts(original_texts, current_limit)
                self.max_words_per_input = current_limit
                print(
                    f"Retrying Ollama embeddings for {self.model_name} with inputs capped at {current_limit} words "
                    "after a context-length error."
                )

    def embed_documents(self, documents: list) -> list:
        return self._embed_with_retry(_coerce_texts(documents))

    def embed_query(self, query: str) -> list:
        return self._embed_with_retry([query])[0]


def get_supported_embedders(provider: str = DEFAULT_PROVIDER) -> list[str]:
    match provider:
        case "huggingface":
            return list(HUGGINGFACE_NAME_MAP.keys())
        case "ollama":
            return list(OLLAMA_EMBEDDERS)
        case _:
            raise ValueError(f"Unknown provider {provider}")


def get_chunking_directory(provider: str) -> str:
    try:
        return CHUNKING_DIRECTORY_BY_PROVIDER[provider]
    except KeyError as exc:
        raise ValueError(f"Unknown provider {provider}") from exc


def get_embeddings_path(base_path: Path, provider: str, embedder: str, chunk_token_length: int, overlap: int) -> Path:
    return base_path / provider / embedder / get_chunking_directory(provider) / str(chunk_token_length) / str(overlap)


def build_embeddings(model_name: str, provider: str = DEFAULT_PROVIDER):
    match provider:
        case "huggingface":
            return HuggingFaceEmbeddingAdapter(model_name=HUGGINGFACE_NAME_MAP[model_name], trust_remote_code=True)
        case "ollama":
            return OllamaEmbeddingAdapter(model_name=model_name, base_url=f"http://{OLLAMA_URL}:{OLLAMA_PORT}")
        case _:
            raise ValueError(f"Unknown provider {provider}")


class ResultSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResultSingleton, cls).__new__(cls)
            cls._instance.results = []  # List of dictionaries with the results.

        return cls._instance

    def append(self, result: dict[str, object]) -> None:
        self.results.append(result)

    def save_results(self, name: str, path: Path = RESULT_PATH) -> None:
        # Convert to DataFrame
        # Every dictionary has the same keys
        df = pd.DataFrame(self.results)
        df.to_csv(path / f"{name}_results.csv", index=False)

    def check_if_results_exist(self, name: str) -> bool:
        return (RESULT_PATH / f"{name}_results.csv").exists()


def read_results_and_incumbents(columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
