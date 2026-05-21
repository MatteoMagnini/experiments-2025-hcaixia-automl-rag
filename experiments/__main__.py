from random import seed
from time import time
from functools import partial
from pathlib import Path
import shutil
import subprocess
import traceback
from langchain_core.documents import Document
import fire
import pandas as pd
from chromadb.errors import InternalError
from smac import Scenario
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Categorical,
    EqualsCondition,
)
from smac.multi_objective.parego import ParEGO
from smac import HyperparameterOptimizationFacade as HPOFacade
from data import PATH as DATA_PATH, TRAINING_FILE_NAME
from results import save_incumbents
from utils import (
    DEFAULT_PROVIDER,
    OLLAMA_PORT,
    OLLAMA_URL,
    ResultSingleton,
    build_embeddings,
    get_embeddings_path,
    get_supported_embedders,
)
from langchain_chroma import Chroma
from chroma import PATH as CHROMA_PATH, DATABASE_NAME_FAQ, CHUNK_LOOKUP_FILE_NAME, CHUNK_DOCUMENTS_LOOKUP_FILE_NAME
from chroma.__main__ import main as create_embeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import ChatOllama
from results.cache import PATH as CACHE_PATH


TRAINING_FILE = DATA_PATH / TRAINING_FILE_NAME
MULTI_QUERY_MODELS = [
    "qwen3.5:4b",
    "qwen3.5:2b",
    "qwen3.5:0.8b",
    "gemma4:e2b",
    "gemma4",
    "medgemma:latest",
    "lfm2.5-thinking:latest",
]
FAILURE_ACCURACY_COST = 1.0


def load_chunk_lookup_mappings(embeddings_path: Path) -> tuple[dict[str, int], dict[int, int]]:
    chunk_lookup = pd.read_csv(embeddings_path / CHUNK_LOOKUP_FILE_NAME)
    chunk_document_lookup = pd.read_csv(embeddings_path / CHUNK_DOCUMENTS_LOOKUP_FILE_NAME)
    return (
        {row["chunk"]: int(row["id"]) for _, row in chunk_lookup.iterrows()},
        {int(row["chunk_id"]): int(row["document_id"]) for _, row in chunk_document_lookup.iterrows()},
    )


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
    retriever = BM25Retriever.from_documents(build_bm25_documents(embeddings_path))
    retriever.k = number_of_docs
    return retriever


def build_vector_retriever(vectorstore: Chroma, number_of_docs: int):
    return vectorstore.as_retriever(search_kwargs={"k": number_of_docs})


def build_mmr_retriever(
    vectorstore: Chroma,
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


def get_available_ollama_models() -> set[str]:
    try:
        output = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except FileNotFoundError as exc:
        raise RuntimeError("Ollama CLI is required for the multi_query retriever.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Unable to inspect installed Ollama models for the multi_query retriever.") from exc

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    model_lines = lines[1:] if lines and lines[0].lower().startswith("name") else lines
    return {line.split()[0] for line in model_lines}


def ensure_ollama_model_available(model_name: str) -> None:
    available_models = get_available_ollama_models()
    if model_name not in available_models:
        available_models_display = ", ".join(sorted(available_models)) or "none"
        raise ValueError(
            f"The multi_query model '{model_name}' is not installed in Ollama. "
            f"Available models: {available_models_display}."
        )


def build_multi_query_retriever(vectorstore: Chroma, number_of_docs: int, model_name: str):
    ensure_ollama_model_available(model_name)
    llm = ChatOllama(
        model=model_name,
        base_url=f"http://{OLLAMA_URL}:{OLLAMA_PORT}",
        temperature=0,
    )
    base_retriever = build_vector_retriever(vectorstore, number_of_docs)
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        include_original=True,
    )


def get_supported_multi_query_models() -> list[str]:
    try:
        available_models = get_available_ollama_models()
    except RuntimeError as exc:
        print(f"Disabling multi_query retriever: {exc}")
        return []

    supported_models = [model for model in MULTI_QUERY_MODELS if model in available_models]
    missing_models = [model for model in MULTI_QUERY_MODELS if model not in available_models]
    if missing_models:
        print(f"Skipping unavailable multi_query models: {', '.join(missing_models)}")

    return supported_models


def build_retriever(config: Configuration, vectorstore: Chroma, embeddings_path: Path):
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
        case "multi_query":
            return build_multi_query_retriever(
                vectorstore,
                number_of_docs,
                model_name=str(config["query_llm_model"]),
            )
        case _:
            raise ValueError(f"Unknown retriever type {retriever_type}")


def trim_batch_results(batch_results: list[list[Document]], limit: int) -> list[list[Document]]:
    return [list(result)[:limit] for result in batch_results]


def normalize_optional_config_value(value):
    return value if value is not None else ""


def build_result_payload(config: Configuration, number_of_docs: int, accuracy_cost: float) -> dict[str, float]:
    return {
        "1 - accuracy": float(accuracy_cost),
        "number of documents": float(number_of_docs),
    }


def build_configuration_payload(config: Configuration) -> dict[str, object]:
    return {
        "chunk_token_length": int(config["chunk_token_length"]),
        "overlap_percentage": float(config["overlap_percentage"]),
        "retriever": str(config["retriever"]),
        "embedder": str(config["embedder"]),
        "mmr_fetch_k": normalize_optional_config_value(config.get("mmr_fetch_k")),
        "mmr_lambda_mult": normalize_optional_config_value(config.get("mmr_lambda_mult")),
        "query_llm_model": normalize_optional_config_value(config.get("query_llm_model")),
    }


def cache_experiment_row(row: dict[str, object]) -> None:
    ResultSingleton().append(row)
    pd.DataFrame([row]).to_csv(CACHE_PATH / "results.csv", mode="a", header=False, index=False)


def is_recoverable_chroma_error(exc: Exception) -> bool:
    return isinstance(exc, InternalError) and "metadata segment" in str(exc).lower()


def rebuild_embeddings_cache(embeddings_path: Path, chunk_token_length: int, overlap_percentage: float, embedder: str, provider: str) -> None:
    if embeddings_path.exists():
        shutil.rmtree(embeddings_path, ignore_errors=True)
    create_embeddings(chunk_token_length, overlap_percentage, embedder, provider=provider)


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


def run_experiment(config: Configuration, seed: int = 0, budget: int = 100, provider: str = DEFAULT_PROVIDER) -> dict[str, float]:
    number_of_docs = int(config["num_docs"])
    configuration = build_configuration_payload(config)
    chunk_token_length = int(config["chunk_token_length"])
    overlap_percentage = float(config["overlap_percentage"])
    embedder = str(config["embedder"])
    overlap = int(chunk_token_length * overlap_percentage)
    embeddings_path = get_embeddings_path(
        CHROMA_PATH / DATABASE_NAME_FAQ,
        provider,
        embedder,
        chunk_token_length,
        overlap,
    )

    for attempt in range(2):
        # Check if the embeddings are already available
        try:
            print(f"Running experiment with config: {config}")
            if embeddings_cache_ready(embeddings_path):
                print(f"Skipping embeddings creation for {config['embedder']} with chunk token length {chunk_token_length} and overlap {overlap}")
            else:
                create_embeddings(chunk_token_length, overlap_percentage, embedder, provider=provider)

            # Retrieve the embeddings for the training set
            training_questions = pd.read_csv(TRAINING_FILE)
            embeddings = build_embeddings(embedder, provider)
            vectorstore = Chroma(persist_directory=str(embeddings_path), embedding_function=embeddings)

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
            ## Each question has been generated from a document (see the id column)
            ## Each chunk comes from a document (see the document_id column in the lookup table)
            ## If at least one chunk from the retrieved documents is from the same document as the question, the retrieval is considered correct
            chunk_lookup, chunk_document_lookup = load_chunk_lookup_mappings(embeddings_path)
            accuracy = compute_accuracy(training_questions, results, chunk_lookup, chunk_document_lookup)
            print(f"Accuracy: {accuracy}")
            result = build_result_payload(config, number_of_docs, 1 - accuracy)
            cache_experiment_row(configuration | result)
            return result
        except Exception as exc:
            if attempt == 0 and is_recoverable_chroma_error(exc):
                print(f"Recovering corrupted Chroma cache at {embeddings_path}: {exc}")
                rebuild_embeddings_cache(embeddings_path, chunk_token_length, overlap_percentage, embedder, provider)
                continue

            print(f"Experiment failed for config: {config}")
            print(f"{type(exc).__name__}: {exc}")
            print(traceback.format_exc())
            failure_result = build_result_payload(config, number_of_docs, FAILURE_ACCURACY_COST)
            cache_experiment_row(configuration | failure_result | {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            return failure_result



def main(provider: str = DEFAULT_PROVIDER):
    seed(0)
    cs = ConfigurationSpace()
    chunk_token_length = Integer("chunk_token_length", (100, 500), default=100)
    overlap_percentage = Float("overlap_percentage", (0.1, 0.5), default=0.1)
    query_llm_models = get_supported_multi_query_models()
    retriever_choices = ["base", "ensemble", "bm25_only", "mmr"]
    if query_llm_models:
        retriever_choices.append("multi_query")
    retriever = Categorical("retriever", retriever_choices, default="base")
    embedders = get_supported_embedders(provider)
    embedder = Categorical("embedder", embedders, default=embedders[0])
    num_docs = Integer("num_docs", (1, 20), default=1)
    mmr_fetch_k = Integer("mmr_fetch_k", (20, 100), default=40)
    mmr_lambda_mult = Float("mmr_lambda_mult", (0.1, 0.9), default=0.5)
    cs.add([
        chunk_token_length,
        overlap_percentage,
        retriever,
        embedder,
        num_docs,
        mmr_fetch_k,
        mmr_lambda_mult,
    ])
    cs.add(EqualsCondition(mmr_fetch_k, retriever, "mmr"))
    cs.add(EqualsCondition(mmr_lambda_mult, retriever, "mmr"))
    if query_llm_models:
        query_llm_model = Categorical("query_llm_model", query_llm_models, default=query_llm_models[0])
        cs.add([query_llm_model])
        cs.add(EqualsCondition(query_llm_model, retriever, "multi_query"))
    objectives = ["1 - accuracy", "number of documents"]
    # Clear the cache
    open(CACHE_PATH / "results.csv", "w").close()

    scenario = Scenario(
        cs,
        objectives=objectives,
        #walltime_limit=24 * 60 * 60,  # After 24 hours, we stop the hyperparameter optimization
        walltime_limit=60 * 60 * 1,  # After 6 hours, we stop the hyperparameter optimization
        n_trials=10000,  # Evaluate max 10^4 different trials
        n_workers=1  # multiprocessing.cpu_count()
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)
    multi_objective_algorithm = ParEGO(scenario)
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)

    # Create our SMAC object and pass the scenario and the train method
    smac = HPOFacade(
        scenario,
        partial(run_experiment, provider=provider),
        initial_design=initial_design,
        multi_objective_algorithm=multi_objective_algorithm,
        intensifier=intensifier,
        overwrite=True,
    )

    # Let's optimize
    incumbents = smac.optimize()

    # Get cost of default configuration
    # default_cost = smac.validate(mlp.configspace.get_default_configuration())
    # print(f"Validated costs from default config: \n--- {default_cost}\n")

    print("Validated costs from the Pareto front (incumbents):")
    for incumbent in incumbents:
        cost = smac.runhistory.average_cost(incumbent)
        print("---", cost)
    save_incumbents(smac, incumbents, "incumbents.csv")
    ResultSingleton().save_results("incumbents.csv")


if __name__ == "__main__":
    fire.Fire(main)
