from random import seed
from time import time
from langchain_core.documents import Document
import fire
import pandas as pd
from smac import Scenario
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer, Categorical
)
from langchain_ollama import OllamaEmbeddings
from smac.multi_objective.parego import ParEGO
from smac import HyperparameterOptimizationFacade as HPOFacade
from data import PATH as DATA_PATH, TRAINING_FILE_NAME
from experiments import EMBEDDERS
from results import save_incumbents
from utils import ResultSingleton, OLLAMA_URL, OLLAMA_PORT, HuggingFaceEmbeddingAdapter, \
    HUGGINGFACE_NAME_MAP
from langchain_chroma import Chroma
from chroma import PATH as CHROMA_PATH, DATABASE_NAME_FAQ
from chroma.__main__ import main as create_embeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from results.cache import PATH as CACHE_PATH


TRAINING_FILE = DATA_PATH / TRAINING_FILE_NAME


def run_experiment(config: Configuration, seed: int = 0, budget: int = 100, provider: str = "huggingface") -> dict[str, float]:
    singleton = ResultSingleton()
    # Check if the embeddings are already available
    print(f"Running experiment with config: {config}")
    overlap = int(config["chunk_length"] * config["overlap_percentage"])
    embeddings_path = CHROMA_PATH / DATABASE_NAME_FAQ
    embeddings_path /= config["embedder"]
    embeddings_path /= str(config["chunk_length"])
    embeddings_path /= str(overlap)
    if embeddings_path.exists():
        print(f"Skipping embeddings creation for {config['embedder']} with chunk length {config['chunk_length']} and overlap {overlap}")
    else:
        create_embeddings(config["chunk_length"], config["overlap_percentage"], config["embedder"])

    # Retrieve the embeddings for the training set
    training_questions = pd.read_csv(TRAINING_FILE)
    match provider:
        case "huggingface":
            embeddings = HuggingFaceEmbeddingAdapter(model_name=HUGGINGFACE_NAME_MAP[config["embedder"]], trust_remote_code=True)
        case "ollama":
            embeddings = OllamaEmbeddings(model=config["embedder"], base_url=f"http://{OLLAMA_URL}:{str(OLLAMA_PORT)}")
        case _:
            raise ValueError(f"Unknown provider {provider}")
    vectorstore = Chroma(persist_directory=str(embeddings_path), embedding_function=embeddings)

    ## Retriever part
    number_of_docs = config["num_docs"]
    retriever_type = config["retriever"]

    def get_bm25():
        chunks = pd.read_csv(embeddings_path / "chunk_lookup.csv")
        chunks = [Document(chunk) for chunk in chunks["chunk"]]
        r = BM25Retriever.from_documents(chunks)
        r.k = number_of_docs
        return r

    match retriever_type:
        case "base":
            retriever = vectorstore.as_retriever(search_kwargs={'k': number_of_docs})
        case "ensemble":
            base = vectorstore.as_retriever(search_kwargs={'k': number_of_docs})
            bm25 = get_bm25()
            retriever = EnsembleRetriever(retrievers=[base, bm25], weights=[0.5, 0.5])
        case _:
            raise ValueError(f"Unknown retriever type {retriever_type}")

    ## Retrieve
    results: list[list] = []
    print("Retrieving...")
    batch_size = 128
    questions = training_questions["question"]
    start_time = time()
    for j in range(0, len(questions), batch_size):
        print(f"Retrieving batch {j} to {j + batch_size}")
        batch_questions = list(questions[j:j + batch_size])
        batch_results = retriever.batch(batch_questions)
        results.extend(batch_results)
    end_time = time()
    retrieval_time = end_time - start_time
    print(f"Retrieval time: {retrieval_time}")

    ## Evaluate
    ## Each question has been generated from a document (see the id column)
    ## Each chunk comes from a document (see the document_id column in the lookup table)
    ## If at least one chunk from the retrieved documents is from the same document as the question, the retrieval is considered correct
    correct_retrievals = 0
    chunk_lookup = pd.read_csv(embeddings_path / "chunk_lookup.csv")
    chunk_document_lookup = pd.read_csv(embeddings_path / "chunk_documents_lookup.csv")
    # Lookup to dictionary
    chunk_lookup = {row["chunk"]: row["id"] for i, row in chunk_lookup.iterrows()}
    chunk_document_lookup = {row["chunk_id"]: row["document_id"] for i, row in chunk_document_lookup.iterrows()}
    for i, question in training_questions.iterrows():
        question_document_id = question["id"]
        retrieved_documents = [chunk_document_lookup[chunk_lookup[result.page_content]] for result in results[i]]
        if question_document_id in retrieved_documents:
            correct_retrievals += 1
    accuracy = correct_retrievals / len(training_questions)
    print(f"Accuracy: {accuracy}")
    # Save to the result cache
    result = {
        "1 - accuracy": 1 - accuracy,
        "number of documents": number_of_docs
    }
    configuration = {
        "chunk_length": config["chunk_length"],
        "overlap_percentage": config["overlap_percentage"],
        "retriever": retriever_type,
        "embedder": config["embedder"],
    }
    conf_and_result = configuration | result
    singleton.append(conf_and_result)
    df_result = pd.DataFrame([conf_and_result])
    df_result.to_csv(CACHE_PATH / "results.csv", mode="a", header=False, index=False)
    return {
        "1 - accuracy": 1 - accuracy,
        "number of documents": number_of_docs
    }



def main():
    seed(0)
    cs = ConfigurationSpace()
    chunk_length = Integer("chunk_length", (100, 500), default=100)
    overlap_percentage = Float("overlap_percentage", (0.1, 0.5), default=0.1)
    retriever = Categorical("retriever", ["base", "ensemble"], default="base")
    embedder = Categorical("embedder", EMBEDDERS, default="nomic-embed-text")
    num_docs = Integer("num_docs", (1, 20), default=1)
    cs.add([chunk_length, overlap_percentage, retriever, embedder, num_docs])
    objectives = ["1 - accuracy", "number of documents"]
    # Clear the cache
    open(CACHE_PATH / "results.csv", "w").close()

    scenario = Scenario(
        cs,
        objectives=objectives,
        walltime_limit=24 * 60 * 60,  # After 24 hours, we stop the hyperparameter optimization
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
        run_experiment,
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
