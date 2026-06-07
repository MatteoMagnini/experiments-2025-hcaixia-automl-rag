"""Entrypoint: build the SMAC scenario/config space and run the optimization.

The heavy lifting lives in focused modules:
  - constants            : SMAC objectives + eval defaults
  - runner.run_experiment: evaluate one configuration end-to-end
  - retrievers / accuracy / evaluation / metrics / generation : the trial steps
  - results_recording    : shape + persist each trial's result row
"""
import os
from functools import partial
from random import seed
import fire
import pandas as pd
from dotenv import load_dotenv
from experiments import (
    DEFAULT_WALLTIME_LIMIT,
    MIN_CHUNK_TOKEN_SIZE,
    MAX_CHUNK_TOKEN_SIZE,
    MIN_OVERLAP_PERCENTAGE,
    MAX_OVERLAP_PERCENTAGE
)

load_dotenv()

from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    Integer,
)
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.multi_objective.parego import ParEGO

from results import save_incumbents
from utils import DEFAULT_PROVIDER, ResultSingleton, get_supported_embedders

from experiments.constants import DEFAULT_EVAL_SAMPLE_SIZE, SMAC_OBJECTIVES
from experiments.results_recording import reset_results_cache
from experiments.runner import TRAINING_FILE, run_experiment


def main(
        provider: str = DEFAULT_PROVIDER,
        embedders: str | list[str] = None,
        walltime_limit: int = DEFAULT_WALLTIME_LIMIT,
        gen_model: str = None,
        eval_sample_size: int = DEFAULT_EVAL_SAMPLE_SIZE
):
    seed(0)
    cs = ConfigurationSpace()
    chunk_token_length = Integer(
        "chunk_token_length",
        (MIN_CHUNK_TOKEN_SIZE, MAX_CHUNK_TOKEN_SIZE),
        default=MIN_CHUNK_TOKEN_SIZE
    )
    overlap_percentage = Float(
        "overlap_percentage",
        (MIN_OVERLAP_PERCENTAGE, MAX_OVERLAP_PERCENTAGE),
        default=MIN_OVERLAP_PERCENTAGE
    )
    retriever_choices = ["base", "ensemble", "bm25_only", "mmr"]
    retriever = Categorical("retriever", retriever_choices, default="base")

    if embedders:
        if isinstance(embedders, str):
            embedder_list = [e.strip() for e in embedders.split(",") if e.strip()]
        else:
            embedder_list = list(embedders)
    else:
        embedder_list = get_supported_embedders(provider)
        print(f"List of supported embedders:\n {'\n'.join(embedder_list)}")

    if not embedder_list:
        raise ValueError("No embedder models specified or found.")

    embedder = Categorical("embedder", embedder_list, default=embedder_list[0])

    if gen_model:
        if isinstance(gen_model, str):
            gen_model_list = [g.strip() for g in gen_model.split(",") if g.strip()]
        else:
            gen_model_list = list(gen_model)
    else:
        if os.environ.get("OPENROUTER_API_KEY"):
            # Two model families at increasing sizes, so SMAC can explore the
            # effect of generator size on answer quality within each family.
            gen_model_list = [
                # Gemma 3 family
                "google/gemma-3-4b-it",
                "google/gemma-3-12b-it",
                "google/gemma-3-27b-it",
                # Qwen3 family (dense instruct)
                "qwen/qwen3-8b",
                "qwen/qwen3-14b",
                "qwen/qwen3-32b",
            ]
        else:
            gen_model_list = ["gemini-1.5-flash"]

    gen_model_param = Categorical("gen_model", gen_model_list, default=gen_model_list[0])

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
        gen_model_param,
    ])
    cs.add(EqualsCondition(mmr_fetch_k, retriever, "mmr"))
    cs.add(EqualsCondition(mmr_lambda_mult, retriever, "mmr"))

    objectives = list(SMAC_OBJECTIVES)

    # Clear the cache from previous runs
    reset_results_cache()

    scenario = Scenario(
        cs,
        objectives=objectives,
        walltime_limit=walltime_limit,
        n_trials=10000,  # Evaluate max 10^4 different trials
        n_workers=1  # multiprocessing.cpu_count()
    )

    # Load gold standard answers for the (same) evaluation sample used in run_experiment.
    pregenerated_gold_answers = None
    if eval_sample_size > 0:
        try:
            training_questions = pd.read_csv(TRAINING_FILE).reset_index(drop=True)
            n_eval = min(eval_sample_size, len(training_questions))
            eval_questions = training_questions.sample(n=n_eval, random_state=42)

            if "gold_standard_answer" in eval_questions.columns:
                print("Loading pregenerated gold standard answers from TRAINING_FILE column...")
                pregenerated_gold_answers = eval_questions["gold_standard_answer"].tolist()
                print("Pregeneration of gold standard answers complete (loaded from CSV).")
        except Exception as e:
            print(f"Warning: Failed to pregenerate gold standard answers: {e}")

    # We want to run five random configurations before starting the optimization.
    initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)
    multi_objective_algorithm = ParEGO(scenario)
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)

    # Create our SMAC object and pass the scenario and the train method
    smac = HPOFacade(
        scenario,
        partial(
            run_experiment,
            provider=provider,
            gen_model=gen_model,
            pregenerated_gold_answers=pregenerated_gold_answers,
            eval_sample_size=eval_sample_size,
        ),
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
