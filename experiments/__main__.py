import fire
from smac import Scenario
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer, Categorical
)
from smac.multi_objective.parego import ParEGO
from smac import HyperparameterOptimizationFacade as HPOFacade
from results import save_incumbents
from utils import ResultSingleton, plot_pareto


def run_experiment(config: Configuration, seed: int = 0, budget: int = 100) -> dict[str, float]:
    pass


def main():
    cs = ConfigurationSpace()
    chunk_length = Integer("chunk_length", (200, 1000), default=200)
    overlap_percentage = Float("overlap_percentage", (0.1, 0.5), default=0.1)
    retriever = Categorical("retriever", ["base", "BM25"], default="base")
    embedder = Categorical("embedder", ["mxbai-embed-large", "nomic-embed-text"], default="base")
    num_docs = Integer("num_docs", (1, 20), default=1)
    cs.add([chunk_length, overlap_percentage, retriever, embedder, num_docs])
    objectives = ["1 - accuracy", "number of documents"]

    scenario = Scenario(
        cs,
        objectives=objectives,
        walltime_limit=12 * 60 * 60,  # After 12 hour, we stop the hyperparameter optimization
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

    # Let's plot a pareto front
    plot_pareto(smac, incumbents)


if __name__ == "__main__":
    fire.Fire(main)