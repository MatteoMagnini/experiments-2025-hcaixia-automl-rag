import os
from figures import plot_pareto_raw, PATH as FIGURE_PATH
from utils import read_results_and_incumbents


if __name__ == "__main__":
    results, incumbents = read_results_and_incumbents(columns=["1 - accuracy", "number of documents"])
    # results = results.sort_values(by="1 - accuracy")
    incumbents = incumbents.sort_values(by="1 - accuracy")
    plot_pareto_raw(
        costs=results,
        pareto_costs=incumbents,
        file_path=[os.path.join(FIGURE_PATH, "result.eps"), os.path.join(FIGURE_PATH, "result.png")],
        obj0='1 - accuracy',
        obj1='number of documents')