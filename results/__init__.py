from pathlib import Path
import pandas as pd
from smac import HyperparameterOptimizationFacade as HPOFacade


PATH = Path(__file__).parents[0]


def incumbents_to_dataframe(incumbents: list) -> pd.DataFrame:
    data = []
    for incumbent in incumbents:
        data.append(incumbent.get_dictionary())

    return pd.DataFrame(data)


def save_incumbents(smac: HPOFacade, incumbents: list, filename: str):
    df = []
    # Dynamically extract objectives from the SMAC scenario if available
    if hasattr(smac, "scenario") and getattr(smac.scenario, "objectives", None) is not None:
        costs_names = list(smac.scenario.objectives)
    else:
        costs_names = ["1 - accuracy", "number of documents"]
        
    for incumbent in incumbents:
        config = incumbent.get_dictionary()
        costs = {k: v for k, v in zip(costs_names, smac.runhistory.average_cost(incumbent))}
        config.update(costs)
        df.append(config)
    df = pd.DataFrame(df)
    df.to_csv(PATH / filename, index=False)
