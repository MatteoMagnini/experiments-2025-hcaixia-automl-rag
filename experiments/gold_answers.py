"""Loading of the pregenerated gold-standard answers.

The gold answers are produced offline (see data/generate_gold_standards.py) and
stored in the train/test CSVs; here we just read them back keyed by question.
"""
import pandas as pd

from data import PATH as DATA_PATH


def load_pregenerated_gold_answers_mapping() -> dict[str, str]:
    mapping = {}
    for filename in ["train.csv", "test.csv"]:
        path = DATA_PATH / filename
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "gold_standard_answer" in df.columns:
                    for _, row in df.iterrows():
                        q = row.get("question")
                        ans = row.get("gold_standard_answer")
                        if pd.notna(q) and pd.notna(ans):
                            mapping[str(q).strip()] = str(ans).strip()
            except Exception as e:
                print(f"Error loading pregenerated gold answers from {filename}: {e}")
    return mapping
