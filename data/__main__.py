from random import seed

import numpy as np
import pandas as pd
import fire
from data import PATH as DATASET_PATH, CHUNKS_QUESTIONS_FILE, TEST_FILE_NAME, TRAINING_FILE_NAME
from documents import PATH as DOCUMENT_PATH, LOOKUP_FILE_NAME


CHUNK_FILE = DATASET_PATH / CHUNKS_QUESTIONS_FILE
TRAIN_FILE = DATASET_PATH / TRAINING_FILE_NAME
TEST_FILE = DATASET_PATH / TEST_FILE_NAME
LOOKUP_FILE = DOCUMENT_PATH / LOOKUP_FILE_NAME


def main():
    # Load the chunk_file and create the training and test datasets
    # The training set has Domanda_1 (renamed in question) and File (renamed in file_name)
    # The training set has 5.000 rows randomly selected from the chunk_file
    # The test set has 1.000 rows randomly selected from the chunk_file
    # Add the id column from the lookup table
    seed(0)
    np.random.seed(0)
    chunk_df = pd.read_csv(CHUNK_FILE, sep="\t")
    lookup_df = pd.read_csv(LOOKUP_FILE)
    df = chunk_df[["Domanda_1", "File"]].copy()
    df.rename(columns={"Domanda_1": "question", "File": "file_name"}, inplace=True)
    df = df.merge(lookup_df, left_on="file_name", right_on="file_name")
    train_df = df.sample(n=5000, random_state=0)
    test_df = df.drop(train_df.index).sample(n=1000, random_state=0)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE_NAME, index=False)

if __name__ == "__main__":
    fire.Fire(main)