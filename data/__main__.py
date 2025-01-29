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
    # The test set has Domanda_2 (renamed in question) and File (renamed in file_name)
    # Add the id column from the lookup table
    chunk_df = pd.read_csv(CHUNK_FILE, sep="\t")
    lookup_df = pd.read_csv(LOOKUP_FILE)
    train_df = chunk_df[["Domanda_1", "File"]].copy()
    train_df.rename(columns={"Domanda_1": "question", "File": "file_name"}, inplace=True)
    train_df = train_df.merge(lookup_df, left_on="file_name", right_on="file_name")
    test_df = chunk_df[["Domanda_2", "File"]].copy()
    test_df.rename(columns={"Domanda_2": "question", "File": "file_name"}, inplace=True)
    test_df = test_df.merge(lookup_df, left_on="file_name", right_on="file_name")
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE_NAME, index=False)

if __name__ == "__main__":
    fire.Fire(main)