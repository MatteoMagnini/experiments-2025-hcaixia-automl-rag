import fire
import pandas as pd
from documents import PATH as DOCUMENT_PATH, LOOKUP_FILE_NAME


LOOKUP_FILE = DOCUMENT_PATH / LOOKUP_FILE_NAME


def main():
    # Create a lookup table for the documents to speed up operations
    # Consider all .txt files in the documents folder
    lookup = []
    for file in DOCUMENT_PATH.glob("*.txt"):
        with open(file, "r") as _:
            lookup.append(file.name)
    lookup_df = pd.DataFrame(sorted(lookup), columns=["file_name"])
    lookup_df.to_csv(LOOKUP_FILE, index=True, index_label="id")


if __name__ == "__main__":
    fire.Fire(main)