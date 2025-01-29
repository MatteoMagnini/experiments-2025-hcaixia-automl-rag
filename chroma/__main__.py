from time import time
import fire
import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chroma import PATH as CHROMA_PATH, DATABASE_NAME_FAQ, CHUNK_LOOKUP_FILE_NAME, CHUNK_DOCUMENTS_LOOKUP_FILE_NAME
from documents import PATH as DOCUMENT_PATH, LOOKUP_FILE_NAME
from utils import OLLAMA_URL, OLLAMA_PORT, HUGGINGFACE_NAME_MAP, HuggingFaceEmbeddingAdapter

LOOKUP_FILE = DOCUMENT_PATH / LOOKUP_FILE_NAME
FAQ_DATABASE = CHROMA_PATH / DATABASE_NAME_FAQ


def main(chunk_length: int, overlap_percentage: float, embedder: str, provider: str = "huggingface"):
    overlap = int(chunk_length * overlap_percentage)
    # Create the folder for the FAQ database if it does not exist
    FAQ_DATABASE.mkdir(exist_ok=True)
    # Create the subfolder for the embeddings
    faq_embedder_folder = FAQ_DATABASE / embedder
    faq_embedder_folder.mkdir(exist_ok=True)  # noqa
    faq_embedder_folder /= str(chunk_length)
    faq_embedder_folder.mkdir(exist_ok=True)  # noqa
    faq_embedder_folder /= str(overlap)
    faq_embedder_folder.mkdir(exist_ok=True)  # noqa
    # Embed every .txt file in the documents folder
    # Create a lookup table for the chunks to speed up operations
    # Create a lookup table for the chunks-documents to speed up operations
    # (each new chunk is associated with the original file id)
    document_lookup = pd.read_csv(LOOKUP_FILE, index_col="id")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_length, chunk_overlap=overlap)
    documents = [
        Document(page_content=open(DOCUMENT_PATH / file_name[0], "r").read(), metadata={"document_id": id})
        for id, file_name in document_lookup.iterrows()
    ]
    chunks = text_splitter.split_documents(documents)
    # Create the lookup table for the chunks-documents
    document_ids = [chunk.metadata["document_id"] for chunk in chunks]
    chunk_document_lookup = pd.DataFrame(document_ids)
    chunk_document_lookup.columns = ["document_id"]
    chunk_document_lookup.to_csv(faq_embedder_folder / CHUNK_DOCUMENTS_LOOKUP_FILE_NAME, index=True, index_label="chunk_id")
    # Create the lookup table for the chunks
    chunk_lookup = pd.DataFrame([chunk.page_content for chunk in chunks])
    chunk_lookup.columns = ["chunk"]
    chunk_lookup.to_csv(faq_embedder_folder / CHUNK_LOOKUP_FILE_NAME, index=True, index_label="id")
    # Embed the chunks
    match provider:
        case "huggingface":
            embeddings = HuggingFaceEmbeddingAdapter(model_name=HUGGINGFACE_NAME_MAP[embedder], trust_remote_code=True)
        case "ollama":
            embeddings = OllamaEmbeddings(model=embedder, base_url=f"http://{OLLAMA_URL}:{str(OLLAMA_PORT)}")
        case _:
            raise ValueError(f"Unknown provider {provider}")
    vectorstore = Chroma(persist_directory=str(faq_embedder_folder), embedding_function=embeddings)

    print(f'Starting to embed {len(chunks)} chunks with {embedder}')
    start_time = time()
    vectorstore.add_documents(chunks)
    end_time = time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time}")


if __name__ == "__main__":
    fire.Fire(main)