from time import time
import fire
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chroma import PATH as CHROMA_PATH, DATABASE_NAME_FAQ, CHUNK_LOOKUP_FILE_NAME, CHUNK_DOCUMENTS_LOOKUP_FILE_NAME
from documents import PATH as DOCUMENT_PATH, LOOKUP_FILE_NAME
from utils import DEFAULT_PROVIDER, HUGGINGFACE_NAME_MAP, build_embeddings, get_embeddings_path
from transformers import AutoTokenizer
import unicodedata
from pathlib import Path


def resolve_existing_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    for normalization in ("NFC", "NFD"):
        normalized = Path(unicodedata.normalize(normalization, str(candidate)))
        if normalized.exists():
            return normalized

    return candidate


LOOKUP_FILE = DOCUMENT_PATH / LOOKUP_FILE_NAME
FAQ_DATABASE = CHROMA_PATH / DATABASE_NAME_FAQ


def word_count(text: str) -> int:
    return len(text.split())


def build_text_splitter(provider: str, embedder: str, chunk_token_length: int, overlap: int) -> RecursiveCharacterTextSplitter:
    if provider == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_NAME_MAP[embedder], trust_remote_code=True)
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_token_length,
            chunk_overlap=overlap,
        )

    # Ollama does not expose a Hugging Face-compatible tokenizer API.
    # For the default path, count words and prefer sentence boundaries so
    # chunks stay readable and are less likely to split a sentence mid-way.
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        keep_separator="end",
        chunk_size=chunk_token_length,
        chunk_overlap=overlap,
        length_function=word_count,
    )


def main(chunk_token_length: int, overlap_percentage: float, embedder: str, provider: str = DEFAULT_PROVIDER):
    overlap = int(chunk_token_length * overlap_percentage)
    faq_embedder_folder = get_embeddings_path(FAQ_DATABASE, provider, embedder, chunk_token_length, overlap)
    faq_embedder_folder.mkdir(parents=True, exist_ok=True)
    # Embed every .txt file in the documents folder
    # Create a lookup table for the chunks to speed up operations
    # Create a lookup table for the chunks-documents to speed up operations
    # (each new chunk is associated with the original file id)
    document_lookup = pd.read_csv(LOOKUP_FILE, index_col="id")
    text_splitter = build_text_splitter(provider, embedder, chunk_token_length, overlap)
    documents = [
        Document(page_content=resolve_existing_path(DOCUMENT_PATH / file_name["file_name"]).read_text(), metadata={"document_id": id})
        for id, file_name in document_lookup.iterrows()
    ]
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents with chunk token length {chunk_token_length} and overlap {overlap}")
    # chunk example 
    print(f"Chunk 0 has {word_count(chunks[0].page_content)} words")
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
    embeddings = build_embeddings(embedder, provider)
    vectorstore = Chroma(persist_directory=str(faq_embedder_folder), embedding_function=embeddings)

    print(f'Starting to embed {len(chunks)} chunks with {embedder}')
    start_time = time()

    batch_size = 1024
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
    end_time = time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time}")


if __name__ == "__main__":
    fire.Fire(main)
