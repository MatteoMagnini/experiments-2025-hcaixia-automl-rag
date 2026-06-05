from time import time
import os
import shutil
import fire
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant import PATH as QDRANT_PATH, DATABASE_NAME_FAQ, CHUNK_LOOKUP_FILE_NAME, CHUNK_DOCUMENTS_LOOKUP_FILE_NAME
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
FAQ_DATABASE = QDRANT_PATH / DATABASE_NAME_FAQ


def word_count(text: str) -> int:
    return len(text.split())


def build_text_splitter(provider: str, embedder: str, chunk_token_length: int, overlap: int) -> RecursiveCharacterTextSplitter:
    if provider == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_NAME_MAP[embedder], trust_remote_code=True)
        tokenizer.model_max_length = int(1e9)
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_token_length,
            chunk_overlap=overlap,
        )

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
    shutil.rmtree(faq_embedder_folder, ignore_errors=True)
    faq_embedder_folder.mkdir(parents=True, exist_ok=True)
    # Embed every .txt file in the documents folder
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

    print(f'Starting to embed {len(chunks)} chunks with {embedder}', flush=True)
    start_time = time()

    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    batch_size = 1024
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    def log_embed_progress(done: int, batch_index: int) -> None:
        elapsed = time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total_chunks - done) / rate if rate > 0 else 0
        print(
            f"  embedded {done}/{total_chunks} chunks "
            f"(batch {batch_index}/{total_batches}) "
            f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s",
            flush=True,
        )

    first_batch = chunks[:batch_size]

    if qdrant_url:
        vectorstore = QdrantVectorStore.from_documents(
            first_batch,
            embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=DATABASE_NAME_FAQ,
        )
    else:
        vectorstore = QdrantVectorStore.from_documents(
            first_batch,
            embeddings,
            path=str(faq_embedder_folder),
            collection_name=DATABASE_NAME_FAQ,
        )
    log_embed_progress(min(batch_size, total_chunks), 1)

    for i in range(batch_size, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        log_embed_progress(min(i + batch_size, total_chunks), i // batch_size + 1)
    end_time = time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time}")

    # Release the local Qdrant storage lock so the caller can reopen the same
    # path (otherwise qdrant-client raises "already accessed by another instance").
    if not qdrant_url:
        try:
            vectorstore.client.close()
        except Exception:
            pass


if __name__ == "__main__":
    fire.Fire(main)
