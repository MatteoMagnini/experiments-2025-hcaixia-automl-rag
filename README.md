## Embeddings Provider

The project now defaults to Ollama for embedding generation and retrieval experiments.

- Default provider: `ollama`
- Default Ollama embedders: `nomic-embed-text`, `mxbai-embed-large`
- Hugging Face remains available as an explicit fallback provider

The embedding cache is provider-specific. 
Ollama and Hugging Face runs write to different directories under `chroma/faq/`, so existing Hugging Face caches are not reused by Ollama runs.

## Setup

Install dependencies:

```bash
uv sync
```

Ensure Ollama is reachable at the host configured in `utils/__init__.py` and that the embedding models are available there. Example pulls for a local Ollama instance:

```bash
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
```

## Usage

Run experiments with the default Ollama provider:

```bash
PYTHONPATH="." uv run experiments/__main__.py
```

Run experiments with the Hugging Face fallback provider:

```bash
PYTHONPATH="." uv run experiments/__main__.py --provider huggingface
```

Create chunked embeddings directly with Ollama:

```bash
PYTHONPATH="." uv run chroma/__main__.py --chunk_token_length 100 --overlap_percentage 0.1 --embedder nomic-embed-text --provider ollama
```

Run the standalone chunking benchmark to estimate splitter overhead on the local document corpus:

```bash
PYTHONPATH="." uv run experiments/chunking_benchmark.py --document_limit 10 --repeat 3 --chunk_token_length 256 --include_huggingface False
```

The raw timing results are saved to `results/chunking_benchmark.csv`.
