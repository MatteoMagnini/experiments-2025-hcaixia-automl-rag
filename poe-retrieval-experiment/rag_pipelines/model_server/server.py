import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request, HTTPException
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from transformers import AutoTokenizer, AutoModel

app = FastAPI(title="Embedding + Reranking Model Server")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CACHE_DIR = os.getenv(
    "FASTEMBED_CACHE_PATH",
    os.getenv("FASTEMBED_CACHE", "/tmp/models")
)

GPU_PROVIDERS = ["CUDAExecutionProvider"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

# ---------------------------------------------------------------------------
# ENV-DRIVEN MODEL REGISTRY
# ---------------------------------------------------------------------------

EMBEDDING_MODELS = [
    m.strip()
    for m in os.getenv(
        "EMBEDDING_MODELS",
        "nomic-ai/nomic-embed-text-v2-moe,Qwen/Qwen3-Embedding-0.6B"
    ).split(",")
    if m.strip()
]

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "jinaai/jina-reranker-v2-base-multilingual"
)

# ---------------------------------------------------------------------------
# MODEL CACHE
# ---------------------------------------------------------------------------

loaded_models: dict = {}
models_ready = False


# ---------------------------------------------------------------------------
# HF FALLBACK
# ---------------------------------------------------------------------------

class HuggingFaceFallbackEmbedder:
    def __init__(self, model_name: str, cache_dir: str):
        print(f"[HF fallback] Loading {model_name} on {DEVICE}")

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        ).to(DEVICE)

        self.model.eval()

    def embed(self, texts: list[str], batch_size: int = BATCH_SIZE):
        vectors = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                output = self.model(**encoded)

            token_embeddings = output.last_hidden_state

            mask = encoded["attention_mask"].unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()

            pooled = torch.sum(token_embeddings * mask, 1) / torch.clamp(
                mask.sum(1),
                min=1e-9
            )

            normalized = F.normalize(pooled, p=2, dim=1)

            vectors.extend(normalized.cpu().numpy())

        return vectors


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def get_embedder(model_name: str):
    if model_name not in loaded_models:
        try:
            print(f"[fastembed] Loading {model_name}")

            loaded_models[model_name] = TextEmbedding(
                model_name=model_name,
                cache_dir=CACHE_DIR,
                providers=GPU_PROVIDERS
            )

        except Exception as fastembed_err:
            print(f"[fastembed FAILED] {fastembed_err}")
            print(f"[fallback → HF] {model_name}")

            loaded_models[model_name] = HuggingFaceFallbackEmbedder(
                model_name=model_name,
                cache_dir=CACHE_DIR
            )

    return loaded_models[model_name]


def get_reranker(model_name: str):
    if model_name not in loaded_models:
        print(f"[reranker] Loading {model_name}")

        try:
            loaded_models[model_name] = TextCrossEncoder(
                model_name=model_name,
                cache_dir=CACHE_DIR,
                providers=GPU_PROVIDERS
            )
        except Exception as e:
            raise RuntimeError(f"Reranker load failed: {e}")

    return loaded_models[model_name]


# ---------------------------------------------------------------------------
# STARTUP (CLEAN + ENV DRIVEN)
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    global models_ready

    print("\n================ MODEL SERVER START ================\n")

    print("Embedding models from ENV:")
    for m in EMBEDDING_MODELS:
        print(f"  - {m}")
        get_embedder(m)

    print(f"\nReranker model from ENV:")
    print(f"  - {RERANKER_MODEL}")
    get_reranker(RERANKER_MODEL)

    models_ready = True

    print("\n================ ALL MODELS READY ================\n")


# ---------------------------------------------------------------------------
# EMBEDDINGS ENDPOINT
# ---------------------------------------------------------------------------

@app.post("/embeddings")
async def embeddings(request: Request):
    data = await request.json()

    inputs = data.get("input", [])
    model_name = data.get("model")

    if isinstance(inputs, str):
        inputs = [inputs]

    if not inputs:
        return {"data": []}

    if model_name is None:
        raise HTTPException(
            status_code=400,
            detail="Missing 'model' field. Must specify embedding model."
        )

    try:
        model = get_embedder(model_name)
        vectors = list(model.embed(inputs, batch_size=BATCH_SIZE))

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "data": [
            {"embeddings": v.tolist(), "index": i}
            for i, v in enumerate(vectors)
        ],
        "model": model_name
    }


# ---------------------------------------------------------------------------
# RERANKING ENDPOINT
# ---------------------------------------------------------------------------

@app.post("/rerank")
async def rerank(request: Request):
    data = await request.json()

    query = data.get("query")
    docs = data.get("documents", [])

    try:
        reranker = get_reranker(RERANKER_MODEL)
        scores = list(reranker.rerank(query, docs))

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "results": [
            {"index": i, "score": float(s)}
            for i, s in enumerate(scores)
        ]
    }


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/ready")
def ready():
    if not models_ready:
        from fastapi import Response
        return Response(status_code=503, content="Models not loaded yet")

    return {"status": "ready", "device": DEVICE}


@app.get("/models")
def models():
    return {
        "loaded_models": list(loaded_models.keys()),
        "embedding_models_env": EMBEDDING_MODELS,
        "reranker": RERANKER_MODEL
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("MODEL_SERVER_PORT", "7997"))
    )


if __name__ == "__main__":
    main()