"""Answer-quality metrics: ROUGE, BERTScore and embedding similarity.

Every metric is defensive: an unavailable/failed computation returns None
(skipped) rather than a fake 0.0, so it is never confused with a genuinely poor
score downstream.
"""
import os


def compute_rouge_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        total_rouge1 = 0.0
        total_rouge2 = 0.0
        total_rougeL = 0.0

        count = len(predictions)
        if count == 0:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            total_rouge1 += scores['rouge1'].fmeasure
            total_rouge2 += scores['rouge2'].fmeasure
            total_rougeL += scores['rougeL'].fmeasure

        return {
            "rouge1": total_rouge1 / count,
            "rouge2": total_rouge2 / count,
            "rougeL": total_rougeL / count,
        }
    except Exception as e:
        print(f"Error computing ROUGE scores: {e}")
        return {"rouge1": None, "rouge2": None, "rougeL": None}


_SKIPPED_BERT_SCORES = {
    "bertscore_precision": None,
    "bertscore_recall": None,
    "bertscore_f1": None,
}

# A BERTScorer is expensive to build (it downloads/loads a multilingual BERT
# model). Build it once and reuse it across every call/trial instead of
# reloading the model each time, which used to dominate the runtime.
_bert_scorer = None


def bertscore_enabled() -> bool:
    return os.environ.get("ENABLE_BERTSCORE", "1") not in ("0", "false", "False", "")


def get_bert_scorer():
    global _bert_scorer
    if _bert_scorer is None:
        from bert_score import BERTScorer
        _bert_scorer = BERTScorer(lang="it", rescale_with_baseline=False)
    return _bert_scorer


def compute_bert_scores(predictions: list[str], references: list[str]) -> dict[str, float | None]:
    if not bertscore_enabled():
        return dict(_SKIPPED_BERT_SCORES)
    try:
        import warnings

        scorer = get_bert_scorer()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            P, R, F1 = scorer.score(predictions, references)

        return {
            "bertscore_precision": float(P.mean().item()),
            "bertscore_recall": float(R.mean().item()),
            "bertscore_f1": float(F1.mean().item()),
        }
    except Exception as e:
        # A failure here (e.g. offline, no model) is NOT a "bad answer"; mark it
        # as skipped (None) so it is not confused with a genuine zero score.
        print(f"Error calculating BERTScore: {e}. Marking BERTScore as skipped.")
        return dict(_SKIPPED_BERT_SCORES)


def compute_embedding_similarity(predictions: list[str], references: list[str], embeddings_model) -> float:
    try:
        import numpy as np
        if not predictions or not references:
            return None

        pred_embs = embeddings_model.embed_documents(predictions)
        ref_embs = embeddings_model.embed_documents(references)

        similarities = []
        for p_emb, r_emb in zip(pred_embs, ref_embs):
            p_vec = np.array(p_emb)
            r_vec = np.array(r_emb)

            p_norm = np.linalg.norm(p_vec)
            r_norm = np.linalg.norm(r_vec)

            if p_norm > 0 and r_norm > 0:
                sim = np.dot(p_vec, r_vec) / (p_norm * r_norm)
            else:
                sim = 0.0
            similarities.append(sim)

        return float(np.mean(similarities))
    except Exception as e:
        print(f"Error calculating embedding similarity: {e}")
        return None
