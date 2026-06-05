"""Generative-evaluation orchestration.

Ties together generation (system answers) and the metrics (ROUGE / BERTScore /
embedding similarity), scored against the gold standard, the query and the
retrieved context.
"""
import pandas as pd

from .generation import generate_answers_for_eval
from .gold_answers import load_pregenerated_gold_answers_mapping
from .metrics import (
    _SKIPPED_BERT_SCORES,
    compute_bert_scores,
    compute_embedding_similarity,
    compute_rouge_scores,
)


def run_alternative_evaluation(
    eval_user_inputs: list[str],
    eval_retrieved_contexts: list[list[str]],
    provider: str,
    embeddings_model,
    gen_model: str = None,
    pregenerated_gold_answers: list[str] = None,
) -> dict[str, float]:
    # 1. Generate system answers
    system_answers = generate_answers_for_eval(
        eval_user_inputs,
        eval_retrieved_contexts,
        gen_model=gen_model,
    )

    # 2. Get pregenerated gold standard answers (the ground truth we compare against)
    if pregenerated_gold_answers is not None:
        gold_answers = list(pregenerated_gold_answers)
    else:
        # Load from pregenerated CSV columns if available
        gold_answers_mapping = load_pregenerated_gold_answers_mapping()
        gold_answers = [gold_answers_mapping.get(q.strip()) for q in eval_user_inputs]
        missing = sum(1 for a in gold_answers if a is None)
        if missing:
            print(f"[Eval] {missing}/{len(gold_answers)} questions have no gold answer; *_gold metrics will be skipped for them.")

    # Track which gold answers are actually available so we don't compare against
    # a placeholder (that would silently distort the *_gold metrics).
    gold_pairs = [
        (pred, ans)
        for pred, ans in zip(system_answers, gold_answers)
        if ans is not None and pd.notna(ans)
    ]
    gold_preds = [p for p, _ in gold_pairs]
    gold_refs = [str(a).strip() for _, a in gold_pairs]

    skipped_rouge = {"rouge1": None, "rouge2": None, "rougeL": None}
    joined_contexts = ["\n\n".join(c) for c in eval_retrieved_contexts]

    # 4. Compute ROUGE scores (gold metrics only over questions that have a gold answer)
    rouge_gold = compute_rouge_scores(gold_preds, gold_refs) if gold_pairs else dict(skipped_rouge)
    rouge_query = compute_rouge_scores(system_answers, eval_user_inputs)
    rouge_context = compute_rouge_scores(system_answers, joined_contexts)

    # 5. Compute BERTScores
    bert_gold = compute_bert_scores(gold_preds, gold_refs) if gold_pairs else dict(_SKIPPED_BERT_SCORES)
    bert_query = compute_bert_scores(system_answers, eval_user_inputs)
    bert_context = compute_bert_scores(system_answers, joined_contexts)

    # 6. Compute Embedding similarities
    emb_gold = compute_embedding_similarity(gold_preds, gold_refs, embeddings_model) if gold_pairs else None
    emb_query = compute_embedding_similarity(system_answers, eval_user_inputs, embeddings_model)
    emb_context = compute_embedding_similarity(system_answers, joined_contexts, embeddings_model)

    return {
        # System vs Gold standard
        "rouge1_gold": rouge_gold["rouge1"],
        "rouge2_gold": rouge_gold["rouge2"],
        "rougeL_gold": rouge_gold["rougeL"],
        "bert_f1_gold": bert_gold["bertscore_f1"],
        "bert_precision_gold": bert_gold["bertscore_precision"],
        "bert_recall_gold": bert_gold["bertscore_recall"],
        "emb_sim_gold": emb_gold,

        # System vs Query/Question
        "rouge1_query": rouge_query["rouge1"],
        "rouge2_query": rouge_query["rouge2"],
        "rougeL_query": rouge_query["rougeL"],
        "bert_f1_query": bert_query["bertscore_f1"],
        "emb_sim_query": emb_query,

        # System vs Context
        "rouge1_context": rouge_context["rouge1"],
        "rouge2_context": rouge_context["rouge2"],
        "rougeL_context": rouge_context["rougeL"],
        "bert_f1_context": bert_context["bertscore_f1"],
        "emb_sim_context": emb_context,
    }
