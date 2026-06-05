"""Shared constants for the AutoML-RAG experiment.

Kept dependency-free so every other experiment module can import it without
risking circular imports. `load_dotenv()` runs here (before EVAL_SAMPLE_SIZE is
read) so a `.env` file is honoured regardless of import order.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# A failed/unavailable metric is reported to SMAC as this worst-case cost, since
# SMAC cannot handle NaN/None costs.
FAILURE_ACCURACY_COST = 1.0

# Objectives that actually drive the SMAC optimization. We keep these to the
# cheap/deterministic retrieval metrics plus the gold-answer generation quality.
# The query/context BERTScores are still computed and logged, but they are poor
# optimization targets (bert_f1_context rewards copying the context verbatim),
# so they are intentionally NOT optimized.
SMAC_OBJECTIVES = ["1 - accuracy", "number of documents", "1 - bert_f1_gold"]

# How many training questions to use for the (expensive) generative evaluation.
DEFAULT_EVAL_SAMPLE_SIZE = int(os.environ.get("EVAL_SAMPLE_SIZE", "100"))
