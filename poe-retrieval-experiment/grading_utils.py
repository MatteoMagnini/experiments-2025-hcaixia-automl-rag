"""
Utilities to format retriever docs for grading and parse grading results back.
"""

from __future__ import annotations

import json


def docs_to_grading_input(docs: list[dict], query: str) -> dict:
    """
    Convert a list of retriever output dicts into the testi_gravidanza-style
    dict expected by AskPoE.

    Args:
        docs:  List of retriever output dicts (each with at least a 'text' key).
        query: The original user query, added as 'QUERY' key.

    Returns:
        Dict with 'QUERY' plus keys 'text1', 'text2', ... mapping to each doc's text.
    """
    grading_input = {"QUERY": query}
    for i, doc in enumerate(docs, 1):
        grading_input[f"text{i}"] = doc.get("text", "")
    return grading_input


def merge_labels_into_docs(docs: list[dict], labels: list[str]) -> list[dict]:
    """
    Merge grading labels back into the retriever output dicts.

    Args:
        docs:   Original list of retriever output dicts.
        labels: List of labels in the same order as docs.

    Returns:
        New list of dicts, each with an added 'relevance' key.

    Raises:
        ValueError: if the number of labels does not match the number of docs.
    """
    if len(labels) != len(docs):
        raise ValueError(
            f"Label count ({len(labels)}) does not match doc count ({len(docs)})"
        )
    return [{**doc, "relevance": label} for doc, label in zip(docs, labels)]
