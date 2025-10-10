"""Collate functions for DataLoader."""

from typing import Any

import numpy as np


def unified_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Unified collate function for both single and multi-object datasets.

    Args:
        batch: List of dataset items

    Returns:
        Collated batch dictionary
    """
    images = [b["image"] for b in batch]
    labels_id = np.array([b["label_id"] for b in batch], dtype=np.int64)
    labels_cls = [b["label_cls"] for b in batch]
    metas = [b["meta"] for b in batch]
    files = [b["filename"] for b in batch]

    # Handle multi-object specific fields
    questions = [b.get("question") for b in batch]
    answers = [b.get("answer") for b in batch]
    options = [b.get("options") for b in batch]

    return {
        "image": images,
        "label": labels_id,
        "label_id": labels_id,
        "label_cls": labels_cls,
        "meta": metas,
        "filename": files,
        "question": questions,
        "answer": answers,
        "options": options,
    }
