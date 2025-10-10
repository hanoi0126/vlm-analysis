"""Collate functions for DataLoader."""

from typing import Any, Dict, List

import numpy as np


def unified_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def collate_keep_pil(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for single-object dataset (keeps PIL images).

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

    return {
        "image": images,
        "label": labels_id,
        "label_id": labels_id,
        "label_cls": labels_cls,
        "meta": metas,
        "filename": files,
    }


def collate_keep_pil_multi(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for multi-object dataset (keeps PIL images).

    Args:
        batch: List of dataset items

    Returns:
        Collated batch dictionary
    """
    images = [b["image"] for b in batch]

    return {
        "image": images,
        "objects": [b["objects"] for b in batch],
        "filename": [b["filename"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        "question": [b.get("question") for b in batch],
        "answer": [b.get("answer") for b in batch],
        "options": [b.get("options") for b in batch],
        "qa_meta": [b.get("qa_meta") for b in batch],
        "label_id": None
        if any(b.get("label_id") is None for b in batch)
        else np.asarray([b["label_id"] for b in batch], dtype=np.int64),
        "label_cls": [b.get("label_cls") for b in batch],
    }
