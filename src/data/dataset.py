"""Unified dataset class for single and multi-object tasks."""

from pathlib import Path
from typing import Any

from datasets import load_dataset
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class HuggingFaceDataset(Dataset):
    """Dataset wrapper for HuggingFace datasets."""

    def __init__(
        self,
        dataset_name: str,
        *,
        split: str = "train",
        subset: str | None = None,
        task: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """
        Initialize HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset name (e.g., 'username/dataset-name')
            split: Dataset split to use (default: 'train')
            subset: Dataset subset/config name (optional)
            task: Task name filter (optional)
            cache_dir: Cache directory for downloaded datasets
        """
        # Load dataset from HuggingFace
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset

        print(f"Loading HuggingFace dataset: {dataset_name} (split={split}, subset={subset})")
        self.hf_dataset = load_dataset(
            dataset_name,
            name=subset,
            split=split,
            cache_dir=cache_dir,
            verification_mode="no_checks",  # Skip verification to avoid 'all' split conflict
        )

        # Column names (use defaults)
        self.image_column = "image"
        self.label_column = "label"
        self.task_column = "task"

        # Detect dataset structure
        self.column_names = self.hf_dataset.column_names  # type: ignore[attr-defined]

        # Filter by task if specified
        if task is not None and self.task_column in self.column_names:  # type: ignore[operator]
            self.hf_dataset = self.hf_dataset.filter(lambda x: x[self.task_column] == task)  # type: ignore[index]
            print(f"Filtered to task '{task}': {len(self.hf_dataset)} samples")  # type: ignore[arg-type]

        dataset_len = len(self.hf_dataset)  # type: ignore[arg-type]
        if dataset_len == 0:
            error_msg = f"No samples found (task={task})"
            raise ValueError(error_msg)

        # Check if dataset has question field
        # All tasks (both single-obj and multi-obj) now use question field
        self.has_question = "question" in self.column_names  # type: ignore[operator]
        self.has_description = "description" in self.column_names  # type: ignore[operator]

        # Build class mappings
        if self.has_question:
            # Multi-object: use answer/label field
            labels = []
            for item in self.hf_dataset:
                label = item[self.label_column]  # type: ignore[index]
                if label is not None:
                    labels.append(str(label))
            classes = sorted(set(labels))
        else:
            # Single-object: use label field
            labels = [item[self.label_column] for item in self.hf_dataset if item[self.label_column] is not None]  # type: ignore[index]
            classes = sorted(set(labels))

        self.classes = classes
        self.cls2id = {c: i for i, c in enumerate(classes)}
        self.id2cls = {i: c for c, i in self.cls2id.items()}

    def __len__(self) -> int:
        return len(self.hf_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get dataset item.

        Returns unified format compatible with UnifiedDataset.

        Returns:
            Dictionary containing:
                - image: PIL Image
                - label_id: integer label ID
                - label_cls: string label class
                - filename: image filename (or index as string)
                - meta: full item metadata
                - question: question text (multi-obj only)
                - answer: answer text (multi-obj only)
        """
        item = self.hf_dataset[idx]  # type: ignore[index]

        # Load image
        img = item[self.image_column]  # type: ignore[index]
        if img is None:
            message: str = f"Image column '{self.image_column}' not found"
            raise ValueError(message)

        # Ensure PIL Image
        if not isinstance(img, Image.Image):  # type: ignore[misc]
            if isinstance(img, str | Path):
                img = Image.open(img).convert("RGB")
            elif hasattr(img, "shape"):
                # Try to convert from numpy array
                img = Image.fromarray(img).convert("RGB")  # type: ignore[attr-defined]

        # Parse label
        label_value = item.get(self.label_column, "") if hasattr(item, "get") else item[self.label_column]  # type: ignore[index]
        label_str = str(label_value)
        if label_str in self.cls2id:
            label_id = self.cls2id[label_str]
            label_cls = label_str
        else:
            label_id = -1
            label_cls = None

        # Build meta dictionary
        meta: dict[str, Any] = dict(item) if hasattr(item, "items") else {}  # type: ignore[call-overload]
        # Use image_id or index as filename
        if hasattr(item, "get"):
            image_id = item.get("image_id", item.get("filename", f"hf_{idx}"))  # type: ignore[attr-defined]
        else:
            image_id = item.get("image_id") if "image_id" in item else f"hf_{idx}"  # type: ignore[operator]
        filename = str(image_id) if not isinstance(image_id, str) else image_id
        meta["filename"] = filename

        # Unified return format
        sample: dict[str, Any] = {
            "image": img,
            "label_id": np.int64(label_id),
            "label_cls": label_cls,
            "filename": filename,
            "meta": meta,
        }

        # Add multi-object specific fields
        if self.has_question:
            if hasattr(item, "get"):
                sample["question"] = item.get("question", None)  # type: ignore[attr-defined]
                sample["answer"] = item.get(self.label_column, None)  # type: ignore[attr-defined]
                sample["options"] = item.get("options", None)  # type: ignore[attr-defined]
            else:
                sample["question"] = item.get("question", None)  # type: ignore[attr-defined]
                sample["answer"] = item.get(self.label_column, None)  # type: ignore[attr-defined]
                sample["options"] = item.get("options", None)  # type: ignore[attr-defined]
        else:
            sample["question"] = None
            sample["answer"] = None
            sample["options"] = None

        return sample
