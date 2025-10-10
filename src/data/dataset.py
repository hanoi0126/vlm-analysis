"""Unified dataset class for single and multi-object tasks."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

try:
    from datasets import load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def _parse_json(s: Optional[str]) -> Optional[Any]:
    """Parse JSON string safely."""
    if s is None or s == "":
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


class SingleObjDataset(Dataset):
    """Dataset for single-object tasks (shape, color, size, location, angle)."""

    def __init__(
        self,
        csv_path: str,
        task: str,
        images_root: Optional[str] = None,
    ) -> None:
        """
        Initialize single-object dataset.

        Args:
            csv_path: Path to CSV metadata file
            task: Task name (e.g., 'shape', 'color')
            images_root: Root directory for images (optional)
        """
        with open(csv_path, newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r["task"] == task]

        if not self.rows:
            raise ValueError(f"No rows for task={task} in {csv_path}")

        self.images_root = images_root or str(Path(csv_path).parent.parent)

        # Build class mappings
        self.classes = sorted({r["label"] for r in self.rows})
        self.cls2id = {c: i for i, c in enumerate(self.classes)}
        self.id2cls = {i: c for c, i in self.cls2id.items()}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get dataset item.

        Args:
            idx: Index

        Returns:
            Dictionary containing image, label info, and metadata
        """
        r = self.rows[idx]
        path = Path(self.images_root) / r["filepath"]
        img = Image.open(path).convert("RGB")

        y_id = self.cls2id[r["label"]]
        y_cls = r["label"]

        meta = dict(r)
        meta["filename"] = Path(r["filepath"]).name

        return {
            "image": img,
            "label": y_id,
            "label_id": y_id,
            "label_cls": y_cls,
            "meta": meta,
            "filename": meta["filename"],
        }


class MultiObjDataset(Dataset):
    """Dataset for multi-object tasks (count, position, occlusion)."""

    def __init__(
        self,
        objects_csv: str,
        task: Optional[str] = None,
        images_root: Optional[str] = None,
    ) -> None:
        """
        Initialize multi-object dataset.

        Args:
            objects_csv: Path to objects CSV file
            task: Task name filter (optional)
            images_root: Root directory for images (optional)
        """
        self.objects_csv = Path(objects_csv)

        # Load CSV
        with open(self.objects_csv, newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]

        if task is not None:
            rows = [r for r in rows if r.get("task") == task]

        if not rows:
            raise ValueError(f"No rows (task={task}) in {objects_csv}")

        # Detect if inline QA format
        fieldset = set(rows[0].keys())
        has_inline_qa = "question" in fieldset

        # Infer images root
        sample_fp = Path(rows[0]["filepath"])
        base = self.objects_csv.parent
        if images_root:
            self.images_root = Path(images_root)
        elif (base / sample_fp).exists():
            self.images_root = base
        elif (base.parent / sample_fp).exists():
            self.images_root = base.parent
        else:
            self.images_root = base

        self.by_img: Dict[str, List[Dict]] = {}
        self.index: List = []
        self.qas: List = []

        if has_inline_qa:
            # Per-QA mode
            for r in rows:
                img_id = r.get("image_id") or Path(r["filepath"]).stem
                r["image_id"] = img_id
                objs = _parse_json(r.get("objects")) or []
                self.by_img[img_id] = objs if isinstance(objs, list) else []

                qa = {
                    "image_id": img_id,
                    "question": r.get("question"),
                    "answer": r.get("label"),
                    "options": _parse_json(r.get("options")),
                    "criterion": _parse_json(r.get("criterion")),
                    "filepath": r["filepath"],
                }
                self.qas.append(qa)
                self.index.append((img_id, r["filepath"], qa))

            self.mode = "per_qa"
        else:
            # Per-image mode
            for r in rows:
                img_id = r.get("image_id") or Path(r["filepath"]).stem
                r["image_id"] = img_id
                self.by_img.setdefault(img_id, []).append(r)

            self.mode = "per_image"
            for img_id, objs in self.by_img.items():
                self.index.append((img_id, objs[0]["filepath"], None))

        # Build vocabulary for per_qa mode
        if self.mode == "per_qa":
            opts, ans = [], []
            for _, _, qa in self.index:
                if qa and qa.get("options"):
                    opts += list(qa["options"])
                if qa and qa.get("answer") is not None:
                    ans.append(str(qa["answer"]))
            classes = sorted(set(opts)) if opts else sorted(set(ans))
            self.classes = classes
            self.cls2id = {c: i for i, c in enumerate(classes)}
            self.id2cls = {i: c for c, i in self.cls2id.items()}
        else:
            self.classes = []
            self.cls2id = {}
            self.id2cls = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get dataset item.

        Args:
            idx: Index

        Returns:
            Dictionary containing image, objects, QA info, etc.
        """
        img_id, rel_fp, qa = self.index[idx]
        img = Image.open(self.images_root / rel_fp).convert("RGB")

        sample = {
            "image": img,
            "objects": self.by_img.get(img_id, []),
            "filename": Path(rel_fp).name,
            "image_id": img_id,
            "question": None,
            "answer": None,
            "options": None,
            "qa_meta": None,
            "label_id": None,
            "label_cls": None,
        }

        if qa is not None:
            sample["question"] = qa.get("question")
            sample["answer"] = qa.get("answer")
            sample["options"] = qa.get("options")
            sample["qa_meta"] = qa

            if self.classes and str(sample["answer"]) in self.cls2id:
                y = self.cls2id[str(sample["answer"])]
                sample["label_id"] = np.int64(y)
                sample["label_cls"] = self.id2cls[y]

        return sample


class UnifiedDataset(Dataset):
    """Unified dataset for both single and multi-object tasks."""

    def __init__(
        self,
        csv_path: str,
        task: Optional[str] = None,
        images_root: Optional[str] = None,
    ) -> None:
        """
        Initialize unified dataset.

        Auto-detects whether it's single-obj or multi-obj based on CSV structure.

        Args:
            csv_path: Path to CSV metadata file
            task: Task name filter (optional)
            images_root: Root directory for images (optional)
        """
        self.csv_path = Path(csv_path)

        # Load CSV
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            rows = [dict(r) for r in csv.DictReader(f)]

        if not rows:
            raise ValueError(f"Empty CSV: {csv_path}")

        # Filter by task if specified
        if task is not None:
            rows = [r for r in rows if r.get("task") == task]
            if not rows:
                raise ValueError(f"No rows for task={task} in {csv_path}")

        # Auto-detect dataset type
        fieldset = set(rows[0].keys())

        # Check if dataset has question field
        # All tasks (both single-obj and multi-obj) now use question field
        self.has_question = "question" in fieldset
        self.has_description = "description" in fieldset

        # Infer images root
        sample_fp = Path(rows[0]["filepath"])
        base = self.csv_path.parent
        if images_root:
            self.images_root = Path(images_root)
        elif (base / sample_fp).exists():
            self.images_root = base
        elif (base.parent / sample_fp).exists():
            self.images_root = base.parent
        else:
            self.images_root = base

        self.rows = rows

        # Build class mappings
        if self.has_question:
            # Multi-object: use answer field
            opts, ans = [], []
            for r in rows:
                options = _parse_json(r.get("options"))
                if options:
                    opts.extend(options)
                if r.get("label"):
                    ans.append(str(r["label"]))
            classes = sorted(set(opts)) if opts else sorted(set(ans))
        else:
            # Single-object: use label field
            classes = sorted({r["label"] for r in rows if r.get("label")})

        self.classes = classes
        self.cls2id = {c: i for i, c in enumerate(classes)}
        self.id2cls = {i: c for c, i in self.cls2id.items()}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item.

        Returns unified format for both single and multi-object tasks.

        Returns:
            Dictionary containing:
                - image: PIL Image
                - label_id: integer label ID
                - label_cls: string label class
                - filename: image filename
                - meta: full row metadata (includes description if available)
                - question: question text (multi-obj only, None for single-obj)
                - answer: answer text (multi-obj only, None for single-obj)
        """
        r = self.rows[idx]

        # Load image
        img_path = self.images_root / r["filepath"]
        img = Image.open(img_path).convert("RGB")

        # Parse label
        label_str = r.get("label", "")
        if label_str in self.cls2id:
            label_id = self.cls2id[label_str]
            label_cls = label_str
        else:
            label_id = -1
            label_cls = None

        # Build meta dictionary (includes description if present)
        meta = dict(r)
        meta["filename"] = Path(r["filepath"]).name

        # Unified return format
        sample = {
            "image": img,
            "label_id": np.int64(label_id),
            "label_cls": label_cls,
            "filename": meta["filename"],
            "meta": meta,
        }

        # Add multi-object specific fields
        if self.has_question:
            sample["question"] = r.get("question")
            sample["answer"] = r.get("label")
            sample["options"] = _parse_json(r.get("options"))
        else:
            sample["question"] = None
            sample["answer"] = None
            sample["options"] = None

        return sample


class HuggingFaceDataset(Dataset):
    """Dataset wrapper for HuggingFace datasets."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        subset: Optional[str] = None,
        task: Optional[str] = None,
        image_column: str = "image",
        label_column: str = "label",
        task_column: str = "task",
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset name (e.g., 'username/dataset-name')
            split: Dataset split to use (default: 'train')
            subset: Dataset subset/config name (optional)
            task: Task name filter (optional)
            image_column: Name of the image column
            label_column: Name of the label column
            task_column: Name of the task column
            cache_dir: Cache directory for downloaded datasets
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "datasets library is not installed. "
                "Please install it with: pip install datasets"
            )

        # Load dataset from HuggingFace
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset

        print(
            f"Loading HuggingFace dataset: {dataset_name} (split={split}, subset={subset})"
        )
        self.hf_dataset = load_dataset(
            dataset_name,
            name=subset,
            split=split,
            cache_dir=cache_dir,
            verification_mode="no_checks",  # Skip verification to avoid 'all' split conflict
        )

        self.image_column = image_column
        self.label_column = label_column
        self.task_column = task_column

        # Filter by task if specified
        if task is not None and task_column in self.hf_dataset.column_names:
            self.hf_dataset = self.hf_dataset.filter(
                lambda x: x.get(task_column) == task
            )
            print(f"Filtered to task '{task}': {len(self.hf_dataset)} samples")

        if len(self.hf_dataset) == 0:
            raise ValueError(f"No samples found (task={task})")

        # Detect dataset structure
        self.column_names = self.hf_dataset.column_names

        # Check if dataset has question field
        # All tasks (both single-obj and multi-obj) now use question field
        self.has_question = "question" in self.column_names
        self.has_description = "description" in self.column_names

        # Build class mappings
        if self.has_question:
            # Multi-object: use answer/label field
            labels = []
            for item in self.hf_dataset:
                label = item.get(label_column)
                if label is not None:
                    labels.append(str(label))
            classes = sorted(set(labels))
        else:
            # Single-object: use label field
            labels = [
                item.get(label_column)
                for item in self.hf_dataset
                if item.get(label_column) is not None
            ]
            classes = sorted(set(labels))

        self.classes = classes
        self.cls2id = {c: i for i, c in enumerate(classes)}
        self.id2cls = {i: c for c, i in self.cls2id.items()}

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
        item = self.hf_dataset[idx]

        # Load image
        img = item.get(self.image_column)
        if img is None:
            raise ValueError(f"Image column '{self.image_column}' not found")

        # Ensure PIL Image
        if not isinstance(img, Image.Image):
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            else:
                # Try to convert from other formats (numpy array, etc.)
                img = (
                    Image.fromarray(img).convert("RGB")
                    if hasattr(img, "shape")
                    else img
                )

        # Parse label
        label_str = str(item.get(self.label_column, ""))
        if label_str in self.cls2id:
            label_id = self.cls2id[label_str]
            label_cls = label_str
        else:
            label_id = -1
            label_cls = None

        # Build meta dictionary
        meta = dict(item)
        # Use image_id or index as filename
        filename = item.get("image_id", item.get("filename", f"hf_{idx}"))
        if not isinstance(filename, str):
            filename = str(filename)
        meta["filename"] = filename

        # Unified return format
        sample = {
            "image": img,
            "label_id": np.int64(label_id),
            "label_cls": label_cls,
            "filename": filename,
            "meta": meta,
        }

        # Add multi-object specific fields
        if self.has_question:
            sample["question"] = item.get("question")
            sample["answer"] = item.get(self.label_column)
            sample["options"] = item.get("options")
        else:
            sample["question"] = None
            sample["answer"] = None
            sample["options"] = None

        return sample
