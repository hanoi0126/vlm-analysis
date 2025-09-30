"""Dataset classes for single and multi-object tasks."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def _parse_json(s: Optional[str]) -> Optional[any]:
    """Parse JSON string safely."""
    if s is None or s == "":
        return None

    return json.loads(s)


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
