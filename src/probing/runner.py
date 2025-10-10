"""Experiment runner for feature extraction and probing."""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config.schema import Config
from ..data import (
    HuggingFaceDataset,
    UnifiedDataset,
    unified_collate,
)
from ..models.base import BaseFeatureExtractor
from .trainer import train_eval_probe

# Multi-object task names
MULTI_OBJ_TASKS = {"count", "position", "occlusion"}


def run_extract_probe_decode(
    extractor: BaseFeatureExtractor,
    config: Config,
    use_image: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Run feature extraction, probing, and decoding for specified tasks.

    Args:
        extractor: Feature extractor model
        config: Experiment configuration
        show_progress: Show progress bars

    Returns:
        DataFrame with summary results
    """
    out_root = Path(config.output.results_root)

    summaries: List[Dict] = []

    for task in config.experiment.tasks:
        is_multi = task in MULTI_OBJ_TASKS

        # Load dataset (HuggingFace or local CSV)
        if config.dataset.hf_dataset is not None:
            # Load from HuggingFace
            # For visual-object-attributes: split="task_name" (e.g., "color", "count")
            # or split="train" for all tasks combined
            actual_split = (
                task if config.dataset.hf_split == "auto" else config.dataset.hf_split
            )
            # If using task-specific split, no filtering needed
            # If using "train" split, need to filter by task
            filter_task = None if actual_split == task else task

            ds = HuggingFaceDataset(
                dataset_name=config.dataset.hf_dataset,
                split=actual_split,
                subset=config.dataset.hf_subset,
                task=filter_task,
                image_column=config.dataset.image_column,
                label_column=config.dataset.label_column,
                task_column=config.dataset.task_column,
            )
        else:
            # Load from local CSV
            dataset_dir = Path(config.dataset.dataset_dir)
            csv_path = dataset_dir / config.dataset.csv_filename
            ds = UnifiedDataset(
                str(csv_path),
                task=task,
                images_root=str(dataset_dir),
            )

        dl = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=unified_collate,
        )

        # Buffers for features
        arr_buf: Dict[str, List[np.ndarray]] = {"pre": [], "post": []}
        labels_chunks: List[np.ndarray] = []
        filenames_all: List[str] = []
        gen_texts_all: List[str] = []
        gen_parsed_all: List[Optional[str]] = []
        questions_all: List[str] = []

        # Iterate over dataset
        it_total = math.ceil(len(ds) / config.batch_size)
        it = tqdm(dl, total=it_total, disable=not show_progress, desc=f"{task}")

        with torch.no_grad():
            for b in it:
                # Prepare texts based on use_image flag
                if use_image:
                    # Image mode: use question from dataset (works for both single-obj and multi-obj tasks)
                    texts = b["question"]
                    if any(
                        (t is None) or (not isinstance(t, str)) or (not t.strip())
                        for t in texts
                    ):
                        raise ValueError(
                            f"[{task}] batch contains empty/None question: {texts}"
                        )
                else:
                    # Text-only mode: use description + question
                    texts = []
                    for i, meta in enumerate(b["meta"]):
                        desc = meta.get("description", "")
                        question = b["question"][i] if b["question"] else ""
                        text = f"{desc}\n\n{question}" if desc else question
                        texts.append(text)

                # Extract features
                to = extractor(
                    b["image"] if use_image else None,
                    texts=texts,
                    use_image=use_image,
                    decode=config.decode,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                )

                # Collect features
                if to.pre is not None:
                    arr_buf["pre"].append(to.pre.detach().cpu().numpy())
                if to.post is not None:
                    arr_buf["post"].append(to.post.detach().cpu().numpy())
                for name, ten in to.layers.items():
                    if name not in arr_buf:
                        arr_buf[name] = []
                    arr_buf[name].append(ten.detach().cpu().numpy())

                # Collect labels and questions
                if is_multi:
                    if b.get("label_id") is not None:
                        y_ids = b["label_id"]
                    else:
                        y_ids = np.asarray(
                            [ds.cls2id[str(a)] for a in b["answer"]], dtype=np.int64
                        )
                    labels_chunks.append(y_ids)
                else:
                    labels_chunks.append(b["label"])

                # Collect questions for all task types
                questions_all.extend(texts)

                filenames_all.extend(b["filename"])
                if config.decode:
                    if to.gen_texts is not None:
                        gen_texts_all.extend(to.gen_texts)
                    if to.gen_parsed is not None:
                        gen_parsed_all.extend(to.gen_parsed)

        # Concatenate arrays
        y = np.concatenate(labels_chunks, axis=0)
        X = {
            k: (np.concatenate(v, axis=0) if len(v) > 0 else None)
            for k, v in arr_buf.items()
        }

        # Save features
        outdir = out_root / f"{task}{config.output.suffix}"
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / "labels.npy", y)

        saved = []
        for k, v in X.items():
            if v is not None and v.size > 0:
                np.save(outdir / f"features_{k}.npy", v)
                saved.append((k, v.shape))

        # Save decode log if enabled
        decode_acc = np.nan
        if config.decode:
            if (
                len(filenames_all) != len(y)
                or len(gen_texts_all) != len(y)
                or len(gen_parsed_all) != len(y)
            ):
                raise ValueError(
                    f"[{task}] length mismatch: files={len(filenames_all)} labels={len(y)} "
                    f"texts={len(gen_texts_all)} parsed={len(gen_parsed_all)}"
                )

            # Get ground truth class names
            if is_multi:
                if hasattr(ds, "id2cls") and ds.id2cls:
                    gt_cls = [ds.id2cls[int(i)] for i in y]
                else:
                    gt_cls = [str(a) for a in questions_all]
            else:
                id2cls = getattr(ds, "id2cls", {i: c for c, i in ds.cls2id.items()})
                gt_cls = [id2cls[int(i)] for i in y]

            decode_acc = float(
                np.mean([p == g for p, g in zip(gen_parsed_all, gt_cls)])
            )

            # Save CSV
            with open(
                outdir / "decode_log.csv", "w", newline="", encoding="utf-8"
            ) as f:
                w = csv.writer(f)
                # Include question column for all task types
                header = [
                    "idx",
                    "task",
                    "filename",
                    "question",
                    "label_id",
                    "ground_truth",
                    "gen_text",
                    "gen_parsed",
                    "correct",
                ]
                w.writerow(header)

                for i in range(len(y)):
                    row = [
                        i,
                        task,
                        filenames_all[i],
                        questions_all[i],
                        int(y[i]),
                        gt_cls[i],
                        gen_texts_all[i],
                        gen_parsed_all[i],
                        int(gen_parsed_all[i] == gt_cls[i]),
                    ]
                    w.writerow(row)

        print(f"[{task}] Saved to: {outdir}")
        if config.decode:
            print(f"  decode_acc = {decode_acc:.3f} (gen_parsed == ground_truth)")

        summaries.append(
            {
                "task": task,
                "n": int(y.shape[0]),
                "saved_feats": ",".join([k for k, v in sorted(saved)]),
                "decode_acc": decode_acc,
                "outdir": str(outdir),
            }
        )

    df = pd.DataFrame(summaries).sort_values("task").reset_index(drop=True)
    return df


def probe_all_tasks(
    results_root: Path,
    tasks: List[str],
    suffix: str,
    n_folds: int = 5,
    seed: int = 0,
    max_iter: int = 2000,
    C: float = 1.0,
    solver: str = "lbfgs",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run probing on all tasks and save metrics.

    Args:
        results_root: Results root directory
        tasks: List of task names
        suffix: Directory suffix
        n_folds: Number of CV folds
        seed: Random seed
        max_iter: Max iterations for LogisticRegression
        C: Inverse regularization strength
        solver: Solver name
        verbose: Print verbose output

    Returns:
        Summary DataFrame with best layer per task
    """
    results_root = Path(results_root)
    rows: List[Dict] = []

    for task in tasks:
        task_dir = results_root / f"{task}{suffix}"

        if not task_dir.exists():
            # Try to find candidate
            cands = sorted(results_root.glob(f"{task}*"))
            cands = [c for c in cands if (c / "labels.npy").exists()]
            if not cands:
                if verbose:
                    print(f"[SKIP] no dir for task='{task}' under {results_root}")
                continue
            task_dir = cands[0]

        metrics = _probe_task_dir(
            task_dir,
            n_folds=n_folds,
            seed=seed,
            max_iter=max_iter,
            C=C,
            solver=solver,
            verbose=verbose,
        )

        # Find best layer by accuracy
        def _safe(val):
            return (
                -1.0
                if (val is None or (isinstance(val, float) and np.isnan(val)))
                else val
            )

        best_name, best_acc, best_auc = None, -1.0, -1.0
        for k, m in metrics.items():
            if _safe(m.get("acc_mean")) > best_acc:
                best_name = k
                best_acc = _safe(m.get("acc_mean"))
                best_auc = _safe(m.get("auc_mean"))

        rows.append(
            {
                "task": task,
                "task_dir": str(task_dir),
                "best_layer": best_name,
                "best_acc": best_acc if best_acc >= 0 else np.nan,
                "best_auc": best_auc if best_auc >= 0 else np.nan,
                "n": int(next(iter(metrics.values())).get("n")) if metrics else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("task").reset_index(drop=True)


def _probe_task_dir(
    task_dir: Path,
    n_folds: int = 5,
    seed: int = 0,
    max_iter: int = 2000,
    C: float = 1.0,
    solver: str = "lbfgs",
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Probe all features in a task directory.

    Args:
        task_dir: Task directory containing features_*.npy
        n_folds: Number of CV folds
        seed: Random seed
        max_iter: Max iterations for LogisticRegression
        C: Inverse regularization strength
        solver: Solver name
        verbose: Print output

    Returns:
        Dictionary mapping feature name to metrics
    """
    y_path = task_dir / "labels.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"labels.npy not found in {task_dir}")

    y = np.load(y_path)
    files = sorted(task_dir.glob("features_*.npy"))

    metrics: Dict[str, Dict] = {}
    for fp in files:
        name = fp.stem.replace("features_", "")
        X = np.load(fp)

        if X.shape[0] != y.shape[0] or X.ndim != 2:
            if verbose:
                print(
                    f"[WARN] Skip {name}: X.shape={X.shape} incompatible with y.shape[0]={y.shape[0]}"
                )
            continue

        m = train_eval_probe(
            X, y, n_splits=n_folds, seed=seed, max_iter=max_iter, C=C, solver=solver
        )
        metrics[name] = m

        if verbose:
            accm, accs = m["acc_mean"], m["acc_std"]
            aucm, aucs = m["auc_mean"], m["auc_std"]
            print(
                f"[{task_dir.name}] {name:>10s} -> acc={accm:.3f}±{accs:.3f} | auc={aucm:.3f}±{aucs:.3f}"
            )

    # Save metrics.json
    (task_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if verbose:
        print(f"Saved: {task_dir / 'metrics.json'}")

    return metrics
