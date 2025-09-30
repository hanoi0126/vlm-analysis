"""Probing trainer for evaluating layer representations."""

from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def safe_macro_ovr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_classes: int,
) -> float:
    """
    Safely compute macro-average one-vs-rest AUC.

    Args:
        y_true: True labels
        y_score: Prediction scores
        n_classes: Number of classes

    Returns:
        Macro-average AUC score
    """
    aucs = []
    for c in range(n_classes):
        pos = y_true == c
        neg = ~pos
        if not (pos.any() and neg.any()):
            continue
        try:
            aucs.append(roc_auc_score(pos.astype(int), y_score[:, c]))
        except Exception:
            pass
    return float(np.mean(aucs)) if len(aucs) else float("nan")


def train_eval_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    max_iter: int = 2000,
    C: float = 1.0,
    solver: str = "liblinear",
) -> Dict:
    """
    Train and evaluate linear probe with cross-validation.

    Args:
        X: Feature matrix (N, D)
        y: Labels (N,)
        n_splits: Number of CV folds
        seed: Random seed
        max_iter: Max iterations for LogisticRegression
        C: Inverse regularization strength
        solver: Solver for LogisticRegression

    Returns:
        Dictionary with metrics: acc_mean, acc_std, auc_mean, auc_std, conf_mats, n, d
    """
    # Check if we have enough samples per class
    binc = np.bincount(y)
    min_per_class = int(binc.min()) if len(binc) > 1 else 0

    if min_per_class < 2:
        return {
            "acc_mean": float("nan"),
            "acc_std": float("nan"),
            "auc_mean": float("nan"),
            "auc_std": float("nan"),
            "conf_mats": [],
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "note": "not enough samples per class for StratifiedKFold",
        }

    # Adjust n_splits if needed
    n_splits = max(2, min(n_splits, min_per_class))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs: List[float] = []
    aucs: List[float] = []
    cms: List[List] = []
    n_classes = int(y.max()) + 1

    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        clf = LogisticRegression(max_iter=max_iter, C=C, solver=solver)
        clf.fit(Xtr, y[tr])

        y_pred = clf.predict(Xte)
        accs.append(accuracy_score(y[te], y_pred))
        cms.append(confusion_matrix(y[te], y_pred).tolist())

        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(Xte)
        else:
            df = clf.decision_function(Xte)
            if df.ndim == 1:
                df = np.vstack([-df, df]).T
            y_score = df

        aucs.append(safe_macro_ovr_auc(y[te], y_score, n_classes))

    return {
        "acc_mean": float(np.nanmean(accs)),
        "acc_std": float(np.nanstd(accs)),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
        "conf_mats": cms,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
    }
