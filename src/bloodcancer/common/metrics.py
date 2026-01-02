from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

def compute_metrics(
    y_true,
    y_prob,
    y_pred=None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute classification metrics.

    Returns:
      - accuracy, f1_macro, optional auc (binary)
      - confusion_matrix (list[list[int]])
      - per_class (dict[class_name] -> {precision, recall, f1, support})
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_pred is None:
        y_pred = y_prob.argmax(axis=1)
    y_pred = np.asarray(y_pred)

    out: Dict[str, Any] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))

    # AUC only for binary by default
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception:
            out["auc"] = None

    cm = confusion_matrix(y_true, y_pred)
    out["confusion_matrix"] = cm.tolist()

    # per-class stats
    try:
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        names = class_names if class_names and len(class_names) == len(p) else [str(i) for i in range(len(p))]
        per = {}
        for i, name in enumerate(names):
            per[name] = {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(s[i]),
            }
        out["per_class"] = per
    except Exception:
        out["per_class"] = {}

    return out
