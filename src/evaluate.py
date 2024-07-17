"""Evaluation, model comparison and stability diagnostics.

Because the positive class is rare (~15%), accuracy is a vanity metric (predict
'never defaults' -> 85% accuracy). We lead with F1, ROC-AUC and PR-AUC, check
probability calibration (so a "30% risk" really means 30%), and tune the
decision threshold to the business objective instead of blindly using 0.5.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Full metric suite at a given decision threshold."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def find_best_threshold(y_true, y_proba, objective: str = "f1") -> float:
    """Pick the probability cut-off that maximises F1 (default) or Youden's J."""
    if objective == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_proba)
        return float(thr[np.argmax(tpr - fpr)])
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    # thr has len-1 vs prec/rec; align
    best = int(np.argmax(f1[:-1])) if len(thr) else 0
    return float(thr[best]) if len(thr) else 0.5


def overfit_gap(train_metrics: dict, test_metrics: dict, key: str = "roc_auc") -> float:
    """train - test on a metric; large positive gap => overfitting."""
    return float(train_metrics[key] - test_metrics[key])


# --------------------------------------------------------------------------- #
# Plots — all saved to results/figures
# --------------------------------------------------------------------------- #
def plot_roc_curves(results: dict, path=None):
    """results: {name: (y_true, y_proba)}."""
    plt.figure(figsize=(7, 6))
    for name, (yt, yp) in results.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yt, yp):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves"); plt.legend(loc="lower right", fontsize=8)
    path = path or config.FIGURES_DIR / "roc_curves.png"
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    return path


def plot_pr_curves(results: dict, path=None):
    plt.figure(figsize=(7, 6))
    for name, (yt, yp) in results.items():
        prec, rec, _ = precision_recall_curve(yt, yp)
        plt.plot(rec, prec, label=f"{name} (AP={average_precision_score(yt, yp):.3f})")
    base = np.mean(list(results.values())[0][0])
    plt.axhline(base, color="k", ls="--", lw=1, label=f"baseline={base:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall curves"); plt.legend(loc="upper right", fontsize=8)
    path = path or config.FIGURES_DIR / "pr_curves.png"
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    return path


def plot_confusion(y_true, y_pred, path=None, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No default", "Default"]); ax.set_yticklabels(["No default", "Default"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.colorbar(im)
    path = path or config.FIGURES_DIR / "confusion_matrix.png"
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    return path


def plot_calibration(y_true, y_proba, path=None, name="best model"):
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    plt.plot(mean_pred, frac_pos, "o-", label=name)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed default rate")
    plt.title("Calibration curve"); plt.legend(loc="upper left")
    path = path or config.FIGURES_DIR / "calibration_curve.png"
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    return path


def plot_model_comparison(comparison_df: pd.DataFrame, metric="f1", path=None):
    df = comparison_df.sort_values(metric, ascending=True)
    plt.figure(figsize=(8, max(3, 0.5 * len(df))))
    plt.barh(df["model"], df[metric], color="#4C72B0")
    plt.xlabel(metric); plt.title(f"Model comparison ({metric})")
    for i, v in enumerate(df[metric]):
        plt.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    path = path or config.FIGURES_DIR / f"model_comparison_{metric}.png"
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    return path


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path
