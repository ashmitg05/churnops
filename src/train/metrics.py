"""Metrics, business-cost threshold, and plots."""

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_cost(y: np.ndarray, proba: np.ndarray, t: float, cost: dict[str, Any]) -> float:
    """Missed churner costs churn_loss; every flagged customer gets an offer costing offer_cost."""
    pred = proba >= t
    fn = int(((~pred) & (y == 1)).sum())
    flagged = int(pred.sum())
    return float(fn * cost["churn_loss"] + flagged * cost["offer_cost"]) / len(y)


def cost_curve(
    y: np.ndarray, proba: np.ndarray, cost: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(0.0, 1.0 + 1e-9, cost["threshold_grid_step"])
    return grid, np.array([expected_cost(y, proba, t, cost) for t in grid])


def best_threshold(y: np.ndarray, proba: np.ndarray, cost: dict[str, Any]) -> float:
    grid, costs = cost_curve(y, proba, cost)
    return float(round(grid[int(np.argmin(costs))], 4))


def compute_metrics(
    y: np.ndarray, proba: np.ndarray, t: float, cost: dict[str, Any]
) -> dict[str, float]:
    pred = (proba >= t).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, proba)),
        "threshold": float(t),
        "cost_per_customer": expected_cost(y, proba, t, cost),
    }


def plot_cost_curve(y: np.ndarray, proba: np.ndarray, cost: dict[str, Any], t: float) -> Figure:
    grid, costs = cost_curve(y, proba, cost)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, costs)
    ax.axvline(t, ls="--", c="r", label=f"chosen t={t:.2f}")
    ax.axvline(0.5, ls=":", c="gray", label="t=0.5")
    ax.set(
        xlabel="threshold",
        ylabel="expected cost / customer",
        title=f"Cost curve (miss={cost['churn_loss']:.0f}, offer={cost['offer_cost']:.0f})",
    )
    ax.legend()
    fig.tight_layout()
    return fig


def plot_calibration(y: np.ndarray, proba: np.ndarray, bins: int) -> Figure:
    frac, mean = calibration_curve(y, proba, n_bins=bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="gray", label="perfect")
    ax.plot(mean, frac, marker="o", label="model")
    ax.set(xlabel="mean predicted", ylabel="observed churn rate", title="Calibration")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion(y: np.ndarray, proba: np.ndarray, t: float) -> Figure:
    cm = confusion_matrix(y, (proba >= t).astype(int))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["stay", "churn"],
        yticklabels=["stay", "churn"],
        xlabel="predicted",
        ylabel="actual",
        title=f"Confusion @ t={t:.2f}",
    )
    fig.tight_layout()
    return fig
