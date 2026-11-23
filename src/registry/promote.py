"""Promotion gate: candidate replaces production only if it is better AND sane.

"Production" = the artifact that is actually deployed (models/model.joblib, DVC-tracked).
The MLflow registry alias is updated to mirror the decision (audit trail), but the deployed
file is the source of truth, so promotion still works in CI without a hosted MLflow server.
"""

import json
import os
import shutil
import sys
import time
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_auc_score

from src.config import ROOT, load_params, setup_mlflow
from src.train.train import load_split


def latency_p95_ms(pipe: Any, X: pd.DataFrame, n: int) -> float:
    times = []
    for i in range(n):
        row = X.iloc[[i % len(X)]]
        t0 = time.perf_counter()
        pipe.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 95))


def checks(
    art: dict[str, Any], X: pd.DataFrame, y: np.ndarray, pp: dict[str, Any]
) -> dict[str, Any]:
    proba = art["pipeline"].predict_proba(X)[:, 1]
    pos_rate = float((proba >= art["threshold"]).mean())
    groups = {
        str(g): float(roc_auc_score(y[m], proba[m]))
        for g in X[pp["group_column"]].unique()
        if len(set(y[(m := (X[pp["group_column"]] == g).to_numpy())])) == 2
    }
    gap = max(groups.values()) - min(groups.values())
    lat = latency_p95_ms(art["pipeline"], X, pp["latency_trials"])
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "positive_rate": pos_rate,
        "group_auc": groups,
        "group_auc_gap": gap,
        "latency_p95_ms": lat,
        "failed": [
            name
            for name, ok in {
                "not_all_one_class": pp["min_positive_rate"] <= pos_rate <= pp["max_positive_rate"],
                "non_constant_scores": float(np.std(proba)) > 0,
                "latency_within_budget": lat <= pp["latency_budget_ms"],
                "group_auc_gap": gap <= pp["max_group_auc_gap"],
            }.items()
            if not ok
        ],
    }


def decide(cand: dict[str, Any], prod: dict[str, Any] | None, min_gain: float) -> tuple[bool, str]:
    if cand["failed"]:
        return False, f"sanity checks failed: {cand['failed']}"
    if prod is None:
        return True, "no production model yet; candidate passed all sanity checks"
    gain = cand["roc_auc"] - prod["roc_auc"]
    if gain < min_gain:
        return False, f"ROC-AUC gain {gain:+.4f} < required {min_gain:+.4f}"
    return True, f"ROC-AUC gain {gain:+.4f} >= {min_gain:+.4f} and all checks passed"


def main() -> None:
    p = load_params()
    pp = p["promotion"]
    X, y = load_split("test", p)  # same held-out set for both models
    cand_art = joblib.load(ROOT / p["serving"]["candidate_path"])
    prod_path = ROOT / p["serving"]["model_path"]
    prod_art = joblib.load(prod_path) if prod_path.exists() else None

    cand = checks(cand_art, X, y, pp)
    prod = checks(prod_art, X, y, pp) if prod_art else None
    promoted, reason = decide(cand, prod, pp["min_auc_gain"])
    decision = {
        "promoted": promoted,
        "reason": reason,
        "decided_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "candidate": {"meta": cand_art["meta"], **cand},
        "production": {"meta": prod_art["meta"], **prod} if prod_art and prod else None,
    }
    out = ROOT / "reports" / "promotion.json"
    out.write_text(json.dumps(decision, indent=2, default=str))
    print(f"promoted={promoted}: {reason}")

    if promoted:
        shutil.copyfile(ROOT / p["serving"]["candidate_path"], prod_path)
    mirror_to_registry(p, cand_art["meta"], promoted, reason)
    if gh := os.environ.get("GITHUB_OUTPUT"):
        with open(gh, "a") as f:
            f.write(f"promoted={str(promoted).lower()}\n")


def mirror_to_registry(
    p: dict[str, Any], meta: dict[str, Any], promoted: bool, reason: str
) -> None:
    setup_mlflow(p)
    client, name, version = (
        mlflow.MlflowClient(),
        p["mlflow"]["model_name"],
        meta["registry_version"],
    )
    try:
        client.set_model_version_tag(name, version, "promotion_decision", str(promoted))
        client.set_model_version_tag(name, version, "promotion_reason", reason)
        if promoted:
            client.set_registered_model_alias(name, "production", version)
    except MlflowException as e:  # CI without the original tracking DB: file decision still holds
        print(f"registry mirror skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
