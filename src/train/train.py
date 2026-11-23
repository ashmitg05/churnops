"""Compare model families x imbalance strategies, tune the best with Optuna, register candidate.

Data roles: train = fit (and Optuna CV), val = model selection + business threshold,
test = untouched until evaluate.py / promotion.
"""

import json
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import ROOT, load_params, setup_mlflow
from src.features.build import make_pipeline
from src.train.metrics import best_threshold, compute_metrics, plot_cost_curve
from src.train.models import build_model, search_space


def load_split(name: str, p: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(ROOT / p["data"]["processed_dir"] / f"{name}.parquet")
    y = (df.pop(p["data"]["target"]) == "Yes").astype(int).to_numpy()
    return df, y


def fit_eval(
    family: str,
    hp: dict[str, Any],
    imbalance: str,
    data: dict[str, tuple[pd.DataFrame, np.ndarray]],
    p: dict[str, Any],
) -> tuple[Any, dict[str, float]]:
    X, y = data["train"]
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    model, sampler = build_model(family, hp, imbalance, pos_weight, p["seed"])
    pipe = make_pipeline(model, p["features"], sampler)
    pipe.fit(X, y)
    Xv, yv = data["val"]
    proba = pipe.predict_proba(Xv)[:, 1]
    t = best_threshold(yv, proba, p["cost"])
    return pipe, {f"val_{k}": v for k, v in compute_metrics(yv, proba, t, p["cost"]).items()}


def main() -> None:
    p = load_params()
    tp = p["train"]
    setup_mlflow(p)
    data = {s: load_split(s, p) for s in ["train", "val"]}

    # 1. Family x imbalance comparison (logreg+class_weight is the Phase 2 baseline).
    results = []
    for family in tp["families"]:
        for imbalance in tp["imbalance"]:
            with mlflow.start_run(run_name=f"{family}-{imbalance}") as run:
                _, m = fit_eval(family, tp[family], imbalance, data, p)
                mlflow.log_params({"family": family, "imbalance": imbalance, **tp[family]})
                mlflow.log_metrics(m)
                mlflow.set_tags({"stage": "comparison", "baseline": str(family == "logreg")})
                results.append({"family": family, "imbalance": imbalance, **m})
                results[-1]["run_id"] = run.info.run_id
                print(f"{family:14s} {imbalance:12s} val_roc_auc={m['val_roc_auc']:.4f}")
    best = max(results, key=lambda r: r["val_roc_auc"])
    family, imbalance = best["family"], best["imbalance"]

    # 2. Optuna on the winning family. Objective = CV ROC-AUC on train only, so 50 trials
    #    cannot overfit the validation set used for selection/threshold.
    X, y = data["train"]
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    cv = StratifiedKFold(tp["cv_folds"], shuffle=True, random_state=p["seed"])

    def objective(trial: optuna.Trial) -> float:
        model, sampler = build_model(
            family, search_space(family, trial), imbalance, pos_weight, p["seed"]
        )
        pipe = make_pipeline(model, p["features"], sampler)
        return float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=p["seed"])
    )
    study.optimize(objective, n_trials=tp["optuna_trials"])
    hp = {**search_space(family, optuna.trial.FixedTrial(study.best_params))}

    # 3. Final candidate: tuned params, threshold chosen on val by business cost.
    with mlflow.start_run(run_name=f"{family}-{imbalance}-tuned") as run:
        pipe, m = fit_eval(family, hp, imbalance, data, p)
        Xv, yv = data["val"]
        proba_v = pipe.predict_proba(Xv)[:, 1]
        threshold = m["val_threshold"]
        mlflow.log_params({"family": family, "imbalance": imbalance, **hp})
        mlflow.log_metrics({**m, "optuna_best_cv_roc_auc": study.best_value})
        mlflow.set_tags({"stage": "candidate", "optuna_trials": str(tp["optuna_trials"])})
        mlflow.log_dict({"costs": p["cost"], "threshold": threshold}, "threshold.json")
        mlflow.log_figure(plot_cost_curve(yv, proba_v, p["cost"], threshold), "cost_curve_val.png")
        trials = study.trials_dataframe()
        trials.to_csv(ROOT / "reports" / "optuna_trials.csv", index=False)
        mlflow.log_artifact(str(ROOT / "reports" / "optuna_trials.csv"))
        mlflow.log_dict({"runs": results}, "comparison.json")

        sig = infer_signature(Xv.head(), proba_v[:5])
        info = mlflow.sklearn.log_model(
            pipe,
            "model",
            signature=sig,
            registered_model_name=p["mlflow"]["model_name"],
        )
        client = mlflow.MlflowClient()
        version = info.registered_model_version
        # MLflow >=2.9 replaced stages with aliases; "staging" alias = old Staging stage.
        client.set_registered_model_alias(p["mlflow"]["model_name"], "staging", version)
        client.set_model_version_tag(
            p["mlflow"]["model_name"], version, "threshold", str(threshold)
        )

    artifact = {
        "pipeline": pipe,
        "threshold": threshold,
        "background": X.sample(p["serving"]["background_rows"], random_state=p["seed"]),
        "meta": {
            "family": family,
            "imbalance": imbalance,
            "params": hp,
            "run_id": run.info.run_id,
            "registry_version": str(version),
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "val_metrics": m,
        },
    }
    out = ROOT / p["serving"]["candidate_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out)
    summary = {"best": artifact["meta"], "comparison": results}
    (ROOT / "reports" / "train.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"candidate {family}/{imbalance} v{version} val_roc_auc={m['val_roc_auc']:.4f}")


if __name__ == "__main__":
    (ROOT / "reports").mkdir(exist_ok=True)
    main()
