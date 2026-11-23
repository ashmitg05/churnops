"""Model factory: family + imbalance strategy -> unfitted sklearn-compatible estimator."""

from typing import Any

from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_model(
    family: str, hp: dict[str, Any], imbalance: str, pos_weight: float, seed: int
) -> tuple[Any, Any]:
    """Returns (model, sampler). sampler is None unless imbalance == 'smote'."""
    weighted = imbalance == "class_weight"
    cw = "balanced" if weighted else None
    if family == "logreg":
        model: Any = LogisticRegression(**hp, class_weight=cw, random_state=seed)
    elif family == "random_forest":
        model = RandomForestClassifier(**hp, class_weight=cw, random_state=seed, n_jobs=-1)
    elif family == "xgboost":
        model = XGBClassifier(
            **hp,
            scale_pos_weight=pos_weight if weighted else 1.0,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
        )
    elif family == "lightgbm":
        model = LGBMClassifier(
            **hp, class_weight=cw, random_state=seed, subsample_freq=1, verbose=-1, n_jobs=-1
        )
    else:
        raise ValueError(f"unknown family {family}")
    sampler = SMOTE(random_state=seed) if imbalance == "smote" else None
    return model, sampler


def search_space(family: str, trial: Any) -> dict[str, Any]:
    if family == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 10, log=True), "max_iter": 2000}
    if family == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    if family == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    raise ValueError(f"unknown family {family}")
