"""SHAP explanations mapped back to human-readable input features."""

from typing import Any

import numpy as np
import pandas as pd
import shap

PRETTY = {
    "tenure": "tenure",
    "MonthlyCharges": "monthly charges",
    "TotalCharges": "total charges",
    "SeniorCitizen": "senior citizen",
    "charges_per_tenure_month": "average monthly spend",
    "charge_increase": "bill above historical average",
    "n_services": "number of services",
    "is_month_to_month": "month-to-month contract",
    "auto_pay": "automatic payment",
    "fiber_no_support": "fiber optic without tech support",
    "new_customer": "new customer",
    "tenure_bucket": "tenure band (months)",
}


def _encoded_to_source(pipe: Any) -> list[str]:
    """Each encoded column -> the engineered feature it came from (one-hot groups collapse)."""
    enc = pipe.named_steps["encode"]
    names: list[str] = []
    for name, trans, cols in enc.transformers_:
        if name == "num":
            names += list(cols)
        elif name == "cat":
            for col, cats in zip(cols, trans.categories_, strict=True):
                names += [col] * len(cats)
    return names


def transform(pipe: Any, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    feats = pipe.named_steps["features"].transform(X)
    return feats, np.asarray(pipe.named_steps["encode"].transform(feats))


class Explainer:
    def __init__(self, pipe: Any, background: pd.DataFrame) -> None:
        self.pipe = pipe
        self.sources = _encoded_to_source(pipe)
        model = pipe.named_steps["model"]
        _, bg = transform(pipe, background)
        if hasattr(model, "coef_"):
            self.exp: Any = shap.LinearExplainer(model, bg)
        else:
            self.exp = shap.TreeExplainer(model)

    def values(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (engineered features, SHAP per engineered feature, log-odds scale)."""
        feats, enc = transform(self.pipe, X)
        sv = self.exp.shap_values(enc)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 1]
        grouped = pd.DataFrame(sv, columns=self.sources).T.groupby(level=0).sum().T
        return feats.reset_index(drop=True), grouped[feats.columns]

    def explain(self, X: pd.DataFrame, k: int) -> list[dict[str, Any]]:
        feats, sv = self.values(X)
        out: list[dict[str, Any]] = []
        for i in range(len(feats)):
            row = sv.iloc[i]
            top = row.reindex(row.abs().sort_values(ascending=False).index[:k])
            out.append(
                {
                    "contributions": [
                        {
                            "feature": f,
                            "value": _fmt(feats.iloc[i][f]),
                            "contribution": round(float(c), 4),
                            "text": _phrase(f, feats.iloc[i][f], float(c)),
                        }
                        for f, c in top.items()
                    ],
                }
            )
            drivers = [d["text"] for d in out[-1]["contributions"] if d["contribution"] > 0]
            out[-1]["summary"] = (
                "Risk driven mainly by: " + "; ".join(drivers)
                if drivers
                else "No strong risk drivers; profile looks like retained customers."
            )
        return out


def _fmt(v: Any) -> Any:
    if isinstance(v, float | np.floating):
        return round(float(v), 2)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _phrase(f: str, v: Any, c: float) -> str:
    name = PRETTY.get(f, f)
    direction = "raises" if c > 0 else "lowers"
    if f in {"is_month_to_month", "auto_pay", "fiber_no_support", "new_customer"}:
        label = name if int(v) == 1 else f"not {name}"
    elif isinstance(v, str):
        label = f"{name}: {v}"
    else:
        label = f"{name} = {_fmt(v)}"
    return f"{label} ({direction} risk)"
