"""FastAPI serving the baked production artifact (models/model.joblib)."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException

from src.config import ROOT, load_params
from src.explain import Explainer
from src.features.build import to_frame
from src.serving.db import get_engine, log_predictions
from src.serving.schemas import Customer, Prediction

P = load_params()
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    path = os.environ.get("MODEL_PATH", str(ROOT / P["serving"]["model_path"]))
    art = joblib.load(path)
    state["art"] = art
    state["explainer"] = Explainer(art["pipeline"], art["background"])
    state["engine"] = get_engine()
    yield
    state.clear()


app = FastAPI(title="ChurnOps", lifespan=lifespan)


def risk_level(prob: float, threshold: float) -> str:
    # high = above business threshold; medium = within half the threshold below it.
    if prob >= threshold:
        return "high"
    return "medium" if prob >= threshold / 2 else "low"


def predict_records(records: list[dict[str, Any]]) -> list[Prediction]:
    art = state["art"]
    X = to_frame(records)
    probs = art["pipeline"].predict_proba(X)[:, 1].tolist()
    t = art["threshold"]
    labels = [p >= t for p in probs]
    expl = state["explainer"].explain(X, P["evaluate"]["top_k_explanations"])
    version = art["meta"]["registry_version"]
    log_predictions(state["engine"], records, probs, labels, version)
    return [
        Prediction(
            churn_probability=p,
            churn_label=lab,
            risk_level=risk_level(p, t),  # type: ignore[arg-type]
            threshold=t,
            model_version=version,
            explanation=e["summary"],
            top_contributions=e["contributions"],
        )
        for p, lab, e in zip(probs, labels, expl, strict=True)
    ]


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:  # type: ignore[valid-type]
    return predict_records([customer.model_dump()])[0]  # type: ignore[attr-defined]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]) -> list[Prediction]:  # type: ignore[valid-type]
    if not customers:
        raise HTTPException(422, "empty batch")
    if len(customers) > P["serving"]["max_batch"]:
        raise HTTPException(413, f"batch > {P['serving']['max_batch']}")
    return predict_records([c.model_dump() for c in customers])  # type: ignore[attr-defined]


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    meta = state["art"]["meta"]
    return {**meta, "threshold": state["art"]["threshold"]}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if "art" in state else "loading"}
