"""FastAPI serving the baked production artifact (models/model.joblib)."""

import hmac
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import ROOT, load_params
from src.explain import Explainer
from src.features.build import to_frame
from src.serving.db import get_engine, log_predictions, recent_predictions
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
REPORT_FILES = ["metrics", "train", "promotion", "drift", "curves"]


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


def require_batch_key(x_api_key: str | None = Header(default=None)) -> None:
    """Batch scoring is costly (SHAP per row) and writes to the monitoring log: key-gated.

    Unset BATCH_API_KEY = open (local dev). Set it on Cloud Run.
    """
    expected = os.environ.get("BATCH_API_KEY")
    if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(401, "missing or invalid X-API-Key")


@app.post(
    "/predict/batch",
    response_model=list[Prediction],
    dependencies=[Depends(require_batch_key)],
)
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


@app.get("/reports")
def reports() -> dict[str, Any]:
    """Latest pipeline reports baked into the image (null when a report was not produced)."""
    out: dict[str, Any] = {}
    for name in REPORT_FILES:
        f = ROOT / "reports" / f"{name}.json"
        out[name] = json.loads(f.read_text()) if f.exists() else None
    return out


@app.get("/monitoring/stats")
def monitoring_stats() -> dict[str, Any]:
    df = recent_predictions(state["engine"], P["monitoring"]["stats_limit"])
    if df.empty:
        return {"total": 0, "series": []}
    df["bucket"] = pd.to_datetime(df["ts"], utc=True).dt.floor(P["monitoring"]["stats_bucket"])
    g = df.groupby("bucket").agg(volume=("id", "size"), churn_rate=("label", "mean"))
    series = [
        {"ts": ts.isoformat(), "volume": int(r.volume), "churn_rate": float(r.churn_rate)}
        for ts, r in g.iterrows()
    ]
    return {"total": len(df), "mean_probability": float(df["probability"].mean()), "series": series}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if "art" in state else "loading"}
