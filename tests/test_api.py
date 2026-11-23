"""API contract tests + served prediction == offline pipeline."""

import json
from collections.abc import Iterator
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import ROOT, load_params
from src.features.build import make_pipeline
from src.train.models import build_model

P = load_params()
TRAIN = ROOT / P["data"]["processed_dir"] / "train.parquet"
pytestmark = pytest.mark.skipif(not TRAIN.exists(), reason="run `dvc repro` first")


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(TRAIN).head(400)
    y = (df.pop(P["data"]["target"]) == "Yes").astype(int)
    return df, y


@pytest.fixture(scope="module")
def client(
    data: tuple[pd.DataFrame, pd.Series], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[TestClient]:
    X, y = data
    model, _ = build_model("logreg", P["train"]["logreg"], "class_weight", 1.0, P["seed"])
    pipe = make_pipeline(model, P["features"]).fit(X, y)
    tmp: Path = tmp_path_factory.mktemp("m")
    art = {
        "pipeline": pipe,
        "threshold": 0.4,
        "background": X.head(50),
        "meta": {"registry_version": "test", "family": "logreg"},
    }
    joblib.dump(art, tmp / "model.joblib")
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(tmp / "model.joblib"))
    mp.setenv("DATABASE_URL", f"sqlite:///{(tmp / 'p.db').as_posix()}")
    from src.serving.app import app

    with TestClient(app) as c:
        c.pipe = pipe  # type: ignore[attr-defined]
        yield c
    mp.undo()


def payloads(X: pd.DataFrame) -> list[dict[str, object]]:
    return list(json.loads(X.to_json(orient="records")))


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_contract(client: TestClient, data: tuple[pd.DataFrame, pd.Series]) -> None:
    r = client.post("/predict", json=payloads(data[0].head(1))[0])
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_label"] == (body["churn_probability"] >= 0.4)
    assert body["risk_level"] in {"low", "medium", "high"}
    assert len(body["top_contributions"]) == P["evaluate"]["top_k_explanations"]
    assert body["explanation"]


def test_served_equals_offline(client: TestClient, data: tuple[pd.DataFrame, pd.Series]) -> None:
    X = data[0].head(50)
    served = [
        p["churn_probability"] for p in client.post("/predict/batch", json=payloads(X)).json()
    ]
    offline = client.pipe.predict_proba(X)[:, 1]  # type: ignore[attr-defined]
    np.testing.assert_array_equal(np.array(served), offline)


@pytest.mark.parametrize(
    "patch",
    [
        {"Contract": "Lifetime"},
        {"tenure": -1},
        {"MonthlyCharges": 999},
        {"unknown_field": 1},
        {"tenure": 5, "TotalCharges": None},
    ],
)
def test_rejects_invalid(
    client: TestClient, data: tuple[pd.DataFrame, pd.Series], patch: dict[str, object]
) -> None:
    body = payloads(data[0].head(1))[0] | patch
    assert client.post("/predict", json=body).status_code == 422


def test_model_info_and_logging(client: TestClient) -> None:
    info = client.get("/model-info").json()
    assert info["registry_version"] == "test" and info["threshold"] == 0.4
    from src.serving.app import state
    from src.serving.db import recent_predictions

    logged = recent_predictions(state["engine"], 1000)
    assert len(logged) >= 51 and set(logged["model_version"]) == {"test"}


def test_reports_and_stats(client: TestClient) -> None:
    rep = client.get("/reports").json()
    assert set(rep) == {"metrics", "train", "promotion", "drift", "curves"}
    stats = client.get("/monitoring/stats").json()
    assert stats["total"] >= 51 and sum(s["volume"] for s in stats["series"]) == stats["total"]


def test_cors_allows_dashboard(client: TestClient) -> None:
    r = client.options(
        "/predict",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_batch_requires_key_when_configured(
    client: TestClient, data: tuple[pd.DataFrame, pd.Series], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BATCH_API_KEY", "test-key")
    body = payloads(data[0].head(2))
    assert client.post("/predict/batch", json=body).status_code == 401
    assert (
        client.post("/predict/batch", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    )
    ok = client.post("/predict/batch", json=body, headers={"X-API-Key": "test-key"})
    assert ok.status_code == 200
