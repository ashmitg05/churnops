"""Training/serving skew: the API path (JSON dict -> to_frame) must equal the training path."""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import ROOT, load_params
from src.features.build import FeatureEngineer, make_pipeline, to_frame
from src.train.models import build_model

P = load_params()
TRAIN = ROOT / P["data"]["processed_dir"] / "train.parquet"
pytestmark = pytest.mark.skipif(not TRAIN.exists(), reason="run `dvc repro` first")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_parquet(TRAIN).drop(columns=[P["data"]["target"]]).head(300)


def as_api_payloads(df: pd.DataFrame) -> list[dict[str, object]]:
    # Round-trip through JSON exactly like an HTTP request body.
    return list(json.loads(df.to_json(orient="records")))


def test_features_identical_batch_vs_api(df: pd.DataFrame) -> None:
    fe = FeatureEngineer(**P["features"])
    offline = fe.transform(df).reset_index(drop=True)
    online = pd.concat(
        [fe.transform(to_frame([r])) for r in as_api_payloads(df)], ignore_index=True
    )
    pd.testing.assert_frame_equal(offline, online)


def test_pipeline_proba_identical(df: pd.DataFrame) -> None:
    y = (pd.read_parquet(TRAIN)[P["data"]["target"]].head(300) == "Yes").astype(int)
    model, _ = build_model("logreg", P["train"]["logreg"], "class_weight", 1.0, P["seed"])
    pipe = make_pipeline(model, P["features"]).fit(df, y)
    np.testing.assert_array_equal(
        pipe.predict_proba(df)[:, 1], pipe.predict_proba(to_frame(as_api_payloads(df)))[:, 1]
    )


def test_engineered_values() -> None:
    row = {
        "customerID": "x", "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 0, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 80.0, "TotalCharges": None,
    }  # fmt: skip
    f = FeatureEngineer(**P["features"]).transform(to_frame([row])).iloc[0]
    assert f["TotalCharges"] == 0.0 and f["charges_per_tenure_month"] == 0.0
    assert f["charge_increase"] == 80.0
    assert f["n_services"] == 3  # phone + backup + internet
    assert f["is_month_to_month"] == 1 and f["auto_pay"] == 0
    assert f["fiber_no_support"] == 1 and f["new_customer"] == 1
