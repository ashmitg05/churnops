"""Feature engineering shared by training AND serving.

Every model artifact is `Pipeline(features -> encode -> model)`, so serving calls the exact
code path training used. No re-implementation in the API = no training/serving skew.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema import CATEGORIES

SERVICES = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
ENGINEERED_NUMERIC = [
    "charges_per_tenure_month",
    "charge_increase",
    "n_services",
    "is_month_to_month",
    "auto_pay",
    "fiber_no_support",
    "new_customer",
]
ENGINEERED_CATEGORICAL = ["tenure_bucket"]
BASE_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
NUMERIC_FEATURES = BASE_NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = [*CATEGORIES, *ENGINEERED_CATEGORICAL]


class FeatureEngineer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Stateless: same input row -> same output row, whether batch or single request."""

    def __init__(self, tenure_bins: list[int], new_customer_months: int) -> None:
        self.tenure_bins = tenure_bins
        self.new_customer_months = new_customer_months

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # tenure==0 customers have no bill yet -> 0 total, not unknown.
        df["TotalCharges"] = df["TotalCharges"].astype(float).fillna(0.0)
        tenure = df["tenure"].astype(float)
        months = tenure.clip(lower=1)
        df["charges_per_tenure_month"] = df["TotalCharges"] / months
        # >0 means current bill is above historical average: recent price rise.
        df["charge_increase"] = df["MonthlyCharges"] - df["charges_per_tenure_month"]
        df["n_services"] = (df[SERVICES] == "Yes").sum(axis=1) + (
            df["InternetService"] != "No"
        ).astype(int)
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
        df["auto_pay"] = df["PaymentMethod"].str.contains("automatic").astype(int)
        df["fiber_no_support"] = (
            (df["InternetService"] == "Fiber optic") & (df["TechSupport"] == "No")
        ).astype(int)
        df["new_customer"] = (tenure < self.new_customer_months).astype(int)
        df["tenure_bucket"] = pd.cut(
            tenure, bins=self.tenure_bins, include_lowest=True, right=True
        ).astype(str)
        return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def encoder() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(model: Any, fparams: dict[str, Any], sampler: Any = None) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("features", FeatureEngineer(fparams["tenure_bins"], fparams["new_customer_months"])),
        ("encode", encoder()),
    ]
    if sampler is not None:  # SMOTE must run after encoding, only during fit
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline([*steps, ("sampler", sampler), ("model", model)])
    return Pipeline([*steps, ("model", model)])


def to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """API payloads -> DataFrame with the same dtypes as training data."""
    df = pd.DataFrame.from_records(records)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"]).astype(float)
    for c in ["SeniorCitizen", "tenure"]:
        df[c] = df[c].astype(np.int64)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    return df
