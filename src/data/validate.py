"""Pandera schema for raw Telco data. Raises on any violation -> pipeline stops."""

import sys
from typing import Any

import pandas as pd
import pandera as pa
from pandera import Check, Column

from src.config import ROOT, load_params

YES_NO = ["Yes", "No"]
NO_INTERNET = ["Yes", "No", "No internet service"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / path)
    # Raw file stores TotalCharges as text; 11 rows with tenure==0 hold " ".
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip().replace("", None))
    return df


def build_schema(v: dict[str, Any], target: str) -> pa.DataFrameSchema:
    def cat(values: list[str]) -> Column:
        return Column(str, Check.isin(values))

    return pa.DataFrameSchema(
        {
            "customerID": Column(str, unique=True),
            "gender": cat(["Male", "Female"]),
            "SeniorCitizen": Column(int, Check.isin([0, 1])),
            "Partner": cat(YES_NO),
            "Dependents": cat(YES_NO),
            "tenure": Column(int, Check.in_range(0, v["tenure_max"])),
            "PhoneService": cat(YES_NO),
            "MultipleLines": cat(["Yes", "No", "No phone service"]),
            "InternetService": cat(["DSL", "Fiber optic", "No"]),
            "OnlineSecurity": cat(NO_INTERNET),
            "OnlineBackup": cat(NO_INTERNET),
            "DeviceProtection": cat(NO_INTERNET),
            "TechSupport": cat(NO_INTERNET),
            "StreamingTV": cat(NO_INTERNET),
            "StreamingMovies": cat(NO_INTERNET),
            "Contract": cat(["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": cat(YES_NO),
            "PaymentMethod": cat(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
            "MonthlyCharges": Column(
                float, Check.in_range(v["monthly_charges_min"], v["monthly_charges_max"])
            ),
            "TotalCharges": Column(float, Check.ge(0), nullable=True),
            target: cat(YES_NO),
        },
        checks=[
            Check(lambda df: len(df) >= v["min_rows"], error="too few rows"),
            # Nulls allowed only for brand-new customers (no bill yet).
            Check(
                lambda df: (df["TotalCharges"].notna() | (df["tenure"] == 0)).all(),
                error="TotalCharges null for tenure>0",
            ),
            Check(
                lambda df: v["churn_rate_min"]
                <= (df[target] == "Yes").mean()
                <= v["churn_rate_max"],
                error="churn rate outside expected band",
            ),
        ],
        strict=True,  # unknown/missing columns fail
        coerce=False,
    )


def validate(df: pd.DataFrame) -> pd.DataFrame:
    p = load_params()
    return build_schema(p["validation"], p["data"]["target"]).validate(df, lazy=True)


def main() -> None:
    p = load_params()
    try:
        validate(load_raw(p["data"]["raw_path"]))
    except pa.errors.SchemaErrors as e:
        print(e.failure_cases.to_string(), file=sys.stderr)
        sys.exit(1)
    print("validation passed")


if __name__ == "__main__":
    main()
