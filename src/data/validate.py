"""Pandera schema for raw Telco data. Raises on any violation -> pipeline stops."""

import sys
from typing import Any

import pandas as pd
import pandera as pa
from pandera import Check, Column

from src.config import ROOT, load_params
from src.data.schema import CATEGORIES, YES_NO


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / path)
    # Raw file stores TotalCharges as text; 11 rows with tenure==0 hold " ".
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip().replace("", None))
    return df


def build_schema(v: dict[str, Any], target: str) -> pa.DataFrameSchema:
    cats = {name: Column(str, Check.isin(values)) for name, values in CATEGORIES.items()}
    return pa.DataFrameSchema(
        {
            "customerID": Column(str, unique=True),
            **cats,
            "SeniorCitizen": Column(int, Check.isin([0, 1])),
            "tenure": Column(int, Check.in_range(0, v["tenure_max"])),
            "MonthlyCharges": Column(
                float, Check.in_range(v["monthly_charges_min"], v["monthly_charges_max"])
            ),
            "TotalCharges": Column(float, Check.ge(0), nullable=True),
            target: Column(str, Check.isin(YES_NO)),
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
