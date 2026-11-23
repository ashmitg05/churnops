"""Generate a 'future' customer batch with shifted distributions and send it through the API.

Shifts (all configurable in params.yaml -> simulate):
  - more month-to-month contracts
  - higher monthly charges (price rise)
  - payment mix tilted to electronic check
  - shorter tenure (influx of new customers)
"""

import argparse
import json

import numpy as np
import pandas as pd
import requests

from src.config import ROOT, load_params


def make_batch(n: int, drift: bool, seed: int, s: dict[str, float]) -> pd.DataFrame:
    p = load_params()
    rng = np.random.default_rng(seed)
    base = pd.read_parquet(ROOT / p["data"]["processed_dir"] / "test.parquet")
    df = base.drop(columns=[p["data"]["target"]]).sample(n, replace=True, random_state=seed)
    df = df.reset_index(drop=True)
    if not drift:
        return df
    flip = rng.random(n) < s["month_to_month_share"]
    df.loc[flip, "Contract"] = "Month-to-month"
    df["MonthlyCharges"] = (
        (df["MonthlyCharges"] * s["charge_multiplier"])
        .clip(upper=p["validation"]["monthly_charges_max"])
        .round(2)
    )
    pay = rng.random(n) < s["electronic_check_share"]
    df.loc[pay, "PaymentMethod"] = "Electronic check"
    df["tenure"] = (df["tenure"] * s["tenure_multiplier"]).round().astype(int)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)
    df.loc[df["tenure"] == 0, "TotalCharges"] = None
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--no-drift", action="store_true", help="send an undrifted control batch")
    a = ap.parse_args()
    p = load_params()
    df = make_batch(a.n, not a.no_drift, p["seed"], p["simulate"])
    records = json.loads(df.to_json(orient="records"))
    size = p["serving"]["max_batch"]
    probs: list[float] = []
    for i in range(0, len(records), size):
        r = requests.post(f"{a.api}/predict/batch", json=records[i : i + size], timeout=120)
        r.raise_for_status()
        probs += [x["churn_probability"] for x in r.json()]
    print(
        f"sent {len(records)} ({'drifted' if not a.no_drift else 'control'}); "
        f"mean predicted churn={np.mean(probs):.3f}"
    )


if __name__ == "__main__":
    main()
