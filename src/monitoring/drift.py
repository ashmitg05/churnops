"""Compare recent served inputs against the training reference; trigger retrain on drift."""

import json
import os
import sys
from typing import Any

import pandas as pd
import requests
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import ROOT, load_params
from src.data.schema import CATEGORIES, NUMERIC
from src.serving.db import get_engine, recent_predictions

FEATURES = [*CATEGORIES, *NUMERIC]


def current_window(limit: int) -> pd.DataFrame:
    df = recent_predictions(get_engine(), limit)
    if df.empty:
        return df
    feats = pd.json_normalize(df["features"].tolist())[FEATURES]
    feats["probability"] = df["probability"].to_numpy()
    return feats


def run_report(reference: pd.DataFrame, current: pd.DataFrame, m: dict[str, Any]) -> dict[str, Any]:
    cm = ColumnMapping(numerical_features=NUMERIC, categorical_features=list(CATEGORIES))
    report = Report(metrics=[DataDriftPreset(stattest_threshold=m["stattest_threshold"])])
    report.run(
        reference_data=reference[FEATURES], current_data=current[FEATURES], column_mapping=cm
    )
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    report.save_html(str(out / "drift.html"))
    res = report.as_dict()["metrics"]
    summary, table = res[0]["result"], res[1]["result"]["drift_by_columns"]
    return {
        "share_drifted": float(summary["share_of_drifted_columns"]),
        "n_drifted": int(summary["number_of_drifted_columns"]),
        "drifted_columns": sorted(c for c, v in table.items() if v["drift_detected"]),
        "n_current": len(current),
        "mean_predicted_churn": float(current["probability"].mean()),
    }


def trigger_retrain(workflow: str, reason: str) -> bool:
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not (token and repo):
        print("GITHUB_TOKEN/GITHUB_REPOSITORY not set; skipping dispatch", file=sys.stderr)
        return False
    r = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={"ref": os.environ.get("GITHUB_REF_NAME", "main"), "inputs": {"reason": reason}},
        timeout=30,
    )
    r.raise_for_status()  # 204 on success
    return True


def main() -> None:
    p = load_params()
    m = p["monitoring"]
    reference = pd.read_parquet(ROOT / p["data"]["processed_dir"] / "train.parquet")
    current = current_window(m["window"])
    if len(current) < 30:
        print(f"only {len(current)} predictions logged; need >= 30 for a drift test")
        return
    result = run_report(reference, current, m)
    result["drift_detected"] = result["share_drifted"] > m["drift_share_threshold"]
    result["checked_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    if result["drift_detected"]:
        reason = f"drift share {result['share_drifted']:.2f}: {result['drifted_columns']}"
        result["retrain_triggered"] = trigger_retrain(m["retrain_workflow"], reason)
    (ROOT / "reports" / "drift.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
