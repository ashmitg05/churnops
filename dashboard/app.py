"""ChurnOps dashboard: Predict / Model / Monitoring. Run: streamlit run dashboard/app.py"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.schema import CATEGORIES  # noqa: E402  same categories as training/serving

API = os.environ.get("API_URL", "http://localhost:8000")
REPORTS = ROOT / "reports"

st.set_page_config(page_title="ChurnOps", layout="wide")
predict_tab, model_tab, monitor_tab = st.tabs(["Predict", "Model", "Monitoring"])

with predict_tab:
    st.header("Score one customer")
    with st.form("customer"):
        cols = st.columns(3)
        row: dict[str, object] = {}
        for i, (name, values) in enumerate(CATEGORIES.items()):
            row[name] = cols[i % 3].selectbox(name, values)
        row["SeniorCitizen"] = int(cols[0].checkbox("Senior citizen"))
        row["tenure"] = cols[1].number_input("tenure (months)", 0, 72, 3)
        row["MonthlyCharges"] = cols[2].number_input("MonthlyCharges", 0.0, 200.0, 85.0)
        total = cols[0].number_input("TotalCharges (0 if new)", 0.0, 20000.0, 255.0)
        row["TotalCharges"] = None if row["tenure"] == 0 else total
        submitted = st.form_submit_button("Predict")
    if submitted:
        r = requests.post(f"{API}/predict", json=row, timeout=30)
        if r.ok:
            res = r.json()
            level = res["risk_level"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Churn probability", f"{res['churn_probability']:.1%}")
            c2.metric("Risk level", level.upper())
            c3.metric("Business threshold", f"{res['threshold']:.2f}")
            st.info(res["explanation"])
            st.dataframe(pd.DataFrame(res["top_contributions"]), hide_index=True)
        else:
            st.error(f"{r.status_code}: {r.text}")

with model_tab:
    st.header("Current model")
    try:
        info = requests.get(f"{API}/model-info", timeout=10).json()
        st.json(info, expanded=False)
    except requests.RequestException as e:
        st.warning(f"API unreachable: {e}")
    if (REPORTS / "metrics.json").exists():
        st.subheader("Test metrics")
        st.dataframe(
            pd.Series(json.loads((REPORTS / "metrics.json").read_text())).to_frame("value")
        )
    c1, c2 = st.columns(2)
    for col, img in [(c1, "calibration.png"), (c2, "cost_curve.png")]:
        if (REPORTS / img).exists():
            col.image(str(REPORTS / img))
    if (REPORTS / "model_comparison.md").exists():
        st.subheader("Model comparison")
        st.markdown((REPORTS / "model_comparison.md").read_text())
    if (REPORTS / "promotion.json").exists():
        st.subheader("Last promotion decision")
        d = json.loads((REPORTS / "promotion.json").read_text())
        st.write(f"**promoted={d['promoted']}** — {d['reason']} ({d['decided_at']})")

with monitor_tab:
    st.header("Monitoring")
    if (REPORTS / "drift.json").exists():
        d = json.loads((REPORTS / "drift.json").read_text())
        c1, c2, c3 = st.columns(3)
        c1.metric("Drifted columns", f"{d['n_drifted']} ({d['share_drifted']:.0%})")
        c2.metric("Drift detected", str(d["drift_detected"]))
        c3.metric("Mean predicted churn", f"{d['mean_predicted_churn']:.1%}")
        st.write("Drifted:", ", ".join(d["drifted_columns"]) or "none")
    else:
        st.write("No drift report yet. Run `python -m src.monitoring.drift`.")
    try:
        from src.serving.db import get_engine, recent_predictions

        df = recent_predictions(get_engine(), 100_000)
        if not df.empty:
            df["hour"] = pd.to_datetime(df["ts"]).dt.floor("h")
            g = df.groupby("hour").agg(volume=("id", "size"), churn_rate=("label", "mean"))
            st.subheader("Prediction volume")
            st.bar_chart(g["volume"])
            st.subheader("Predicted churn rate")
            st.line_chart(g["churn_rate"])
    except Exception as e:  # dashboard must render even without DB access
        st.warning(f"prediction log unavailable: {e}")
