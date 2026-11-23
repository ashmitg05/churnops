"""Evaluate the candidate on the held-out test set; write report + log to its MLflow run."""

import json

import joblib
import matplotlib.pyplot as plt
import mlflow
import shap

from src.config import ROOT, load_params, setup_mlflow
from src.explain import Explainer
from src.train.metrics import compute_metrics, plot_calibration, plot_confusion, plot_cost_curve
from src.train.train import load_split


def main() -> None:
    p = load_params()
    setup_mlflow(p)
    art = joblib.load(ROOT / p["serving"]["candidate_path"])
    pipe, t = art["pipeline"], art["threshold"]
    X, y = load_split("test", p)
    proba = pipe.predict_proba(X)[:, 1]
    metrics = {f"test_{k}": v for k, v in compute_metrics(y, proba, t, p["cost"]).items()}

    rep = ROOT / "reports"
    figs = {
        "calibration.png": plot_calibration(y, proba, p["evaluate"]["calibration_bins"]),
        "cost_curve.png": plot_cost_curve(y, proba, p["cost"], t),
        "confusion_matrix.png": plot_confusion(y, proba, t),
    }
    exp = Explainer(pipe, art["background"])
    sample = X.sample(min(p["evaluate"]["shap_sample"], len(X)), random_state=p["seed"])
    feats, sv = exp.values(sample)
    shap.summary_plot(sv.to_numpy(), feats.astype(str), show=False, plot_type="bar")
    figs["shap_summary.png"] = plt.gcf()

    with mlflow.start_run(run_id=art["meta"]["run_id"]):
        mlflow.log_metrics(metrics)
        for name, fig in figs.items():
            fig.savefig(rep / name, dpi=110, bbox_inches="tight")
            mlflow.log_artifact(str(rep / name), "evaluation")
            plt.close(fig)
        report = {"meta": art["meta"], "test": metrics}
        mlflow.log_dict(report, "evaluation/report.json")

    (rep / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_comparison_table(rep / "train.json", metrics)
    print(json.dumps(metrics, indent=2))


def write_comparison_table(train_json: object, test: dict[str, float]) -> None:
    """Markdown table from the logged runs, for the README (numbers come from MLflow runs)."""
    data = json.loads((ROOT / "reports" / "train.json").read_text())
    rows = [
        "| family | imbalance | val ROC-AUC | val PR-AUC | val F1 | val Brier | threshold | "
        "val cost/cust | MLflow run |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(data["comparison"], key=lambda r: -r["val_roc_auc"]):
        rows.append(
            f"| {r['family']} | {r['imbalance']} | {r['val_roc_auc']:.4f} | {r['val_pr_auc']:.4f}"
            f" | {r['val_f1']:.4f} | {r['val_brier']:.4f} | {r['val_threshold']:.2f}"
            f" | {r['val_cost_per_customer']:.2f} | `{r['run_id'][:8]}` |"
        )
    b = data["best"]
    rows += [
        "",
        f"Tuned candidate ({b['family']}/{b['imbalance']}, run `{b['run_id'][:8]}`) on the"
        " **test** set:",
        "",
        "| ROC-AUC | PR-AUC | precision | recall | F1 | Brier | threshold | cost/cust |",
        "|---|---|---|---|---|---|---|---|",
        f"| {test['test_roc_auc']:.4f} | {test['test_pr_auc']:.4f} | {test['test_precision']:.4f}"
        f" | {test['test_recall']:.4f} | {test['test_f1']:.4f} | {test['test_brier']:.4f}"
        f" | {test['test_threshold']:.2f} | {test['test_cost_per_customer']:.2f} |",
    ]
    (ROOT / "reports" / "model_comparison.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
