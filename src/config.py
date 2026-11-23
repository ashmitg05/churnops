import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params() -> dict[str, Any]:
    with open(ROOT / "params.yaml") as f:
        params: dict[str, Any] = yaml.safe_load(f)
    return params


def setup_mlflow(params: dict[str, Any]) -> None:
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{(ROOT / uri.removeprefix('sqlite:///')).as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(params["mlflow"]["experiment"])
