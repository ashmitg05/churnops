from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params() -> dict[str, Any]:
    with open(ROOT / "params.yaml") as f:
        params: dict[str, Any] = yaml.safe_load(f)
    return params
