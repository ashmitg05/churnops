"""Deploy the API to a Hugging Face Space (Docker SDK, free CPU tier).

Stages exactly what the image needs (same files the Dockerfile copies), pushes it to the
Space repo, and syncs runtime secrets. Run from CI after `dvc pull models/model.joblib`.

Env: HF_TOKEN (write), HF_SPACE ("user/name"); optional runtime secrets
DATABASE_URL, BATCH_API_KEY, ALLOWED_ORIGINS are copied to the Space as secrets.
"""

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SECRETS = ["DATABASE_URL", "BATCH_API_KEY", "ALLOWED_ORIGINS"]
CARD = """---
title: ChurnOps API
emoji: 📉
colorFrom: gray
colorTo: red
sdk: docker
app_port: 8080
pinned: false
---

FastAPI churn model from https://github.com/{repo}. Docs at `/docs`.
"""


def stage(dst: Path) -> None:
    for f in ["Dockerfile", "requirements-serve.txt", "params.yaml"]:
        shutil.copy(ROOT / f, dst / f)
    shutil.copytree(ROOT / "src", dst / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "models").mkdir()
    shutil.copy(ROOT / "models" / "model.joblib", dst / "models" / "model.joblib")
    (dst / "reports").mkdir()
    for f in (ROOT / "reports").glob("*.json"):
        shutil.copy(f, dst / "reports" / f.name)
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    (dst / "README.md").write_text(CARD.format(repo=repo), encoding="utf-8")


def main() -> None:
    space = os.environ["HF_SPACE"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    for name in RUNTIME_SECRETS:
        if value := os.environ.get(name):
            api.add_space_secret(space, name, value)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(
            repo_id=space,
            repo_type="space",
            folder_path=tmp,
            commit_message=f"deploy {os.environ.get('GITHUB_SHA', 'local')[:7]}",
            delete_patterns=["src/**", "reports/**"],  # drop files removed upstream
        )
    print(f"deployed https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
