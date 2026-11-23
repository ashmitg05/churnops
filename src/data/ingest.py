"""Download raw Telco churn CSV. Output is DVC-tracked; never edited by hand."""

import hashlib

import requests

from src.config import ROOT, load_params


def main() -> None:
    p = load_params()["data"]
    out = ROOT / p["raw_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(p["url"], timeout=60)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    digest = hashlib.sha256(resp.content).hexdigest()[:12]
    print(f"saved {out} ({len(resp.content)} bytes, sha256={digest})")


if __name__ == "__main__":
    main()
