"""Materialize engineered features (inspection + monitoring reference).

The model does NOT read these files: it re-applies FeatureEngineer inside its Pipeline,
so this stage can never cause training/serving skew.
"""

import pandas as pd

from src.config import ROOT, load_params
from src.features.build import FeatureEngineer


def main() -> None:
    p = load_params()
    fe = FeatureEngineer(p["features"]["tenure_bins"], p["features"]["new_customer_months"])
    out = ROOT / "data" / "features"
    out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(ROOT / p["data"]["processed_dir"] / f"{split}.parquet")
        feats = fe.transform(df)
        feats[p["data"]["target"]] = df[p["data"]["target"]].to_numpy()
        feats.to_parquet(out / f"{split}.parquet", index=False)
        print(f"{split}: {feats.shape}")


if __name__ == "__main__":
    main()
