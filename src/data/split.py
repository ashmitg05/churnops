"""Stratified train/val/test split with fixed seed."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ROOT, load_params
from src.data.validate import load_raw, validate


def split(
    df: pd.DataFrame, target: str, val_size: float, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val[target],
        random_state=seed,
    )
    return train, val, test


def main() -> None:
    p = load_params()
    d, s = p["data"], p["split"]
    df = validate(load_raw(d["raw_path"]))  # never split unvalidated data
    out = ROOT / d["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    for name, part in zip(
        ["train", "val", "test"],
        split(df, d["target"], s["val_size"], s["test_size"], p["seed"]),
        strict=True,
    ):
        part.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {len(part)} rows, churn={(part[d['target']] == 'Yes').mean():.3f}")


if __name__ == "__main__":
    main()
