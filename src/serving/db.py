"""Prediction log. Postgres in prod (DATABASE_URL), SQLite file locally."""

import os
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from src.config import ROOT

meta = MetaData()
predictions = Table(
    "predictions",
    meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), nullable=False, index=True),
    Column("model_version", String, nullable=False),
    Column("features", JSON, nullable=False),
    Column("probability", Float, nullable=False),
    Column("label", Boolean, nullable=False),
)


def get_engine() -> Engine:
    url = os.environ.get("DATABASE_URL", f"sqlite:///{(ROOT / 'predictions.db').as_posix()}")
    url = url.replace("postgres://", "postgresql+psycopg://", 1).replace(
        "postgresql://", "postgresql+psycopg://", 1
    )
    engine = create_engine(url, pool_pre_ping=True)
    meta.create_all(engine)
    return engine


def log_predictions(
    engine: Engine, rows: list[dict[str, Any]], probs: list[float], labels: list[bool], version: str
) -> None:
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(predictions),
            [
                {"ts": now, "model_version": version, "features": r, "probability": p, "label": lab}
                for r, p, lab in zip(rows, probs, labels, strict=True)
            ],
        )


def recent_predictions(engine: Engine, limit: int) -> pd.DataFrame:
    q = select(predictions).order_by(predictions.c.id.desc()).limit(limit)
    with engine.connect() as conn:
        df = pd.DataFrame(conn.execute(q).mappings().all())
    return df
