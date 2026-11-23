import pandas as pd
import pandera as pa
import pytest

from src.config import ROOT, load_params
from src.data.split import split
from src.data.validate import load_raw, validate

P = load_params()
RAW = ROOT / P["data"]["raw_path"]
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="run `dvc repro` or `dvc pull` first")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_raw(P["data"]["raw_path"])


def test_real_data_passes(df: pd.DataFrame) -> None:
    validate(df)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda d: d.assign(Contract=d["Contract"].replace("Two year", "Lifetime")),
        lambda d: d.assign(tenure=d["tenure"].replace(72, 500)),
        lambda d: d.assign(MonthlyCharges=-d["MonthlyCharges"]),
        lambda d: d.drop(columns=["PaymentMethod"]),
        lambda d: d.assign(TotalCharges=None),
        lambda d: d.assign(Churn="No"),  # churn rate 0 -> out of band
        lambda d: pd.concat([d, d.head(1)]),  # duplicate customerID
    ],
)
def test_bad_data_fails(df: pd.DataFrame, corrupt: object) -> None:
    with pytest.raises(pa.errors.SchemaErrors):
        validate(corrupt(df.copy()))  # type: ignore[operator]


def test_split_stratified_disjoint(df: pd.DataFrame) -> None:
    t = P["data"]["target"]
    parts = split(df, t, P["split"]["val_size"], P["split"]["test_size"], P["seed"])
    ids = [set(x["customerID"]) for x in parts]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(map(len, parts)) == len(df)
    base = (df[t] == "Yes").mean()
    for x in parts:
        assert abs((x[t] == "Yes").mean() - base) < 0.01
    again = split(df, t, P["split"]["val_size"], P["split"]["test_size"], P["seed"])
    assert parts[2]["customerID"].tolist() == again[2]["customerID"].tolist()
