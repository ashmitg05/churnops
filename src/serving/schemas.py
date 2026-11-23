"""Pydantic request schema generated from the same constants as the Pandera schema."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from src.config import load_params
from src.data.schema import CATEGORIES

_v = load_params()["validation"]

_fields: dict[str, Any] = {
    name: (Literal[tuple(values)], ...) for name, values in CATEGORIES.items()
}
_fields |= {
    "customerID": (str | None, None),
    "SeniorCitizen": (Literal[0, 1], ...),
    "tenure": (int, Field(ge=0, le=_v["tenure_max"])),
    "MonthlyCharges": (
        float,
        Field(ge=_v["monthly_charges_min"], le=_v["monthly_charges_max"]),
    ),
    "TotalCharges": (float | None, Field(default=None, ge=0)),
}


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def total_charges_rule(self) -> "_Base":
        # Same rule as Pandera: null TotalCharges only for brand-new customers.
        if self.__dict__["TotalCharges"] is None and self.__dict__["tenure"] != 0:
            raise ValueError("TotalCharges required when tenure > 0")
        return self


Customer = create_model("Customer", __base__=_Base, **_fields)


class Contribution(BaseModel):
    feature: str
    value: Any
    contribution: float
    text: str


class Prediction(BaseModel):
    churn_probability: float
    churn_label: bool
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    model_version: str
    explanation: str
    top_contributions: list[Contribution]
