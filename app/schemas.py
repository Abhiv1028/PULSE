from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PredictRow(BaseModel):
    timestamp: datetime
    latitude: float
    longitude: float
    temperature: float
    precipitation: float
    visibility: float

    # If you have a real feature store, you can omit these.
    # For now they are optional and default to 0.0 if unknown.
    accident_lag_1: Optional[float] = Field(default=0.0)
    accident_lag_3: Optional[float] = Field(default=0.0)
    accident_lag_6: Optional[float] = Field(default=0.0)


class PredictResponse(BaseModel):
    risk_mean: float
    risk_uncertainty: float


class ExplainResponse(BaseModel):
    top_features: list[dict]


class SimulateRequest(BaseModel):
    policies: dict[str, float]
    context: Optional[dict[str, float]] = Field(default=None)