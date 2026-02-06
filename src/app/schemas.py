from __future__ import annotations

from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

FeatureValue = Union[float, int, str, bool, None]

class PredictRequest (BaseModel):
    """
    Requête API: un dictionnaire {feature_name: value}
    On ne déclare pas 512 champs, on valide plutôt:
    - non vide
    - quelques règles métier 
    """
    client_id: Optional[str] = Field(
        default=None,
        description="Identifiant client (SK_ID_CURR), hors features (pour tracabilité)"
    )
    # 1 client : features = dict
    features : Dict[str, FeatureValue] = Field(
        ...,
        description="Features client sous forme dictionnaire {nom_feature: valeur}"
    )

    @field_validator("features")
    @classmethod
    def validate_features_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError("features ne doit pas être vide")
        return v 


class PredictResponse(BaseModel):
    request_id: str
    timestamp: datetime
    latency_ms: float

    probability_default: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., description="0=bon client, 1=défaut")
    decision: str = Field(..., description="Accepté / Refusé")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_features_expected: Optional[int] = None

class MetadataResponse(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    threshold: float
    fn_cost: Optional[float] = None
    fp_cost: Optional[float] = None
    n_features_expected: int