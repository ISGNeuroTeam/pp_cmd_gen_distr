from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass#(frozen=True)
class ModelParams:
    MODELS = {
        "lr": "Линейная регрессия",
        "rf": "Случайный лес",
        "xgb": "XGBoost"
    }
    SEASONALITY_MODES = {
        "multiplicative": "Мультипликативная",
        "additive": "Аддитивная"
    }

    name: str
    # hyperparams: Dict
    seasonality_mode: Optional[str] = None
    is_autoregression: Optional[bool] = None
    is_boxcox: Optional[bool] = False
    freq: Optional[str] = "D"
    # is_exog_features: Optional[bool] = True
    is_feature_selection: Optional[bool] = False
    feature_selection_method: Optional[str] = None
