from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd
try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None

from app.ml.ensemble import EnsembleBundle


@dataclass
class ExplainerBundle:
    explainer: Any
    feature_cols: List[str]


def make_explainer(bundle: EnsembleBundle, X_background: pd.DataFrame) -> ExplainerBundle:
    if shap is None:
        raise RuntimeError("shap is not installed; explanation model unavailable")

    # Use first model as representative explainer target
    model = bundle.models[0]
    background = X_background[bundle.feature_cols].copy()

    explainer = shap.Explainer(model, background)
    return ExplainerBundle(explainer=explainer, feature_cols=bundle.feature_cols)


def explain_row(expl: ExplainerBundle, x_row: pd.DataFrame) -> list[dict]:
    x_use = x_row[expl.feature_cols]
    values = expl.explainer(x_use)

    # values.values shape: (1, n_features)
    impacts = values.values[0]
    names = expl.feature_cols

    pairs = sorted(
        [{"feature": f, "impact": float(v)} for f, v in zip(names, impacts)],
        key=lambda d: abs(d["impact"]),
        reverse=True,
    )
    return pairs[:8]