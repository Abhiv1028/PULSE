import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from joblib import load

from app.config import settings
from app.logging_config import configure_logging
from app.ml.ensemble import EnsembleBundle, ensemble_mean_std
from app.ml.explain import ExplainerBundle, explain_row, make_explainer
from app.ml.features import add_time_features
from app.ml.spatial import GridSpec, assign_grid
from app.schemas import ExplainResponse, PredictResponse, PredictRow, SimulateRequest
from app.services.pulse_service import (
    baseline_payload,
    build_dashboard_payload,
    simulate_policy_scenario,
)

app = FastAPI(title="PULSE Command Intelligence API", version="2.4.1")
log = logging.getLogger("api")


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
INDEX_HTML = STATIC_DIR / "index.html"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _fallback_predict(row: PredictRow) -> tuple[float, float]:
    hour = row.timestamp.hour
    commute = 1.0 if hour in {7, 8, 9, 16, 17, 18} else 0.0
    weather_penalty = 0.06 if row.precipitation > 0 else 0.0
    vis_penalty = max(0.0, min(0.1, (8.0 - row.visibility) / 16.0))
    lag_penalty = min(
        0.12,
        float(row.accident_lag_1 or 0.0) * 0.03
        + float(row.accident_lag_3 or 0.0) * 0.02
        + float(row.accident_lag_6 or 0.0) * 0.015,
    )

    risk = 0.02 + weather_penalty + vis_penalty + (commute * 0.035) + lag_penalty
    risk = max(0.001, min(0.95, risk))

    uncertainty = 0.012 + (0.03 if row.precipitation > 0 else 0.0) + (0.02 if row.visibility < 5 else 0.0)
    uncertainty = max(0.005, min(0.25, uncertainty))
    return float(risk), float(uncertainty)


def _fallback_explain(row: PredictRow) -> list[dict]:
    factors = []
    factors.append({"feature": "precipitation", "impact": round(0.18 if row.precipitation > 0 else 0.01, 4)})
    factors.append({"feature": "visibility", "impact": round(max(0.0, (8.0 - row.visibility) / 10.0), 4)})
    factors.append(
        {
            "feature": "commute_hour",
            "impact": round(0.12 if row.timestamp.hour in {7, 8, 9, 16, 17, 18} else 0.02, 4),
        }
    )
    factors.append({"feature": "accident_lag_1", "impact": round(float(row.accident_lag_1 or 0.0) * 0.12, 4)})
    factors.append({"feature": "accident_lag_3", "impact": round(float(row.accident_lag_3 or 0.0) * 0.08, 4)})
    factors.append({"feature": "accident_lag_6", "impact": round(float(row.accident_lag_6 or 0.0) * 0.05, 4)})
    return sorted(factors, key=lambda x: abs(float(x["impact"])), reverse=True)


@lru_cache(maxsize=1)
def get_bundle() -> EnsembleBundle | None:
    model_path = Path(settings.model_path)
    if not model_path.exists():
        log.warning("Model artifact not found at %s. Using fallback predictor.", settings.model_path)
        return None

    try:
        bundle: EnsembleBundle = load(settings.model_path)
        return bundle
    except Exception:
        log.exception("Failed loading model at %s. Using fallback predictor.", settings.model_path)
        return None


@lru_cache(maxsize=1)
def get_explainer() -> ExplainerBundle | None:
    bundle = get_bundle()
    if bundle is None:
        return None
    try:
        bg = pd.DataFrame([{c: 0.0 for c in bundle.feature_cols} for _ in range(50)])
        return make_explainer(bundle, bg)
    except Exception:
        log.warning("Explainer unavailable. Falling back to heuristic explanations.")
        return None


def row_to_features(row: PredictRow, bundle: EnsembleBundle) -> pd.DataFrame:
    grid = GridSpec(lat_edges=bundle.lat_edges, lon_edges=bundle.lon_edges)

    df = pd.DataFrame(
        [
            {
                "timestamp": row.timestamp,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "temperature": row.temperature,
                "precipitation": row.precipitation,
                "visibility": row.visibility,
                "accident": 0,
            }
        ]
    )

    df = assign_grid(df, grid)
    df = add_time_features(df)
    df["accident_lag_1"] = float(row.accident_lag_1 or 0.0)
    df["accident_lag_3"] = float(row.accident_lag_3 or 0.0)
    df["accident_lag_6"] = float(row.accident_lag_6 or 0.0)
    return df[bundle.feature_cols].astype(float)


@app.on_event("startup")
def startup() -> None:
    configure_logging()
    _ = get_bundle()
    _ = get_explainer()
    log.info("PULSE API startup complete. model_path=%s", settings.model_path)


@app.get("/", include_in_schema=False, response_model=None)
def root():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return JSONResponse(
        {
            "message": "PULSE API is running.",
            "hint": "Add app/static/index.html for full command dashboard UI.",
        }
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(row: PredictRow) -> PredictResponse:
    try:
        bundle = get_bundle()
        if bundle is None:
            mean, std = _fallback_predict(row)
            return PredictResponse(risk_mean=mean, risk_uncertainty=std)

        X = row_to_features(row, bundle=bundle)
        mean, std = ensemble_mean_std(bundle, X)
        return PredictResponse(risk_mean=float(mean[0]), risk_uncertainty=float(std[0]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain", response_model=ExplainResponse)
def explain(row: PredictRow) -> ExplainResponse:
    try:
        bundle = get_bundle()
        explainer = get_explainer()
        if bundle is None or explainer is None:
            return ExplainResponse(top_features=_fallback_explain(row))

        X = row_to_features(row, bundle=bundle)
        top = explain_row(explainer, X)
        return ExplainResponse(top_features=top)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/dashboard/bootstrap")
def dashboard_bootstrap() -> dict:
    try:
        return build_dashboard_payload()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build dashboard payload: {e}")


@app.get("/api/baseline")
def simulator_baseline() -> dict:
    try:
        return baseline_payload()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute baseline: {e}")


@app.post("/api/simulate")
def simulator_run(req: SimulateRequest) -> dict:
    try:
        return simulate_policy_scenario(req.policies, req.context)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")