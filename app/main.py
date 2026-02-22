import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from joblib import load

from app.config import settings
from app.dashboard_data import (
    build_dashboard_payload,
    get_accidents_df,
    get_default_simulation_context,
    get_model_performance_proxy,
    get_scored_df,
)
from app.logging_config import configure_logging
from app.schemas import ExplainResponse, PredictResponse, PredictRow, SimulateRequest
from app.ml.ensemble import EnsembleBundle, ensemble_mean_std
from app.ml.explain import ExplainerBundle, explain_row, make_explainer
from app.ml.spatial import GridSpec, assign_grid
from app.ml.features import add_time_features

app = FastAPI(title="Urban Safety & Resource AI")

UI_PATH = Path(__file__).resolve().parent / "static" / "index.html"

POLICY_EFFECTS = {
    "speed_limit_reduction": 1.85,
    "stop_sign_increase": 1.15,
    "street_lighting_improvement": 1.05,
    "road_surface_improvement": 1.35,
    "traffic_calming_measures": 1.6,
    "pedestrian_crossing_improvements": 1.2,
    "visibility_enhancement": 0.95,
    "weather_responsive_treatment": 1.55,
}

POLICY_RESEARCH_NOTES = {
    "speed_limit_reduction": "Lowering posted speed in high-conflict corridors is strongly associated with reduced severe-crash risk.",
    "stop_sign_increase": "Intersection control upgrades improve conflict predictability at low-to-medium volume nodes.",
    "street_lighting_improvement": "Nighttime crash and injury rates decline when arterial lighting uniformity improves.",
    "road_surface_improvement": "Surface rehabilitation improves stopping performance and lowers wet-weather crash likelihood.",
    "traffic_calming_measures": "Traffic calming tends to reduce both speed variance and peak kinetic-energy impacts.",
    "pedestrian_crossing_improvements": "Protected crossings increase driver yielding behavior and reduce pedestrian exposure.",
    "visibility_enhancement": "Sight-distance and signage improvements reduce reaction-time penalties in complex approaches.",
    "weather_responsive_treatment": "Pre-treatment and adaptive winter operations are linked with lower severe-weather incident spikes.",
}


@lru_cache(maxsize=1)
def get_bundle() -> EnsembleBundle:
    bundle: EnsembleBundle = load(settings.model_path)
    return bundle


@lru_cache(maxsize=1)
def get_explainer() -> ExplainerBundle:
    # Small background for SHAP, based on a synthetic neutral sample.
    # For best results: replace with a real background sample from training data.
    bundle = get_bundle()
    bg = pd.DataFrame([{c: 0.0 for c in bundle.feature_cols} for _ in range(50)])
    return make_explainer(bundle, bg)


def row_to_features(row: PredictRow) -> pd.DataFrame:
    bundle = get_bundle()
    grid = GridSpec(lat_edges=bundle.lat_edges, lon_edges=bundle.lon_edges)

    df = pd.DataFrame([{
        "timestamp": row.timestamp,
        "latitude": row.latitude,
        "longitude": row.longitude,
        "temperature": row.temperature,
        "precipitation": row.precipitation,
        "visibility": row.visibility,
        "accident": 0,  # placeholder for lag pipeline expectations
    }])

    df = assign_grid(df, grid)
    df = add_time_features(df)

    # Lags come from request for now (or from a real feature store)
    df["accident_lag_1"] = float(row.accident_lag_1 or 0.0)
    df["accident_lag_3"] = float(row.accident_lag_3 or 0.0)
    df["accident_lag_6"] = float(row.accident_lag_6 or 0.0)

    # Construct X with exact feature columns
    X = df[get_bundle().feature_cols].astype(float)
    return X


@app.on_event("startup")
def startup() -> None:
    configure_logging()
    log = logging.getLogger("api")
    _ = get_bundle()
    try:
        _ = get_explainer()
    except Exception as exc:
        log.warning("Explainer initialization failed; /explain may be unavailable: %s", exc)
    log.info("API started. Model loaded from %s", settings.model_path)


@app.get("/", include_in_schema=False)
def ui_index() -> FileResponse:
    if not UI_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="UI file not found. Expected app/static/index.html.",
        )
    return FileResponse(UI_PATH)


@app.post("/predict", response_model=PredictResponse)
def predict(row: PredictRow) -> PredictResponse:
    try:
        X = row_to_features(row)
        mean, std = ensemble_mean_std(get_bundle(), X)
        return PredictResponse(risk_mean=float(mean[0]), risk_uncertainty=float(std[0]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain", response_model=ExplainResponse)
def explain(row: PredictRow) -> ExplainResponse:
    try:
        X = row_to_features(row)
        top = explain_row(get_explainer(), X)
        return ExplainResponse(top_features=top)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/dashboard")
def dashboard_payload() -> dict[str, Any]:
    try:
        return build_dashboard_payload()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build dashboard payload: {exc}")


def _sanitize_overrides(overrides: dict[str, float] | None) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    if not overrides:
        return cleaned
    for key, value in overrides.items():
        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError):
            continue
    return cleaned


def _context_to_predict_row(overrides: dict[str, float] | None = None) -> PredictRow:
    base = get_default_simulation_context().copy()
    ovr = _sanitize_overrides(overrides)

    ts = pd.to_datetime(base["timestamp"])
    if "month" in ovr:
        month = int(np.clip(round(ovr["month"]), 1, 12))
        ts = ts.replace(month=month)
    if "hour" in ovr:
        hour = int(np.clip(round(ovr["hour"]), 0, 23))
        ts = ts.replace(hour=hour, minute=0, second=0, microsecond=0)

    temperature = float(base["temperature"])
    precipitation = float(base["precipitation"])
    visibility = float(base["visibility"])

    weather_code = int(ovr.get("weather_code", -1))
    if weather_code == 1:  # rain
        precipitation += 0.20
        visibility *= 0.75
    elif weather_code == 2:  # snow / ice
        precipitation += 0.35
        visibility *= 0.55
        temperature = min(temperature, -2.0)
    elif weather_code == 3:  # fog
        visibility *= 0.40

    surface_code = int(ovr.get("road_surface_code", -1))
    if surface_code == 1:  # wet
        precipitation += 0.10
        visibility *= 0.85
    elif surface_code == 2:  # snow / ice
        precipitation += 0.20
        visibility *= 0.65
        temperature = min(temperature, -1.0)

    for key in ("temperature", "precipitation", "visibility"):
        if key in ovr:
            if key == "temperature":
                temperature = ovr[key]
            elif key == "precipitation":
                precipitation = ovr[key]
            else:
                visibility = ovr[key]

    lat = float(ovr.get("latitude", base["latitude"]))
    lon = float(ovr.get("longitude", base["longitude"]))
    lag_1 = float(ovr.get("accident_lag_1", base["accident_lag_1"]))
    lag_3 = float(ovr.get("accident_lag_3", base["accident_lag_3"]))
    lag_6 = float(ovr.get("accident_lag_6", base["accident_lag_6"]))

    scored = get_scored_df()
    lat = float(np.clip(lat, scored["latitude"].min(), scored["latitude"].max()))
    lon = float(np.clip(lon, scored["longitude"].min(), scored["longitude"].max()))

    return PredictRow(
        timestamp=ts.to_pydatetime(),
        latitude=lat,
        longitude=lon,
        temperature=float(np.clip(temperature, -40.0, 45.0)),
        precipitation=float(np.clip(precipitation, 0.0, 50.0)),
        visibility=float(np.clip(visibility, 0.1, 30.0)),
        accident_lag_1=float(np.clip(lag_1, 0.0, 10.0)),
        accident_lag_3=float(np.clip(lag_3, 0.0, 10.0)),
        accident_lag_6=float(np.clip(lag_6, 0.0, 10.0)),
    )


def _predict_with_uncertainty(row: PredictRow) -> tuple[float, float, pd.DataFrame]:
    X = row_to_features(row)
    mean, std = ensemble_mean_std(get_bundle(), X)
    return float(mean[0]), float(std[0]), X


@app.get("/api/baseline")
def baseline_payload() -> dict[str, Any]:
    try:
        base_row = _context_to_predict_row()
        baseline_risk, uncertainty, _ = _predict_with_uncertainty(base_row)
        perf = get_model_performance_proxy()
        scored = get_scored_df()
        return {
            "baseline_risk": baseline_risk,
            "baseline_severity": baseline_risk,
            "risk_uncertainty": uncertainty,
            "context": {
                "data_source": "PULSE scored.csv + weather_hourly.csv",
                "n_training_samples": int(len(scored)),
                "latitude": base_row.latitude,
                "longitude": base_row.longitude,
                "timestamp": base_row.timestamp.isoformat(),
            },
            "model_performance": perf,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to compute baseline: {exc}")


@app.post("/api/simulate")
def simulate(req: SimulateRequest) -> dict[str, Any]:
    if not req.policies:
        raise HTTPException(status_code=400, detail="At least one policy lever is required.")

    try:
        row = _context_to_predict_row(req.context)
        baseline_risk, baseline_unc, X = _predict_with_uncertainty(row)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not build simulation baseline: {exc}")

    components: dict[str, float] = {}
    for key, val in req.policies.items():
        if key not in POLICY_EFFECTS:
            continue
        intensity = float(np.clip(val, 0.0, 10.0))
        if intensity <= 0:
            continue
        components[key] = POLICY_EFFECTS[key] * intensity

    if not components:
        raise HTTPException(
            status_code=400,
            detail="No recognized policy levers provided.",
        )

    raw_reduction = float(sum(components.values()))
    uncertainty_penalty = min(0.35, baseline_unc * 1.8)
    reduction_pct = min(65.0, raw_reduction) * (1.0 - uncertainty_penalty)
    reduction_pct = float(np.clip(reduction_pct, 0.1, 75.0))

    ci_margin = max(1.0, baseline_unc * 100.0 * 0.30 + len(components) * 0.40)
    ci_low = float(np.clip(reduction_pct - ci_margin, 0.0, 75.0))
    ci_high = float(np.clip(reduction_pct + ci_margin, 0.0, 75.0))

    predicted_risk = max(1e-6, baseline_risk * (1.0 - reduction_pct / 100.0))
    uncertainty_score = float(
        np.clip(baseline_unc * 100.0 + len(components) * 0.7, 0.5, 30.0)
    )

    total_component = sum(components.values())
    policy_contributions = {
        k: (v / total_component) * reduction_pct for k, v in components.items()
    }

    accidents = get_accidents_df()
    if accidents.empty:
        annual_incidents = 0.0
    else:
        span_days = (accidents["timestamp"].max() - accidents["timestamp"].min()).days + 1
        span_days = max(span_days, 365)
        annual_incidents = float(len(accidents) * 365.25 / span_days)

    crash_est = int(round(annual_incidents * reduction_pct / 100.0))
    crash_low = int(round(annual_incidents * ci_low / 100.0))
    crash_high = int(round(annual_incidents * ci_high / 100.0))

    try:
        top = explain_row(get_explainer(), X)
    except Exception:
        top = []
    feature_impacts = {item["feature"]: float(item["impact"]) for item in top[:8]}

    strongest_policy = max(policy_contributions.items(), key=lambda kv: kv[1])[0]
    strongest_effect = policy_contributions[strongest_policy]
    explanation = (
        f"Baseline modeled risk at this context is {baseline_risk:.3f}. "
        f"Applying the selected policy bundle projects risk to {predicted_risk:.3f} "
        f"({reduction_pct:.1f}% reduction).\n\n"
        f"The strongest contributor is '{strongest_policy}' at {strongest_effect:.1f} "
        f"percentage points of total projected reduction. "
        "Uncertainty is derived from ensemble spread and scenario complexity."
    )
    research_notes = {
        k: POLICY_RESEARCH_NOTES[k] for k in components if k in POLICY_RESEARCH_NOTES
    }

    return {
        "risk_reduction_pct": reduction_pct,
        "confidence_interval": [ci_low, ci_high],
        "predicted_risk": predicted_risk,
        "baseline_risk": baseline_risk,
        "crash_count_reduction": {
            "low": crash_low,
            "estimated": crash_est,
            "high": crash_high,
        },
        "uncertainty_score": uncertainty_score,
        "policy_contributions": policy_contributions,
        "feature_impacts": feature_impacts,
        "explanation": explanation,
        "research_notes": research_notes,
    }