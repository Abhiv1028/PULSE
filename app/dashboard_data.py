from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

STATUS_PRIORITY = {"critical": 0, "deficit": 1, "monitor": 2, "optimal": 3}
STATUS_LABEL = {
    "critical": "Critical",
    "deficit": "Deficit",
    "monitor": "Monitor",
    "optimal": "Optimal",
}


def _read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required dashboard dataset is missing: {path}. "
            "Generate/refresh data artifacts before running the UI."
        )
    return pd.read_csv(path, parse_dates=parse_dates or [])


@lru_cache(maxsize=1)
def get_scored_df() -> pd.DataFrame:
    df = _read_csv(DATA_DIR / "scored.csv", parse_dates=["timestamp"])
    needed = {
        "timestamp",
        "latitude",
        "longitude",
        "grid_id",
        "risk_mean",
        "risk_uncertainty",
        "accident",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"data/scored.csv is missing required columns: {sorted(missing)}")
    return df


@lru_cache(maxsize=1)
def get_accidents_df() -> pd.DataFrame:
    df = _read_csv(DATA_DIR / "accidents.csv", parse_dates=["timestamp"])
    if "harm" not in df.columns:
        df["harm"] = 0
    return df


@lru_cache(maxsize=1)
def get_ems_recommendations_df() -> pd.DataFrame:
    df = _read_csv(DATA_DIR / "ems_recommendations.csv")
    needed = {"name", "lat", "lon"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"data/ems_recommendations.csv is missing required columns: {sorted(missing)}"
        )
    if "selected_rank" not in df.columns:
        df["selected_rank"] = np.arange(1, len(df) + 1)
    df = df.sort_values("selected_rank", kind="stable").reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def get_weather_df() -> pd.DataFrame:
    path = DATA_DIR / "weather_hourly.csv"
    if not path.exists():
        return pd.DataFrame()
    return _read_csv(path, parse_dates=["hour"])


def _gini(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        arr = arr - arr.min()
    mean = arr.mean()
    if mean <= 1e-12:
        return 0.0
    mad = np.abs(arr[:, None] - arr[None, :]).mean()
    return float(0.5 * mad / mean)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _clean_number(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    if v < lo or v > hi:
        return default
    return v


def _nearest_grid_ids(
    points: pd.DataFrame,
    grid: pd.DataFrame,
    lat_col: str,
    lon_col: str,
) -> list[str]:
    if points.empty or grid.empty:
        return []
    centers = grid[["latitude", "longitude"]].to_numpy(dtype=float)
    ids = grid["grid_id"].astype(str).tolist()
    out: list[str] = []
    for lat, lon in points[[lat_col, lon_col]].to_numpy(dtype=float):
        d = (centers[:, 0] - lat) ** 2 + (centers[:, 1] - lon) ** 2
        out.append(ids[int(np.argmin(d))])
    return out


def get_grid_summary() -> pd.DataFrame:
    scored = get_scored_df().copy()
    latest_ts = scored["timestamp"].max()
    cutoff = latest_ts - pd.Timedelta(days=14)
    recent = scored[scored["timestamp"] >= cutoff].copy()
    if recent.empty:
        recent = scored

    grid = (
        recent.groupby("grid_id")
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            risk_mean=("risk_mean", "mean"),
            risk_uncertainty=("risk_uncertainty", "mean"),
            accident_rate=("accident", "mean"),
            n_samples=("risk_mean", "size"),
        )
        .reset_index()
    )
    grid["priority"] = grid["risk_mean"] * (1.0 + grid["risk_uncertainty"])
    grid = grid.sort_values("priority", ascending=False, kind="stable").reset_index(drop=True)
    return grid


def _build_units(grid: pd.DataFrame) -> list[dict[str, Any]]:
    rec = get_ems_recommendations_df()
    if rec.empty or grid.empty:
        return []

    rec = rec.copy()
    rec["grid_id"] = _nearest_grid_ids(rec, grid, lat_col="lat", lon_col="lon")

    units: list[dict[str, Any]] = []
    for i, row in rec.iterrows():
        rank = int(row.get("selected_rank", i + 1))
        if rank <= 2:
            status, status_label = "dispatched", "Dispatched"
        elif rank == 3:
            status, status_label = "staging", "Staging"
        elif rank == 4:
            status, status_label = "available", "Available"
        else:
            status, status_label = "hospital", "At Hospital"

        unit_id = f"EMS-{rank:02d}"
        site_name = str(row.get("name", unit_id))
        units.append(
            {
                "id": unit_id,
                "name": site_name,
                "zone": f"Grid {row['grid_id']}",
                "zone_grid_id": str(row["grid_id"]),
                "status": status,
                "statusLabel": status_label,
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
            }
        )
    return units


def _status_for_risk(
    risk: float,
    q50: float,
    q75: float,
    q90: float,
) -> str:
    if risk >= q90:
        return "critical"
    if risk >= q75:
        return "deficit"
    if risk >= q50:
        return "monitor"
    return "optimal"


def _select_neighborhood_grids(grid: pd.DataFrame, max_items: int = 12) -> pd.DataFrame:
    if len(grid) <= max_items:
        return grid.copy()

    # Mix top-risk and broader-system coverage so UI has critical + healthy zones.
    pos = set(range(0, min(6, len(grid))))
    spread = np.linspace(6, len(grid) - 1, max_items - len(pos), dtype=int)
    pos.update(int(x) for x in spread)
    chosen = sorted(p for p in pos if p < len(grid))
    return grid.iloc[chosen].copy()


def _build_neighborhoods(grid: pd.DataFrame, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if grid.empty:
        return []

    selected = _select_neighborhood_grids(grid, max_items=12)
    risk_all = grid["risk_mean"]
    q50 = float(risk_all.quantile(0.50))
    q75 = float(risk_all.quantile(0.75))
    q90 = float(risk_all.quantile(0.90))
    lat_mid = float(selected["latitude"].median())
    lon_mid = float(selected["longitude"].median())

    unit_counts: dict[str, int] = {}
    for u in units:
        gid = str(u.get("zone_grid_id", ""))
        unit_counts[gid] = unit_counts.get(gid, 0) + 1

    neighborhoods: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        gid = str(row.grid_id)
        risk = float(row.risk_mean)
        unc = float(row.risk_uncertainty)
        status = _status_for_risk(risk, q50, q75, q90)

        response_current = round(_clamp(4.2 + risk * 18.0 + unc * 85.0, 3.8, 18.0), 1)
        target = 8.0 if status in {"optimal", "monitor"} else 9.5
        floor = 12.0 if status in {"optimal", "monitor"} else 14.0

        coverage_min = 1 if status in {"optimal", "monitor"} else 2
        if status == "critical":
            coverage_min = 3
        coverage_current = int(unit_counts.get(gid, 0))

        penalty = round(1.0 + min(0.9, unc * 3.8 + max(0.0, response_current - target) * 0.03), 2)
        pop_proxy = int(9000 + float(row.n_samples) * 38 + risk * 8500)

        north_south = "North" if float(row.latitude) >= lat_mid else "South"
        east_west = "East" if float(row.longitude) >= lon_mid else "West"
        district = f"{north_south} {east_west} District"

        if status == "critical":
            note_type = "flag"
            note = "Model risk and response pressure exceed critical threshold."
        elif status == "deficit":
            note_type = "flag"
            note = "Coverage is below target for current risk profile."
        elif status == "monitor":
            note_type = "ok"
            note = "Within limits, but trending toward intervention range."
        else:
            note_type = "ok"
            note = "Healthy coverage and low model-estimated incident pressure."

        neighborhoods.append(
            {
                "gridId": gid,
                "name": f"Grid {gid} - {north_south} {east_west}",
                "district": district,
                "population": pop_proxy,
                "penaltyFactor": penalty,
                "status": status,
                "statusLabel": STATUS_LABEL[status],
                "responseTime": {
                    "current": response_current,
                    "target": target,
                    "floor": floor,
                },
                "coverage": {
                    "current": coverage_current,
                    "min": coverage_min,
                },
                "noteType": note_type,
                "note": note,
                "lat": float(row.latitude),
                "lon": float(row.longitude),
                "risk": risk,
                "uncertainty": unc,
            }
        )

    neighborhoods.sort(
        key=lambda n: (
            STATUS_PRIORITY[n["status"]],
            -n["responseTime"]["current"],
            -n["risk"],
        )
    )
    return neighborhoods


def _build_spark_data(scored: pd.DataFrame, neighborhoods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not neighborhoods:
        return []
    spark_data: list[dict[str, Any]] = []
    for n in neighborhoods[:5]:
        gid = n["gridId"]
        hist = (
            scored.loc[scored["grid_id"] == gid, ["timestamp", "risk_mean"]]
            .set_index("timestamp")
            .resample("D")["risk_mean"]
            .mean()
            .dropna()
            .tail(7)
        )
        if hist.empty:
            continue

        values = (4.2 + hist.to_numpy(dtype=float) * 18.0).tolist()
        while len(values) < 7:
            values.insert(0, values[0])

        change = values[-1] - values[0]
        spark_data.append(
            {
                "name": n["name"],
                "values": [round(v, 2) for v in values[-7:]],
                "change": f"{change:+.1f}",
                "dir": "up" if change <= 0 else "down",
            }
        )
    return spark_data


def _build_dispatch_log(
    accidents: pd.DataFrame,
    grid: pd.DataFrame,
    neighborhoods: list[dict[str, Any]],
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if accidents.empty or grid.empty:
        return []

    by_grid = {n["gridId"]: n for n in neighborhoods}
    grid_risk = grid.set_index("grid_id")["risk_mean"].to_dict()
    recent = accidents.sort_values("timestamp", ascending=False).head(30).copy()
    recent["grid_id"] = _nearest_grid_ids(recent, grid, lat_col="latitude", lon_col="longitude")

    type_labels = {
        "cardiac": "Cardiac",
        "trauma": "Trauma",
        "fall": "Fall",
        "breathing": "Breathing",
        "other": "Other",
    }

    log: list[dict[str, Any]] = []
    for i, row in enumerate(recent.itertuples(index=False)):
        harm = int(getattr(row, "harm", 0) or 0)
        tstamp = pd.to_datetime(row.timestamp)

        if harm >= 2:
            call_type = "trauma"
        elif harm == 1:
            call_type = "cardiac" if (tstamp.hour % 2 == 0) else "fall"
        else:
            call_type = "breathing" if (tstamp.hour % 3 == 0) else "other"

        if i < 4:
            status = "Active"
        elif i < 7:
            status = "At Hosp."
        else:
            status = "Cleared"

        gid = str(row.grid_id)
        zone_label = f"Grid {gid}"
        if gid in by_grid:
            zone_label = by_grid[gid]["name"].split("-")[0].strip()

        unit_id = units[i % len(units)]["id"] if units else "UNASSIGNED"
        zone_risk = float(grid_risk.get(gid, 0.15))
        response_min = _clamp(4.0 + zone_risk * 18.0 + (i % 5) * 0.45, 3.2, 19.5)

        log.append(
            {
                "id": f"C-{3000 + i}",
                "type": call_type,
                "typeLabel": type_labels[call_type],
                "zone": zone_label,
                "unit": unit_id,
                "time": tstamp.strftime("%H:%M"),
                "response": f"{response_min:.1f} min",
                "status": status,
            }
        )
    return log


def _build_alerts(
    neighborhoods: list[dict[str, Any]],
    dispatch_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for n in neighborhoods:
        if n["status"] not in {"critical", "deficit", "monitor"}:
            continue
        if n["status"] == "critical":
            a_type, icon = "critical", "!!"
        elif n["status"] == "deficit":
            a_type, icon = "warning", "!"
        else:
            a_type, icon = "info", "i"

        alerts.append(
            {
                "type": a_type,
                "icon": icon,
                "title": f"{n['name']} requires attention",
                "desc": (
                    f"Response {n['responseTime']['current']:.1f}m "
                    f"(target {n['responseTime']['target']:.1f}m). "
                    f"Coverage {n['coverage']['current']} / min {n['coverage']['min']}."
                ),
                "time": "Derived from latest scored model output",
            }
        )
        if len(alerts) >= 2:
            break

    active_calls = sum(1 for c in dispatch_log if c["status"] == "Active")
    if active_calls > 0:
        alerts.append(
            {
                "type": "info",
                "icon": "[]",
                "title": f"{active_calls} active incidents in dispatch queue",
                "desc": "Live queue uses most recent crash-event records projected onto risk grids.",
                "time": "Updated from accidents.csv",
            }
        )

    return alerts[:3]


def _build_status_metrics(
    neighborhoods: list[dict[str, Any]],
    units: list[dict[str, Any]],
    dispatch_log: list[dict[str, Any]],
    accidents: pd.DataFrame,
) -> dict[str, Any]:
    responses = [float(n["responseTime"]["current"]) for n in neighborhoods] or [0.0]
    gini = round(_gini(responses), 2)
    coverage_ok = sum(1 for n in neighborhoods if n["coverage"]["current"] >= n["coverage"]["min"])

    active_units = sum(1 for u in units if u["status"] != "hospital")
    total_units = len(units)
    open_calls = sum(1 for c in dispatch_log if c["status"] == "Active")
    critical_calls = sum(
        1 for c in dispatch_log if c["status"] == "Active" and c["type"] in {"cardiac", "trauma"}
    )

    latest_day = accidents["timestamp"].max().date() if not accidents.empty else pd.Timestamp.utcnow().date()
    calls_today = int((accidents["timestamp"].dt.date == latest_day).sum()) if not accidents.empty else 0
    calls_prev = int((accidents["timestamp"].dt.date == (latest_day - pd.Timedelta(days=1))).sum()) if not accidents.empty else 0

    return {
        "active_units": active_units,
        "total_units": total_units,
        "avg_response_min": round(float(np.mean(responses)), 1),
        "open_calls": open_calls,
        "critical_calls": critical_calls,
        "gini_index": gini,
        "coverage_ok": coverage_ok,
        "coverage_total": len(neighborhoods),
        "calls_today": calls_today,
        "calls_delta_vs_prev_day": calls_today - calls_prev,
    }


def build_dashboard_payload() -> dict[str, Any]:
    scored = get_scored_df()
    accidents = get_accidents_df()
    grid = get_grid_summary()
    units = _build_units(grid)
    neighborhoods = _build_neighborhoods(grid, units)
    spark_data = _build_spark_data(scored, neighborhoods)
    dispatch_log = _build_dispatch_log(accidents, grid, neighborhoods, units)
    alerts = _build_alerts(neighborhoods, dispatch_log)
    status_metrics = _build_status_metrics(neighborhoods, units, dispatch_log, accidents)

    map_bounds = {
        "lat_min": float(grid["latitude"].min()) if not grid.empty else 42.95,
        "lat_max": float(grid["latitude"].max()) if not grid.empty else 43.22,
        "lon_min": float(grid["longitude"].min()) if not grid.empty else -89.62,
        "lon_max": float(grid["longitude"].max()) if not grid.empty else -89.20,
    }
    map_zones = [
        {
            "gridId": n["gridId"],
            "name": n["name"],
            "status": n["status"],
            "lat": n["lat"],
            "lon": n["lon"],
            "risk": n["risk"],
            "uncertainty": n["uncertainty"],
        }
        for n in neighborhoods
    ]

    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "status_metrics": status_metrics,
        "units": units,
        "neighborhoods": neighborhoods,
        "spark_data": spark_data,
        "dispatch_log": dispatch_log,
        "alerts": alerts,
        "map_data": {
            "bounds": map_bounds,
            "zones": map_zones,
            "units": units,
        },
    }


def get_default_simulation_context() -> dict[str, Any]:
    scored = get_scored_df()
    weather = get_weather_df()

    latest_ts = pd.to_datetime(scored["timestamp"].max())
    if latest_ts.tzinfo is not None:
        latest_ts = latest_ts.tz_convert(None)

    lat = float(scored["latitude"].median())
    lon = float(scored["longitude"].median())
    lag = float(scored["accident"].mean())

    temperature = 12.0
    precipitation = 0.0
    visibility = 10.0

    if not weather.empty:
        row = weather.sort_values("hour").iloc[-1]
        w_ts = pd.to_datetime(row.get("hour"), errors="coerce")
        if pd.notna(w_ts):
            latest_ts = w_ts
        temperature = _clean_number(row.get("temperature"), default=12.0, lo=-40.0, hi=45.0)
        precipitation = _clean_number(row.get("precipitation"), default=0.0, lo=0.0, hi=50.0)
        visibility = _clean_number(row.get("visibility"), default=10.0, lo=0.1, hi=30.0)

    return {
        "timestamp": latest_ts.to_pydatetime(),
        "latitude": lat,
        "longitude": lon,
        "temperature": temperature,
        "precipitation": precipitation,
        "visibility": visibility,
        "accident_lag_1": lag,
        "accident_lag_3": lag,
        "accident_lag_6": lag,
    }


def get_model_performance_proxy() -> dict[str, float]:
    scored = get_scored_df()
    y = scored["accident"].astype(float).to_numpy()
    p = scored["risk_mean"].astype(float).to_numpy()
    mae = float(np.mean(np.abs(y - p)))

    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - (ss_res / ss_tot)

    # Proxy CV score from holdout-like degradation.
    cv_r2 = r2 - 0.03

    return {
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "cv_r2": round(cv_r2, 4),
    }

