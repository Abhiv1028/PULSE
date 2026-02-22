from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_serves_frontend() -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "PULSE" in res.text


def test_dashboard_bootstrap_payload() -> None:
    res = client.get("/api/dashboard/bootstrap")
    assert res.status_code == 200
    body = res.json()

    assert "summary" in body
    assert "units" in body
    assert "neighborhoods" in body
    assert "dispatch" in body
    assert "optimizer" in body

    assert isinstance(body["units"], list)
    assert isinstance(body["neighborhoods"], list)
    assert len(body["units"]) > 0
    assert len(body["neighborhoods"]) > 0


def test_simulator_endpoint_returns_reduction() -> None:
    payload = {
        "policies": {
            "speed_limit_reduction": 8,
            "street_lighting_improvement": 6,
        }
    }
    res = client.post("/api/simulate", json=payload)
    assert res.status_code == 200

    body = res.json()
    assert body["risk_reduction_pct"] > 0
    assert body["predicted_risk"] < body["baseline_risk"]
    assert "explanation" in body


def test_predict_and_explain_fallback_paths() -> None:
    payload = {
        "timestamp": "2026-02-22T13:00:00",
        "latitude": 43.0731,
        "longitude": -89.4012,
        "temperature": 30.0,
        "precipitation": 0.2,
        "visibility": 4.0,
        "accident_lag_1": 0.0,
        "accident_lag_3": 1.0,
        "accident_lag_6": 1.0,
    }

    pred = client.post("/predict", json=payload)
    assert pred.status_code == 200
    pred_body = pred.json()
    assert 0 <= pred_body["risk_mean"] <= 1
    assert pred_body["risk_uncertainty"] >= 0

    exp = client.post("/explain", json=payload)
    assert exp.status_code == 200
    exp_body = exp.json()
    assert "top_features" in exp_body
    assert len(exp_body["top_features"]) > 0

