# PULSE

PULSE is a decision-support system for urban safety planning and EMS command operations.

This repository now includes:

- **FastAPI backend** for risk prediction, explainability, dashboard data, and policy simulation
- **Integrated web dashboard frontend** served directly from the API (`app/static/index.html`)
- **Data-driven operational views** (overview, equity, dispatch, optimizer, simulator)

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open: `http://localhost:8000`

## Core API endpoints

- `POST /predict` — risk forecast for a single point-in-time request
- `POST /explain` — top feature influences for a single forecast
- `GET /api/dashboard/bootstrap` — dashboard data payload (units, alerts, neighborhoods, dispatch, optimizer)
- `GET /api/baseline` — simulator baseline and model metadata
- `POST /api/simulate` — policy what-if simulation output

## Tests

```bash
python3 -m pytest -q
```
