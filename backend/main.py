"""Week 3 API — identical to Week 2, data.py now runs validation at startup."""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import data

app = FastAPI(title="NYC Cab Analytics API — Week 3", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Returns API health + data quality status from the last validation run."""
    dq = data._data_quality_result
    return {
        "status": "ok",
        "data_quality": {
            "is_valid": dq.get("is_valid", True),
            "num_issues": dq.get("num_issues", 0),
            "issues": dq.get("issues", []),
        },
    }


@app.get("/api/heatmap")
def heatmap(
    hour: int = Query(..., ge=0, le=23),
    dow: int = Query(..., ge=0, le=6),
    date: str = Query(...),
    holiday: str = Query("regular"),
):
    return data.get_heatmap(hour, dow, holiday, date)


@app.get("/api/forecast")
def forecast(
    zone_id: int = Query(...),
    hour: int = Query(..., ge=0, le=23),
    dow: int = Query(..., ge=0, le=6),
    date: str = Query(...),
    steps: int = Query(16, ge=1, le=96),
):
    return data.forecast_demand(zone_id, hour, dow, steps, date)


@app.get("/api/recommendations")
def recommendations(
    zone_id: int = Query(...),
    hour: int = Query(..., ge=0, le=23),
    dow: int = Query(..., ge=0, le=6),
    date: str = Query(...),
    n: int = Query(3, ge=1, le=10),
    holiday: str = Query("regular"),
):
    return data.get_recommendations(zone_id, hour, dow, n, holiday, date)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
