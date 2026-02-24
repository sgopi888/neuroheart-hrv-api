from datetime import datetime
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Query

from app.analysis import compute_hrv_for_range
from app.config import API_KEY
from app.schemas import HRVResponse, SummaryMetrics, TimeBucket

app = FastAPI(title="NeuroHeart HRV API", version="1.0.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/v1/hrv/analysis", response_model=HRVResponse)
def hrv_analysis(
    user_id: str = Query(..., description="User UUID"),
    range: Literal["1d", "7d", "30d", "6m"] = Query(..., description="Time range"),
    x_api_key: str = Header(..., alias="x-api-key", description="API key"),
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")

    result = compute_hrv_for_range(user_id, range)

    return HRVResponse(
        user_id=result["user_id"],
        range=result["range"],
        generated_at=datetime.fromisoformat(result["generated_at"]),
        summary_metrics=SummaryMetrics(**result["summary_metrics"]),
        time_series=[TimeBucket(**b) for b in result["time_series"]],
        patterns=result["patterns"],
    )
