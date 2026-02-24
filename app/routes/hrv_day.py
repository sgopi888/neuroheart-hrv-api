from datetime import datetime, timedelta
import zoneinfo

import pandas as pd
from fastapi import APIRouter, Header, HTTPException, Query

from app.config import API_KEY, TZ
from app.db import get_heart_rate_data
from app.hrv_features import HRVFeatureExtractor
from app.hrv_processor import HRVProcessor

router = APIRouter()

_user_tz = zoneinfo.ZoneInfo(TZ)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        return None if (f != f) else f
    except (TypeError, ValueError):
        return None


def _parse_date(date_str: str) -> datetime:
    """Parse YYYY-MM-DD → tz-aware datetime in user TZ. Raises HTTPException on bad input."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=_user_tz)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date '{date_str}'. Use YYYY-MM-DD.")


def _hourly_hrv_for_window(df_raw: pd.DataFrame) -> list[dict]:
    """
    Given a raw heart-rate DataFrame (timestamp TZ-aware, bpm float),
    run the full HRV pipeline and return a list of hourly metric dicts.
    Returns an empty list if there is insufficient data.
    """
    if df_raw.empty:
        return []

    df_windows = HRVProcessor().process(df_raw)
    if df_windows.empty:
        return []

    df_hrv = HRVFeatureExtractor().process_windows(df_windows)
    if df_hrv.empty:
        return []

    df_hrv["timestamp"] = pd.to_datetime(df_hrv["timestamp"])
    if df_hrv["timestamp"].dt.tz is None:
        df_hrv["timestamp"] = df_hrv["timestamp"].dt.tz_localize(TZ)

    df_hrv["hour"] = df_hrv["timestamp"].dt.hour

    hourly = []
    for hr in sorted(df_hrv["hour"].unique()):
        block = df_hrv[df_hrv["hour"] == hr]
        hourly.append({
            "hour":        int(hr),
            "rmssd":       _safe_float(block["rmssd"].mean()),
            "sdnn":        _safe_float(block["sdnn"].mean()),
            "mean_hr":     _safe_float(block["mean_hr"].mean()),
            "lf_hf_ratio": _safe_float(block["lf_hf_ratio"].mean()),
        })
    return hourly


# ---------------------------------------------------------------------------
# GET /v1/hrv/day  — single day, hourly HRV
# ---------------------------------------------------------------------------

@router.get("/v1/hrv/day")
def hrv_by_day(
    user_id: str = Query(..., description="User UUID"),
    date: str = Query(..., description="YYYY-MM-DD in user timezone"),
    x_api_key: str = Header(..., alias="x-api-key"),
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")

    start_local = _parse_date(date)
    end_local = start_local + timedelta(days=1)

    df = get_heart_rate_data(user_id, start_local)
    if df.empty:
        raise HTTPException(status_code=404, detail="No heart rate data for this day.")

    df = df[(df["timestamp"] >= start_local) & (df["timestamp"] < end_local)]
    if df.empty:
        raise HTTPException(status_code=404, detail="No samples inside specified day.")

    hourly_results = _hourly_hrv_for_window(df)
    if not hourly_results:
        raise HTTPException(status_code=404, detail="Insufficient HR data for hourly HRV.")

    return {
        "user_id":         user_id,
        "date":            date,
        "hours_available": len(hourly_results),
        "hourly":          hourly_results,
    }


# ---------------------------------------------------------------------------
# GET /v1/hrv/range  — date range, hourly HRV per day
# ---------------------------------------------------------------------------

@router.get("/v1/hrv/range")
def hrv_by_range(
    user_id: str = Query(..., description="User UUID"),
    start_date: str = Query(..., description="Start date YYYY-MM-DD (inclusive)"),
    end_date: str = Query(..., description="End date YYYY-MM-DD (inclusive)"),
    x_api_key: str = Header(..., alias="x-api-key"),
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")

    range_start = _parse_date(start_date)
    range_end_inclusive = _parse_date(end_date)

    if range_end_inclusive < range_start:
        raise HTTPException(status_code=400, detail="end_date must be >= start_date.")

    total_days = (range_end_inclusive - range_start).days + 1
    if total_days > 366:
        raise HTTPException(status_code=400, detail="Date range cannot exceed 366 days.")

    # Single DB fetch for the entire range — avoids N round-trips
    range_end_exclusive = range_end_inclusive + timedelta(days=1)
    df_all = get_heart_rate_data(user_id, range_start)

    if df_all.empty:
        raise HTTPException(status_code=404, detail="No heart rate data for selected range.")

    df_all = df_all[
        (df_all["timestamp"] >= range_start) & (df_all["timestamp"] < range_end_exclusive)
    ]
    if df_all.empty:
        raise HTTPException(status_code=404, detail="No samples inside specified date range.")

    # Process each day
    days_output = []
    current = range_start
    while current < range_end_exclusive:
        next_day = current + timedelta(days=1)
        df_day = df_all[(df_all["timestamp"] >= current) & (df_all["timestamp"] < next_day)]

        day_str = current.strftime("%Y-%m-%d")
        hourly = _hourly_hrv_for_window(df_day)

        days_output.append({
            "date":            day_str,
            "hours_available": len(hourly),
            "hourly":          hourly,
        })
        current = next_day

    days_with_data = sum(1 for d in days_output if d["hours_available"] > 0)

    return {
        "user_id":        user_id,
        "start_date":     start_date,
        "end_date":       end_date,
        "total_days":     total_days,
        "days_with_data": days_with_data,
        "days":           days_output,
    }
