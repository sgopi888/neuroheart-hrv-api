from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
import zoneinfo

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.config import TZ
from app.db import get_heart_rate_data
from app.hrv_features import HRVFeatureExtractor
from app.hrv_processor import HRVProcessor
from app.weekly_analyzer import WeeklyAnalyzer

# Maps range string → (lookback_days, pandas_resample_freq, strftime_label_format)
_RANGE_CONFIG: dict[str, tuple[int, str, str]] = {
    "1d":  (1,   "h",     "%Y-%m-%d %H:00"),
    "7d":  (7,   "D",     "%Y-%m-%d"),
    "30d": (30,  "W-SUN", "Week of %Y-%m-%d"),
    "6m":  (180, "MS",    "%Y-%m"),
}

_HRV_COLS = ["rmssd", "sdnn", "mean_hr", "lf_hf_ratio"]


def _safe_float(val) -> Optional[float]:
    """Convert numpy scalar or NaN to Python float or None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (f != f) else f  # NaN check: NaN != NaN is True
    except (TypeError, ValueError):
        return None


def compute_hrv_for_range(user_id: str, range: str) -> dict:
    days, freq, label_fmt = _RANGE_CONFIG[range]

    user_tz = zoneinfo.ZoneInfo(TZ)

    # Compute lookback start in user TZ, then convert to UTC for DB query
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(user_tz)
    start_local = now_local - timedelta(days=days)
    start_utc = start_local.astimezone(timezone.utc)

    # Step 1: Fetch raw heart rate data (timestamps returned as TZ-aware)
    df_raw = get_heart_rate_data(user_id, start_utc)
    if df_raw.empty:
        raise HTTPException(status_code=404, detail="No heart rate data for selected range.")

    # Step 2: Window into 15-min IBI windows
    df_windows = HRVProcessor().process(df_raw)
    if df_windows.empty:
        raise HTTPException(status_code=404, detail="No heart rate data for selected range.")

    # Step 3: Extract HRV features per window
    df_hrv = HRVFeatureExtractor().process_windows(df_windows)
    if df_hrv.empty:
        raise HTTPException(status_code=404, detail="No heart rate data for selected range.")

    # Ensure timestamp is TZ-aware (inherit from df_windows if needed)
    df_hrv["timestamp"] = pd.to_datetime(df_hrv["timestamp"])
    if df_hrv["timestamp"].dt.tz is None:
        df_hrv["timestamp"] = df_hrv["timestamp"].dt.tz_localize(TZ)

    # Step 4: Resample — only numeric HRV columns to avoid aggregation errors
    df_numeric = df_hrv[["timestamp"] + _HRV_COLS].copy()
    df_numeric = df_numeric.set_index("timestamp")

    agg_df = (
        df_numeric
        .resample(freq)
        .mean()
        .dropna(how="all")
        .reset_index()
    )

    # Step 5: Build time_series from resampled index
    time_series = []
    for _, row in agg_df.iterrows():
        time_series.append({
            "bucket":      row["timestamp"].strftime(label_fmt),
            "rmssd":       _safe_float(row["rmssd"]),
            "sdnn":        _safe_float(row["sdnn"]),
            "mean_hr":     _safe_float(row["mean_hr"]),
            "lf_hf_ratio": _safe_float(row["lf_hf_ratio"]),
        })

    # Step 6: Summary metrics from aggregated buckets
    summary = {
        "rmssd_mean":       _safe_float(agg_df["rmssd"].mean()),
        "sdnn_mean":        _safe_float(agg_df["sdnn"].mean()),
        "mean_hr":          _safe_float(agg_df["mean_hr"].mean()),
        "lf_hf_ratio_mean": _safe_float(agg_df["lf_hf_ratio"].mean()),
    }

    # Step 7: Patterns only for ranges with multiple days
    patterns = None
    if range in ("7d", "30d", "6m"):
        df_hrv_reset = df_hrv.reset_index(drop=True)
        patterns = WeeklyAnalyzer().create_weekly_summary(df_hrv_reset)

    return {
        "user_id":         user_id,
        "range":           range,
        "generated_at":    now_utc.isoformat(),
        "summary_metrics": summary,
        "time_series":     time_series,
        "patterns":        patterns,
    }
