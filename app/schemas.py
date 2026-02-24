from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class SummaryMetrics(BaseModel):
    rmssd_mean: Optional[float] = None
    sdnn_mean: Optional[float] = None
    mean_hr: Optional[float] = None
    lf_hf_ratio_mean: Optional[float] = None


class TimeBucket(BaseModel):
    bucket: str
    rmssd: Optional[float] = None
    sdnn: Optional[float] = None
    mean_hr: Optional[float] = None
    lf_hf_ratio: Optional[float] = None


class HRVResponse(BaseModel):
    user_id: str
    range: str
    generated_at: datetime
    summary_metrics: SummaryMetrics
    time_series: List[TimeBucket]
    patterns: Optional[dict] = None
