import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text

from app.config import DATABASE_URL, TZ

engine = create_engine(DATABASE_URL)

_SQL = text("""
    SELECT start_time AT TIME ZONE 'UTC' AT TIME ZONE :tz AS start_time,
           value
    FROM health_samples
    WHERE user_id     = :user_id
      AND sample_type = 'heart_rate'
      AND start_time  >= :start_time
    ORDER BY start_time
""")


def get_heart_rate_data(user_id: str, start_time: datetime) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(
            _SQL,
            conn,
            params={"user_id": user_id, "start_time": start_time, "tz": TZ},
        )
    df = df.rename(columns={"start_time": "timestamp", "value": "bpm"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if df["timestamp"].dt.tz is None:
       df["timestamp"] = df["timestamp"].dt.tz_localize(TZ)
    else:
       df["timestamp"] = df["timestamp"].dt.tz_convert(TZ)
       df["bpm"] = df["bpm"].astype(float)
    return df
