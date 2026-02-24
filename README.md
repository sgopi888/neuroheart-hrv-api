# NeuroHeart HRV API

FastAPI service that reads heart rate data from PostgreSQL and returns HRV analysis.

## Local Dev Setup

```bash
git clone https://github.com/sgopi888/neuroheart-hrv-api.git
cd neuroheart-hrv-api

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set DATABASE_URL and API_KEY

uvicorn app.main:app --reload --port 8002
```

## Test Commands

```bash
# Health check
curl http://localhost:8002/health

# HRV analysis (replace values)
curl -H "x-api-key: YOUR_API_KEY" \
  "http://localhost:8002/v1/hrv/analysis?user_id=USER_UUID&range=7d"
```

## API Reference

### GET /health

Returns `{"status": "ok"}`.

### GET /v1/hrv/analysis

**Query params:**
- `user_id` — user UUID (required)
- `range` — one of `1d`, `7d`, `30d`, `6m` (required)

**Header:**
- `x-api-key` — API key from `.env`

**Response:**
```json
{
  "user_id": "...",
  "range": "7d",
  "generated_at": "2026-02-24T12:00:00+00:00",
  "summary_metrics": {
    "rmssd_mean": 42.1,
    "sdnn_mean": 55.3,
    "mean_hr": 68.2,
    "lf_hf_ratio_mean": 1.4
  },
  "time_series": [
    {"bucket": "2026-02-17", "rmssd": 40.1, "sdnn": 52.0, "mean_hr": 69.0, "lf_hf_ratio": 1.5}
  ],
  "patterns": { "..." }
}
```

`patterns` is included for `7d`, `30d`, `6m`. It is `null` for `1d`.

## Deploy (server)

After pushing code:

```bash
cd /opt/neuroheart/hrv-api
./deploy.sh
```
