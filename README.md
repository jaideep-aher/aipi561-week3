# Week 3: Data Quality Validation in CI/CD

AIPI-561 Operationalizing AI | Jaideep Aher

## What this is

Week 2 built a NYC taxi demand forecasting API. This week adds automated data quality validation to catch problems in upstream data before they silently corrupt predictions.

New data coming in after January 16, 2026 has four issues: null values in trip_count, extreme outliers, duplicate rows, and a distribution shift. The validation pipeline detects all four and logs them at startup. The API keeps serving clean predictions regardless.

## Repo structure

```
.
+-- .github/
|   +-- workflows/
|       +-- validate-data.yml   # CI pipeline, runs hourly + on push
+-- backend/
|   +-- data.py                 # loads data, runs validation at startup
|   +-- main.py                 # FastAPI app with /health endpoint
|   +-- requirements.txt
+-- validation/
|   +-- check_data_quality.py   # DataQualityValidator class
|   +-- test_data_quality.py    # 23 tests
|   +-- __init__.py
+-- report.md                   # issues found, schedule choice, degradation strategy
```

## How to run

```bash
pip install pandas numpy pyarrow pytest fastapi uvicorn lightgbm

# Run validation manually
cd week3
python -m validation.check_data_quality .

# Run tests
python -m pytest validation/test_data_quality.py -v

# Start API
cd backend
uvicorn main:app --reload
```

Tests that load the corrupted parquet are skipped if the file is not present at `data/demand_enriched_corrupted.parquet`. Everything else runs fine without it.

## Validation checks

Four checks run against the post-January 16 data window, compared to the clean baseline:

- **Null rate spike** -- trip_count goes from 0% nulls to 18%. Detected at zero tolerance for critical columns.
- **Outliers** -- some trip_count values are around 50,000 when normal is under 200. Flagged at 5 standard deviations above baseline mean.
- **Duplicate rows** -- about 4,500 exact duplicate rows in the corrupted window. Baseline has zero.
- **Distribution shift** -- mean trip_count is 8 standard deviations above baseline even after accounting for the other issues.

## CI/CD schedule

The workflow runs every hour (`0 * * * *`) and also fires on any push to main that touches data or validation files. Hourly catches a bad batch within one shift cycle without over-polling. Details in `report.md`.

## Graceful degradation

The API never crashes due to data quality. `check_and_log_data_quality()` in `data.py` runs at startup, logs everything it finds as structured warnings, and returns without touching any runtime state. The `/health` endpoint exposes the last validation result so operators can check status without rerunning anything.
