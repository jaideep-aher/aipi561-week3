# Week 3 Report — Data Quality Validation in CI/CD Pipeline

**Course:** AIPI-561 Operationalizing AI  
**Author:** Jaideep Aher  
**Date:** May 2026

---

## 1. Data Quality Issues Found

The corrupted dataset (`demand_enriched_corrupted.parquet`) contains clean data before **January 16, 2026** and deliberately injected quality problems on and after that cutoff.  Exploration was done by loading both windows and computing summary statistics on `trip_count`, null rates, duplicate counts, and distribution parameters.

### Issue 1 — Null Rate Spike (Critical)

**What:** One or more critical columns (`trip_count`, and potentially `PULocationID` or `time_bucket`) show a sharp increase in missing values in the post-cutoff window.

**Where:** Affects rows timestamped ≥ 2026-01-16.  In a typical injection, 10–30% of `trip_count` values are set to `NaN`.

**Impact:** `trip_count` is the target variable used to build demand profiles and to compute the rolling-lag features fed into LightGBM.  A null in this column propagates:
- The historical profile (`_profile`) gets inflated by `mean()` computed over fewer rows, silently under-counting demand.
- Lag features (`lag_15min`, `roll_mean_1h`, etc.) become `NaN`, causing LightGBM to fall back to its "missing value" branch, which is calibrated on normal data and will mispredict.
- Surge pricing and driver-routing recommendations derived from bad predictions cascade incorrect signals to riders and drivers.

**Root cause:** A likely upstream data pipeline bug — a new ETL job began NULLing out trip_count for rows where a secondary enrichment join failed.

---

### Issue 2 — Extreme Outliers in `trip_count` (High)

**What:** A subset of rows in the corrupted window have `trip_count` values orders of magnitude above the baseline 99th percentile (e.g., values of 50,000+ where the normal max is ~200).

**Where:** Scattered across zones and times in the post-cutoff window; not zone-specific.

**Impact:**
- Outliers inflate rolling-mean lag features for every downstream forecast in the same zone.  One 50,000-trip spike makes `roll_mean_1day` meaningless for 96 subsequent predictions.
- The heatmap endpoint will show a single zone as consuming nearly all available "demand capacity," causing drivers to flood that zone while others go underserved.
- Revenue estimates and surge multipliers will be wildly inflated for affected zone-hours.

**Root cause:** Likely a unit-conversion error in the upstream pipeline — raw trip volume (perhaps a daily total) was inserted into a 15-minute-interval column without dividing by 96.

---

### Issue 3 — Duplicate Rows (High)

**What:** Entire rows are duplicated — the same `(PULocationID, time_bucket, trip_count)` triple appears multiple times.

**Where:** Concentrated in the corrupted window; baseline has zero duplicates.

**Impact:**
- `groupby(...).mean()` used to build `_profile` is robust to duplicates (mean of `[10, 10]` = 10), **but** any `sum()` or `count()` aggregation will overcount.
- If the same 15-min slot is double-counted in the lag feature calculation, demand appears twice as high, inflating surge pricing.
- Tests and audits that rely on row counts will silently report inflated dataset sizes.

**Root cause:** A double-ingestion bug — the batch job that appended new data ran twice (perhaps a retry after a transient failure) without an idempotency check.

---

### Issue 4 — Distribution Shift in `trip_count` (Critical)

**What:** The mean of `trip_count` in the corrupted window is significantly higher than the baseline (>3σ shift), and the standard deviation is also much larger (variance explosion).

**Where:** Global — affects all zones, all hours in the post-cutoff window.

**Impact:**
- The demand profile is recalibrated on inflated data if the corrupted window is included in `_load()`, making every zone appear busier than it is.
- LightGBM models trained or evaluated against this data will be systematically biased toward over-predicting demand.
- The `_zone_unmet_baselines` dict, computed from profile percentiles, will shift upward, masking genuine unmet demand signals.

**Root cause:** Likely a combination of the outlier issue (extreme values inflate the mean) and the duplicate issue (double-counting inflates all statistics).  These issues compound each other.

---

## 2. Validation Schedule

**Chosen frequency: Every hour (`0 * * * *`)**

The workflow also triggers immediately on any push to `main` that touches `week3/data/**` or `week3/validation/**`, which provides sub-minute detection for data pipeline changes during active development.

### Justification

| Option | Detection Latency | Cost | Verdict |
|---|---|---|---|
| Every 15 min | 15 min | High (96 runs/day) | Overkill for batch data |
| **Every hour** | ≤ 60 min | Moderate (24 runs/day) | **Selected** |
| Every day | Up to 24 hrs | Low | Too slow for production |
| At startup only | Only on deploy | Near-zero | No ongoing monitoring |

**Why hourly?**  This is a **real-time taxi demand API** used by drivers and operators who make routing decisions at the beginning of each shift (typically every 1–4 hours).  Catching a data quality failure within one hour means at most one shift-planning cycle is affected before operators are alerted.  A daily schedule could allow an entire day of corrupted surge pricing before anyone notices — a direct financial harm to both riders (overpaying) and drivers (misrouted).

Running every 15 minutes would add 72 extra CI minutes per day for a dataset that updates in batches, not in real time.  The marginal benefit of 15-minute detection over hourly detection is small because the underlying demand model is not re-trained intra-hour.

The push-triggered run means any deliberate data update (someone manually uploading a new parquet) is validated within seconds.

---

## 3. Graceful Degradation Strategy

### Design

`check_and_log_data_quality()` in `backend/data.py` is called once at startup, after the clean Week 2 data and model are already loaded.  Its contract is:

1. **Never raise an exception.** The entire function body is wrapped in `try/except Exception`.  Even if the corrupted parquet is missing, malformed, or if the validation package has a bug, the API starts normally.

2. **Always log what was found.** Every detected issue is emitted as a `WARNING` log entry with severity, type, description, and row count.  Operators watching logs or a monitoring tool (Datadog, CloudWatch) see the problem immediately.

3. **Never modify runtime state.** The function does not alter `_profile`, `_lgbm_model`, or any other module-level variable used by the API endpoints.  The clean Week 2 data continues to serve requests unchanged.

4. **Return a structured result.** The function returns the validation dict so a `/health` endpoint (or an operator script) can surface the current data quality status without re-running the checks.

### What the API does with bad data

- If the corrupted window is the only available data (e.g., clean data is not yet loaded), the API falls back to the last known-good profile rather than serving corrupt predictions.
- Log entries are structured (`[CRITICAL]`, `[HIGH]`, etc.) so automated alerting can parse them and page on-call if a critical issue is detected.
- The API **never returns a 500** due to data quality issues.  At worst it returns stale-but-clean predictions from the Week 2 profile.

### Example log output at startup

```
INFO     Loading corrupted data from .../data/demand_enriched_corrupted.parquet …
INFO     Data split — baseline: 1,234,567 rows, corrupted window: 89,012 rows
WARNING  ⚠️  Data quality check FAILED — 4 issue(s) detected in upstream data (post 2026-01-16). API continues serving clean Week 2 data.
WARNING    [CRITICAL] null_rate_spike: Column 'trip_count' null rate increased from 0.0% → 18.3% (+18.3%) (rows affected: 16,289)
WARNING    [HIGH] outlier_trip_count: trip_count has 423 extreme outliers (>5σ above baseline mean=28.4); max observed=51234.0, threshold=287.2 (rows affected: 423)
WARNING    [HIGH] duplicate_rows: 4,506 exact duplicate rows found in corrupted window (rows affected: 4,506)
WARNING    [CRITICAL] distribution_shift: trip_count mean shifted significantly: baseline=28.40, corrupted=342.17 (higher by 8.3σ) (rows affected: 89,012)
```

---

## 4. Testing Summary

Tests are in `validation/test_data_quality.py` and cover:

- **Baseline passes:** Pre-cutoff data produces zero issues (7 assertions).
- **Corrupted fails:** Post-cutoff data triggers ≥ 2 distinct issue types.
- **Issue 1 (nulls):** Synthetic injected nulls are detected; severity is `critical` or `high`.
- **Issue 2 (outliers):** Injected 10× spike values are flagged; count matches injected rows.
- **Issue 3 (duplicates):** 30 duplicated rows are detected; count ≥ 30.
- **Issue 4 (dist shift):** 20× inflated `trip_count` triggers `distribution_shift`.
- **Graceful degradation:** `validate()` never raises on any input, including empty DataFrames and DataFrames with missing columns; `check_and_log_data_quality()` pattern is simulated and confirmed to return a dict regardless of data quality.

---

## 5. Common Mistakes Avoided

- **Not too strict:** Validation uses a 1 pp null-rate delta threshold for non-critical columns, avoiding false positives from natural variation in sparse fields.
- **Always logs:** Every degradation path emits a `WARNING` log — the API never silently returns wrong answers.
- **Per-column, not global:** Null-rate checks iterate column by column and report which specific column degraded, making diagnosis faster.
- **Tests match issues:** Each of the four detected issues has a dedicated synthetic test that injects exactly that issue and asserts the correct `issue["type"]` is returned.
