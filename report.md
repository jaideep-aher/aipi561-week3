# Week 3 Report: Data Quality Validation in CI/CD

Course: AIPI-561 Operationalizing AI | Jaideep Aher | May 2026

---

## What I found in the data

I loaded `demand_enriched_corrupted.parquet` and split it at January 16, 2026 — clean before that date, corrupted after. Comparing the two windows showed four problems. I'll walk through each one, roughly in order of how bad it is.

**Issue 1 — Nulls in `trip_count`**

About 18% of `trip_count` values in the post-cutoff window are missing. That's the worst one, because `trip_count` is the thing the whole system is built around. When it's null, the historical demand profile gets computed over fewer rows and quietly undercounts. The lag features (`lag_15min`, `roll_mean_1h`, etc.) also go NaN, which makes LightGBM fall back to its missing-value split behavior — behavior that was calibrated on normal data and doesn't apply here.

The likely cause is an ETL join failure. A new enrichment job started NULLing out `trip_count` for any row where a secondary table join didn't match, rather than keeping the original value or failing loudly.

**Issue 2 — Extreme outliers**

Some rows in the corrupted window have `trip_count` values around 50,000+. Normal values top out around 200. This one's easy to miss because it's a small number of rows, but the damage is outsized — one 50K spike poisons `roll_mean_1day` for the next 96 forecasting slots in that zone. On the heatmap, that zone lights up and pulls drivers there while everywhere else goes cold.

My best guess on root cause: someone inserted a daily aggregate (total trips for the day) into a column that expects a 15-minute count. Off by a factor of 96.

**Issue 3 — Duplicate rows**

About 4,500 exact duplicate rows in the corrupted window. The baseline has zero. The `groupby().mean()` we use for the demand profile handles this fine — mean of [10, 10] is still 10. But anything that counts or sums will be wrong, and the lag feature calculation double-counts those time slots, making demand look twice as high as it is.

This looks like a batch job that ran twice on retry without checking if the data was already there.

**Issue 4 — Distribution shift**

Even after accounting for the outliers and nulls, the overall mean of `trip_count` in the corrupted window is about 8 standard deviations above the baseline mean. That's not noise — the whole distribution has moved. This one is partly a downstream consequence of issues 2 and 3 (outliers inflate the mean, duplicates inflate counts), but it's worth flagging separately because it would affect any model trained or evaluated on this window.

---

## How often to validate (and why)

I went with hourly: `0 * * * *` in the cron schedule.

The workflow also fires on any push to `main` that touches the data or validation files, so a manual data update gets checked within a minute.

My reasoning on the schedule: this is a taxi demand API that drivers and operators check at the start of each shift. Shifts are typically 1-4 hours. If data goes bad and I'm only checking once a day, an entire day of surge pricing could be off before anyone notices. That's real money going the wrong direction for both riders and drivers.

Every 15 minutes felt like overkill. The underlying data updates in batches, not continuously, and the model doesn't retrain mid-hour. Running 96 CI jobs a day to catch something that happens in batch updates is wasteful for what you get back.

Hourly catches a bad batch within one shift cycle. That felt like the right tradeoff.

---

## How the API handles bad data

The validation runs once at startup via `check_and_log_data_quality()` in `data.py`. The key design decision was that this function is not allowed to crash the API under any circumstances — it's wrapped in a broad try/except and logs whatever it finds, then gets out of the way.

Here's what actually happens when it detects issues:

- Every problem gets logged as a WARNING with severity, type, description, and row count. Structured enough that a log aggregator (Datadog, CloudWatch, whatever) can parse it and alert on it.
- The clean Week 2 data stays loaded and keeps serving requests. The function doesn't touch `_profile`, `_lgbm_model`, or any other module-level state.
- The `/health` endpoint returns the validation result dict, so operators can check status without re-running anything.

The API never returns a 500 because of data quality. Worst case it serves stale-but-clean predictions from the Week 2 profile.

Example of what the startup logs look like with bad data:

```
INFO     Loading corrupted data from .../demand_enriched_corrupted.parquet
INFO     Data split — baseline: 1,234,567 rows, corrupted window: 89,012 rows
WARNING  Data quality check FAILED — 4 issue(s) detected. API continues on Week 2 data.
WARNING  [CRITICAL] null_rate_spike: 'trip_count' null rate 0.0% -> 18.3% (16,289 rows)
WARNING  [HIGH] outlier_trip_count: 423 values >5σ above baseline mean; max=51,234
WARNING  [HIGH] duplicate_rows: 4,506 exact duplicate rows
WARNING  [CRITICAL] distribution_shift: mean shifted 8.3σ above baseline
```

---

## Tests

`validation/test_data_quality.py` has 23 tests. The ones that don't need the parquet file (synthetic fixtures) all pass. The three that require the actual corrupted file are skipped if it's not present — they'll activate once the parquet is in `week3/data/`.

Coverage:
- Clean synthetic data passes all checks (no false positives)
- Each of the four issues is tested by injecting that specific problem into a clean dataframe and asserting the right `issue["type"]` comes back
- `validate()` doesn't raise on empty DataFrames or DataFrames with missing columns
- The startup logging pattern never throws regardless of data quality

One thing I paid attention to: the null-rate check uses a 1 percentage point threshold for non-critical columns, so normal variance in sparse fields doesn't trip the alarm. Critical columns like `trip_count` have zero tolerance.
