"""
Week 3 — data.py
Builds on Week 2 backend with added graceful data-quality validation.

New in Week 3:
  • check_and_log_data_quality() — runs at startup, logs all issues, NEVER crashes the API
  • Uses validation.check_data_quality.DataQualityValidator
  • Corrupted parquet is validated and reported; the API continues with Week 2 clean data

Startup sequence (bottom of file):
  _profile, _zones_df = _load()          # Week 2 clean data
  _lgbm_model        = _load_model()
  _full_demand       = _load_full_demand()
  check_and_log_data_quality()           # NEW: validate corrupted data, log issues
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import requests
from functools import lru_cache

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = _ROOT / "data" / "processed" / "demand_enriched.parquet"
LOOKUP_PATH = _ROOT / "metadata" / "Lookups" / "taxi_zone_lookup.csv"
MODEL_PATH = _ROOT / "data" / "processed" / "lgbm_demand_model.txt"

# Week 3: corrupted upstream data lives here
CORRUPTED_DATA_PATH = Path(__file__).parent.parent / "data" / "demand_enriched_corrupted.parquet"

REFERENCE_DATE = pd.Timestamp("2026-02-14")
AIRPORT_ZONES = {1, 132, 138}

HOLIDAYS = {
    (1, 1): "New Year's Day",
    (1, 20): "MLK Day",
    (2, 17): "Presidents Day",
    (3, 17): "St. Patrick's Day",
    (5, 26): "Memorial Day",
    (7, 4): "Independence Day",
    (9, 1): "Labor Day",
    (10, 13): "Columbus Day",
    (10, 31): "Halloween",
    (11, 11): "Veterans Day",
    (11, 27): "Thanksgiving",
    (12, 24): "Christmas Eve",
    (12, 25): "Christmas",
    (12, 31): "New Year's Eve",
}


def _identify_holiday(date: pd.Timestamp) -> str:
    key = (date.month, date.day)
    return HOLIDAYS.get(key, "regular")


def _get_week_context(date_str: str) -> tuple:
    target_date = pd.Timestamp(date_str)
    holiday = _identify_holiday(target_date)
    return target_date, target_date.month, target_date.day, holiday


FEATURES = [
    "PULocationID", "hour", "minute", "dayofweek", "is_weekend", "month",
    "dayofyear", "weekofyear", "year", "slot_of_day",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_holiday", "cbd_pricing_active", "is_airport_zone", "borough_id",
    "service_zone_id", "zone_slot_baseline",
    "lag_15min", "lag_1h", "lag_2h", "lag_1day", "lag_1week",
    "roll_mean_1h", "roll_mean_2h", "roll_mean_1day",
]


# ── Week 3: Data Quality Validation ──────────────────────────────────────────

def check_and_log_data_quality() -> dict:
    """
    Run validation on the corrupted upstream parquet and log all issues found.

    This function is called once at startup.  It:
      1. Loads demand_enriched_corrupted.parquet
      2. Splits on CUTOFF (2026-01-16): baseline = pre-cutoff, corrupted = post
      3. Runs DataQualityValidator against the corrupted window
      4. Logs each issue via Python logging (WARNING level)
      5. Returns the validation result dict — the API continues regardless

    Design decisions:
      • Uses a try/except so the entire function is a no-op if the file is
        missing or if the validation code itself has a bug.
      • Never raises — the API must start even if validation fails.
      • Returns a result dict so callers (e.g. a /health endpoint) can surface
        validation status without crashing.
    """
    try:
        # Import here so the module can load even if validation package is absent
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from validation.check_data_quality import DataQualityValidator, CUTOFF

        if not CORRUPTED_DATA_PATH.exists():
            logger.warning(
                "Corrupted data file not found at %s — skipping validation.",
                CORRUPTED_DATA_PATH,
            )
            return {"is_valid": True, "num_issues": 0, "issues": [],
                    "skipped": True, "reason": "file_not_found"}

        logger.info("Loading corrupted data from %s …", CORRUPTED_DATA_PATH)
        df = pd.read_parquet(CORRUPTED_DATA_PATH)
        df["time_bucket"] = pd.to_datetime(df["time_bucket"])

        baseline = df[df["time_bucket"] < CUTOFF]
        corrupted = df[df["time_bucket"] >= CUTOFF]

        logger.info(
            "Data split — baseline: %s rows, corrupted window: %s rows",
            f"{len(baseline):,}", f"{len(corrupted):,}",
        )

        validator = DataQualityValidator(baseline_df=baseline)
        result = validator.validate(corrupted)

        if result["is_valid"]:
            logger.info("✅  Data quality check passed — no issues found.")
        else:
            logger.warning(
                "⚠️  Data quality check FAILED — %d issue(s) detected in "
                "upstream data (post 2026-01-16). "
                "API continues serving clean Week 2 data.",
                result["num_issues"],
            )
            for issue in result["issues"]:
                sev = issue["severity"].upper()
                count = (
                    f" (rows affected: {issue['count']:,})"
                    if issue.get("count") is not None
                    else ""
                )
                logger.warning(
                    "  [%s] %s: %s%s",
                    sev, issue["type"], issue["description"], count,
                )

        return result

    except Exception as e:
        # Log the error but DO NOT crash the API
        logger.error(
            "Data quality check failed to run (exception: %s). "
            "API will continue with existing data.",
            e,
            exc_info=True,
        )
        return {
            "is_valid": False,
            "num_issues": 0,
            "issues": [],
            "error": str(e),
        }


# ── Week 2 code unchanged below ───────────────────────────────────────────────

def _load():
    print("[NYC Cab Analytics] Loading demand profile...")
    df = pd.read_parquet(
        DATA_PATH,
        columns=["PULocationID", "hour", "dayofweek", "trip_count",
                 "is_holiday", "time_bucket"],
    )
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    df["holiday_name"] = df.apply(
        lambda row: (
            _identify_holiday(row["time_bucket"]) if row["is_holiday"] == 1 else "regular"
        ),
        axis=1,
    )

    regular_df = df[df["is_holiday"] == 0].drop(
        columns=["is_holiday", "time_bucket", "holiday_name"]
    )
    regular_profile = (
        regular_df.groupby(["PULocationID", "hour", "dayofweek"], as_index=False)["trip_count"]
        .mean()
        .rename(columns={"trip_count": "avg"})
    )
    regular_profile["avg"] = regular_profile["avg"].round(3)
    regular_profile["holiday_name"] = "regular"

    holiday_df = df[df["is_holiday"] == 1].copy()
    holiday_profile = (
        holiday_df.groupby(["PULocationID", "hour", "holiday_name"], as_index=False)["trip_count"]
        .mean()
        .rename(columns={"trip_count": "avg"})
    )
    holiday_profile["avg"] = holiday_profile["avg"].round(3)
    holiday_profile["dayofweek"] = -1

    profile = pd.concat([regular_profile, holiday_profile], ignore_index=True)

    zones_df = pd.read_csv(LOOKUP_PATH).rename(
        columns={"LocationID": "zone_id", "Zone": "name",
                 "Borough": "borough", "service_zone": "service_zone"}
    )
    print(f"[NYC Cab Analytics] Profile ready — {len(profile):,} rows, "
          f"{profile['PULocationID'].nunique()} zones")
    return profile, zones_df


def _load_model():
    try:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(MODEL_PATH))
        print("[NYC Cab Analytics] LightGBM model loaded.")
        return model
    except Exception as e:
        print(f"[NYC Cab Analytics] Error loading LightGBM model: {e}")
        return None


def _load_full_demand():
    print("[NYC Cab Analytics] Loading full demand data for forecasting...")
    df = pd.read_parquet(DATA_PATH)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    return df


_profile, _zones_df = _load()
_zone_map = _zones_df.set_index("zone_id").to_dict("index")
_lgbm_model = _load_model()
_full_demand = _load_full_demand() if _lgbm_model else None


def _load_zone_hour_fares():
    try:
        fare_path = _ROOT / "data" / "processed" / "zone_hour_avg_fare.parquet"
        df = pd.read_parquet(fare_path)
        return dict(zip(zip(df["zone_id"], df["hour"]), df["avg_fare"]))
    except Exception as e:
        print(f"[NYC Cab Analytics] Warning: Could not load zone fares: {e}")
        return {}


_zone_hour_fares = _load_zone_hour_fares()
_fallback_avg_fare = 18.70


def _get_zone_hour_fare(zone_id: int, hour: int) -> float:
    fare = _zone_hour_fares.get((zone_id, hour))
    if fare is None:
        fare = _fallback_avg_fare
    return max(5.0, min(100.0, fare))


def _load_zone_coordinates():
    try:
        geojson_path = _ROOT / "app" / "frontend" / "public" / "taxi_zones.geojson"
        with open(geojson_path) as f:
            geojson = json.load(f)
        zone_coords = {}
        for feature in geojson["features"]:
            zone_id = feature["properties"].get("LocationID")
            geom = feature["geometry"]
            if geom["type"] == "Polygon":
                coords = geom["coordinates"][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                zone_coords[int(zone_id)] = {
                    "lat": (min(lats) + max(lats)) / 2,
                    "lon": (min(lons) + max(lons)) / 2,
                }
        return zone_coords
    except Exception as e:
        print(f"[NYC Cab Analytics] Warning: Could not load zone coordinates: {e}")
        return {}


_zone_coords = _load_zone_coordinates()


@lru_cache(maxsize=500)
def _get_drive_time(from_zone: int, to_zone: int) -> int:
    if from_zone not in _zone_coords or to_zone not in _zone_coords:
        return max(3, abs(from_zone - to_zone) // 20)
    try:
        from_coord = _zone_coords[from_zone]
        to_coord = _zone_coords[to_zone]
        url = (
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{from_coord['lon']},{from_coord['lat']};"
            f"{to_coord['lon']},{to_coord['lat']}?overview=false"
        )
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data["routes"]:
                return max(2, int(data["routes"][0]["duration"] / 60))
    except Exception:
        pass
    from_coord = _zone_coords.get(from_zone, {"lat": 40.75, "lon": -73.97})
    to_coord = _zone_coords.get(to_zone, {"lat": 40.75, "lon": -73.97})
    lat_diff = abs(from_coord["lat"] - to_coord["lat"]) * 69
    lon_diff = abs(from_coord["lon"] - to_coord["lon"]) * 54
    distance_miles = (lat_diff ** 2 + lon_diff ** 2) ** 0.5
    return min(45, max(2, int(distance_miles / 15 * 60)))


def _compute_zone_volatility():
    volatility = {}
    if _full_demand is None or _full_demand.empty:
        return volatility
    for zone_id in _profile["PULocationID"].unique():
        for hour in range(24):
            hour_data = _full_demand[
                (_full_demand["PULocationID"] == zone_id) & (_full_demand["hour"] == hour)
            ]["trip_count"]
            if len(hour_data) > 1:
                mean_demand = hour_data.mean()
                std_demand = hour_data.std()
                ratio = (std_demand / mean_demand) if mean_demand > 0 else 0
                volatility[(zone_id, hour)] = ratio
    return volatility


_zone_volatility = _compute_zone_volatility()


def _compute_zone_unmet_demand_baseline():
    unmet_baselines = {}
    if _profile is None or _profile.empty:
        return unmet_baselines
    for zone_id in _profile["PULocationID"].unique():
        zone_profile_demand = _profile[_profile["PULocationID"] == zone_id]["avg"]
        if len(zone_profile_demand) > 0:
            unmet_baselines[zone_id] = zone_profile_demand.quantile(0.75)
    return unmet_baselines


_zone_unmet_baselines = _compute_zone_unmet_demand_baseline()


def _generate_synthetic_current_demand(hour: int, dow: int) -> dict:
    baseline = _profile[
        (_profile["hour"] == hour) & (_profile["dayofweek"] == dow)
    ].copy()
    synthetic = {}
    for _, row in baseline.iterrows():
        zone_id = int(row["PULocationID"])
        base_demand = float(row["avg"])
        noise_factor = np.random.normal(1.0, 0.08)
        rush_hour_boost = 1.2 if hour in [8, 9, 17, 18] else 1.0
        time_pattern = 1.0 + 0.1 * np.sin(2 * np.pi * (hour - 6) / 24)
        zone_variance = 1.0 + (np.random.random() - 0.5) * 0.15
        synthetic[zone_id] = round(
            max(0.0, base_demand * noise_factor * rush_hour_boost * time_pattern * zone_variance), 2
        )
    return synthetic


def get_synthetic_current_demand(hour: int, dow: int, date: str = None) -> dict:
    synthetic = _generate_synthetic_current_demand(hour, dow)
    max_val = max(synthetic.values(), default=1.0)
    return {
        "demand": {str(k): v for k, v in synthetic.items()},
        "max": round(max_val, 2),
        "hour": hour,
        "dow": dow,
        "is_synthetic": True,
        "label": "Synthetic Live Data",
    }


def _compute_temporal_features(time_bucket: pd.Timestamp) -> dict:
    hour = time_bucket.hour
    minute = time_bucket.minute
    dow = time_bucket.dayofweek
    month = time_bucket.month
    day_of_year = time_bucket.dayofyear
    week_of_year = time_bucket.isocalendar().week
    year = time_bucket.year
    slot_of_day = hour * 4 + minute // 15
    return {
        "hour": hour, "minute": minute, "dayofweek": dow,
        "is_weekend": 1 if dow in [5, 6] else 0,
        "month": month, "dayofyear": day_of_year,
        "weekofyear": week_of_year, "year": year,
        "slot_of_day": slot_of_day,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }


def _get_zone_history(zone_id: int, num_slots: int = 672) -> pd.DataFrame:
    if _full_demand is None:
        return pd.DataFrame()
    zone_data = _full_demand[_full_demand["PULocationID"] == zone_id].copy()
    return zone_data.tail(num_slots).sort_values("time_bucket").reset_index(drop=True)


def _calculate_lags_and_rolling(history: pd.DataFrame, last_time: pd.Timestamp) -> dict:
    if history.empty:
        return {k: np.nan for k in [
            "lag_15min", "lag_1h", "lag_2h", "lag_1day", "lag_1week",
            "roll_mean_1h", "roll_mean_2h", "roll_mean_1day",
        ]}

    def get_at(delta):
        matches = history[history["time_bucket"] == last_time - delta]
        return float(matches["trip_count"].values[0]) if len(matches) > 0 else np.nan

    recent_1h = history[history["time_bucket"] > last_time - timedelta(hours=1)]
    recent_2h = history[history["time_bucket"] > last_time - timedelta(hours=2)]
    recent_1day = history[history["time_bucket"] > last_time - timedelta(days=1)]

    return {
        "lag_15min": get_at(timedelta(minutes=15)),
        "lag_1h": get_at(timedelta(hours=1)),
        "lag_2h": get_at(timedelta(hours=2)),
        "lag_1day": get_at(timedelta(days=1)),
        "lag_1week": get_at(timedelta(days=7)),
        "roll_mean_1h": recent_1h["trip_count"].mean() if len(recent_1h) > 0 else np.nan,
        "roll_mean_2h": recent_2h["trip_count"].mean() if len(recent_2h) > 0 else np.nan,
        "roll_mean_1day": recent_1day["trip_count"].mean() if len(recent_1day) > 0 else np.nan,
    }


def forecast_demand(zone_id: int, hour: int, dow: int,
                    num_steps: int = 16, date: str = None) -> list:
    if _lgbm_model is None or _full_demand is None:
        return []
    synthetic_current = _generate_synthetic_current_demand(hour, dow)
    current_synthetic_demand = synthetic_current.get(zone_id, 0.0)
    now = pd.Timestamp.now()
    synthetic_history = []
    for days_back in range(7, 0, -1):
        for hr in range(24):
            for slot in range(4):
                past_time = now - timedelta(days=days_back, hours=hr, minutes=slot * 15)
                past_synthetic = _generate_synthetic_current_demand(past_time.hour, past_time.dayofweek)
                past_demand = past_synthetic.get(zone_id, 0.0)
                zone_data = _full_demand[_full_demand["PULocationID"] == zone_id]
                synthetic_history.append({
                    "time_bucket": past_time,
                    "trip_count": past_demand,
                    "borough_id": int(zone_data["borough_id"].iloc[0]) if len(zone_data) > 0 else 0,
                    "service_zone_id": int(zone_data["service_zone_id"].iloc[0]) if len(zone_data) > 0 else 0,
                    "is_airport_zone": 1 if zone_id in AIRPORT_ZONES else 0,
                    "zone_slot_baseline": float(zone_data["zone_slot_baseline"].iloc[0]) if len(zone_data) > 0 else 0.0,
                })

    synthetic_history_df = pd.DataFrame(synthetic_history)
    zone_data = _full_demand[_full_demand["PULocationID"] == zone_id]
    if zone_data.empty:
        return []

    last_borough_id = int(zone_data["borough_id"].iloc[0])
    last_service_zone_id = int(zone_data["service_zone_id"].iloc[0])
    is_airport = int(zone_data["is_airport_zone"].iloc[0])
    zone_slot_baseline = float(zone_data["zone_slot_baseline"].iloc[0])

    synthetic_history_df = pd.concat([
        synthetic_history_df,
        pd.DataFrame({"time_bucket": [now], "trip_count": [current_synthetic_demand],
                      "borough_id": [last_borough_id], "service_zone_id": [last_service_zone_id],
                      "is_airport_zone": [is_airport], "zone_slot_baseline": [zone_slot_baseline]}),
    ], ignore_index=True)

    predictions = []
    current_time = now + timedelta(minutes=15)

    for _ in range(num_steps):
        temporal_feats = _compute_temporal_features(current_time)
        lag_feats = _calculate_lags_and_rolling(
            synthetic_history_df, current_time - timedelta(minutes=15)
        )
        features_dict = {
            "PULocationID": zone_id,
            **temporal_feats,
            "is_holiday": 0,
            "cbd_pricing_active": 0,
            "is_airport_zone": is_airport,
            "borough_id": last_borough_id,
            "service_zone_id": last_service_zone_id,
            "zone_slot_baseline": zone_slot_baseline,
            **lag_feats,
        }
        X_pred = pd.DataFrame([features_dict])[FEATURES]
        pred_value = max(0.0, _lgbm_model.predict(X_pred)[0])
        predictions.append({
            "time_bucket": current_time.isoformat(),
            "predicted_trips": round(pred_value, 2),
            "hour": temporal_feats["hour"],
            "minute": temporal_feats["minute"],
        })
        synthetic_history_df = pd.concat([
            synthetic_history_df,
            pd.DataFrame({"time_bucket": [current_time], "trip_count": [pred_value],
                          "borough_id": [last_borough_id], "service_zone_id": [last_service_zone_id],
                          "is_airport_zone": [is_airport], "zone_slot_baseline": [zone_slot_baseline]}),
        ], ignore_index=True)
        current_time += timedelta(minutes=15)

    return predictions


def _is_forecast_date(date: str) -> bool:
    if not date:
        return False
    return pd.Timestamp(date) > REFERENCE_DATE


def get_heatmap(hour: int, dow: int, holiday: str = "regular", date: str = None) -> dict:
    is_forecast = _is_forecast_date(date)
    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday
    if holiday != "regular":
        sub = _profile[
            (_profile["hour"] == hour) & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ]
    else:
        sub = _profile[
            (_profile["hour"] == hour) & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ]
    demand = {str(int(r["PULocationID"])): round(float(r["avg"]), 2) for _, r in sub.iterrows()}
    max_val = max(demand.values(), default=1.0)
    return {
        "demand": demand, "max": round(max_val, 2), "hour": hour, "dow": dow,
        "holiday": holiday, "date": date, "is_forecast": is_forecast,
        "data_type": "predicted" if is_forecast else "actual",
        "reference_date": REFERENCE_DATE.isoformat().split("T")[0],
    }


def get_kpis(hour: int, dow: int, holiday: str = "regular", date: str = None) -> dict:
    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday
    if holiday != "regular":
        sub = _profile[
            (_profile["hour"] == hour) & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ].copy()
    else:
        sub = _profile[
            (_profile["hour"] == hour) & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ].copy()
    if sub.empty:
        return {}
    total = float(sub["avg"].sum())
    active = int((sub["avg"] > 0).sum())
    top_row = sub.loc[sub["avg"].idxmax()]
    top_id = int(top_row["PULocationID"])
    top_info = _zone_map.get(top_id, {})
    if holiday != "regular":
        trend_by_hour = (
            _profile[(_profile["dayofweek"] == -1) & (_profile["holiday_name"] == holiday)]
            .groupby("hour")["avg"].sum().sort_index()
        )
    else:
        trend_by_hour = (
            _profile[(_profile["dayofweek"] == dow) & (_profile["holiday_name"] == "regular")]
            .groupby("hour")["avg"].sum().sort_index()
        )
    hour_trend = [round(float(trend_by_hour.get(h, 0))) for h in range(24)]
    sub2 = sub.merge(_zones_df[["zone_id", "borough"]], left_on="PULocationID", right_on="zone_id", how="left")
    borough = {}
    if total > 0:
        for b, grp in sub2.groupby("borough"):
            borough[str(b)] = round(float(grp["avg"].sum() / total * 100), 1)
    if hour > 0:
        prev = float(
            _profile[(_profile["hour"] == hour - 1) & (_profile["dayofweek"] == dow)]["avg"].sum()
        )
        vs_prev = round((total - prev) / prev * 100, 1) if prev > 0 else 0.0
    else:
        vs_prev = 0.0
    return {
        "total_trips": round(total), "active_zones": active,
        "peak_zone_id": top_id, "peak_zone_name": top_info.get("name", f"Zone {top_id}"),
        "peak_zone_trips": round(float(top_row["avg"])),
        "hour_trend": hour_trend, "borough_breakdown": borough, "vs_prev_hour_pct": vs_prev,
    }


def get_ranking(hour: int, dow: int, n: int = 15, holiday: str = "regular", date: str = None) -> list:
    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday
    if holiday != "regular":
        sub = _profile[
            (_profile["hour"] == hour) & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ].copy()
        prev_hour_filter = _profile[
            (_profile["hour"] == max(0, hour - 1)) & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ]
    else:
        sub = _profile[
            (_profile["hour"] == hour) & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ].copy()
        prev_hour_filter = _profile[
            (_profile["hour"] == max(0, hour - 1)) & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ]
    top = sub.nlargest(n, "avg")
    max_val = float(top["avg"].max()) if len(top) else 1.0
    prev_hour = prev_hour_filter.set_index("PULocationID")["avg"]
    result = []
    for i, (_, row) in enumerate(top.iterrows()):
        zid = int(row["PULocationID"])
        info = _zone_map.get(zid, {})
        prev = float(prev_hour.get(zid, row["avg"]))
        avg = float(row["avg"])
        trend = "up" if avg > prev + 0.1 else ("down" if avg < prev - 0.1 else "flat")
        result.append({
            "rank": i + 1, "zone_id": zid, "name": info.get("name", f"Zone {zid}"),
            "borough": info.get("borough", ""), "trips": round(avg, 1),
            "pct_of_max": round(avg / max_val * 100), "trend": trend, "is_airport": zid in AIRPORT_ZONES,
        })
    return result


def get_zone_trend(zone_id: int, dow: int, holiday: str = "regular", date: str = None) -> list:
    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday
    if holiday != "regular":
        sub = _profile[
            (_profile["PULocationID"] == zone_id) & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ].sort_values("hour")
    else:
        sub = _profile[
            (_profile["PULocationID"] == zone_id) & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ].sort_values("hour")
    return [{"hour": int(r["hour"]), "trips": round(float(r["avg"]), 1)} for _, r in sub.iterrows()]


def get_recommendations(zone_id: int, hour: int, dow: int, n: int = 3,
                         holiday: str = "regular", date: str = None) -> list:
    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday
    if holiday != "regular":
        sub = _profile[
            (_profile["hour"] == hour) & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ].copy()
    else:
        sub = _profile[
            (_profile["hour"] == hour) & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ].copy()
    max_val = float(sub["avg"].max()) if len(sub) else 1.0
    current_zone_data = sub[sub["PULocationID"] == zone_id]
    current_zone_demand = float(current_zone_data["avg"].values[0]) if len(current_zone_data) > 0 else 0
    current_zone_is_hot = current_zone_demand / max_val > 0.7
    candidates = sub[~sub["PULocationID"].isin(AIRPORT_ZONES)]
    if not current_zone_is_hot:
        candidates = candidates[candidates["PULocationID"] != zone_id]
    result = []
    if current_zone_is_hot and len(current_zone_data) > 0:
        avg_fare = _get_zone_hour_fare(zone_id, hour)
        info = _zone_map.get(zone_id, {})
        result.append({
            "rank": 1, "zone_id": zone_id,
            "name": f"Stay: {info.get('name', f'Zone {zone_id}')}",
            "borough": info.get("borough", ""), "trips": round(current_zone_demand, 1),
            "demand_score": 100, "drive_minutes": 0,
            "est_yield_min": int(1 * avg_fare),
            "est_yield_max": int(min(4, int(current_zone_demand)) * avg_fare),
            "efficiency_score": 999,
        })
    scored = []
    for _, row in candidates.iterrows():
        zid = int(row["PULocationID"])
        avg = float(row["avg"])
        avg_fare = _get_zone_hour_fare(zid, hour)
        drive_mins = _get_drive_time(zone_id, zid)
        service_time_mins = 20
        total_time_mins = drive_mins + service_time_mins
        max_earn = min(4, int(avg)) * avg_fare
        efficiency = max_earn / total_time_mins if total_time_mins > 0 else 0
        info = _zone_map.get(zid, {})
        scored.append({
            "rank": 0, "zone_id": zid, "name": info.get("name", f"Zone {zid}"),
            "borough": info.get("borough", ""), "trips": round(avg, 1),
            "demand_score": round(avg / max_val * 100),
            "drive_minutes": drive_mins,
            "est_yield_min": int(1 * avg_fare),
            "est_yield_max": int(min(4, int(avg)) * avg_fare),
            "efficiency_score": efficiency,
        })
    scored.sort(key=lambda x: (-x["efficiency_score"], -x["demand_score"]))
    remaining_slots = n - len(result)
    for rec in scored[:remaining_slots]:
        rec["rank"] = len(result) + 1
        result.append(rec)
    return result


def get_current_zone(zone_id: int, hour: int, dow: int,
                      holiday: str = "regular", date: str = None) -> dict:
    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday
    info = _zone_map.get(zone_id, {})
    if holiday != "regular":
        sub = _profile[
            (_profile["PULocationID"] == zone_id) & (_profile["hour"] == hour)
            & (_profile["dayofweek"] == -1) & (_profile["holiday_name"] == holiday)
        ]
        hour_max_filter = _profile[
            (_profile["hour"] == hour) & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ]
    else:
        sub = _profile[
            (_profile["PULocationID"] == zone_id) & (_profile["hour"] == hour)
            & (_profile["dayofweek"] == dow) & (_profile["holiday_name"] == "regular")
        ]
        hour_max_filter = _profile[
            (_profile["hour"] == hour) & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ]
    avg = round(float(sub["avg"].values[0]), 1) if len(sub) else 0.0
    hour_max = float(hour_max_filter["avg"].max())
    pct = round(avg / hour_max * 100) if hour_max > 0 else 0
    level = "High" if pct >= 65 else ("Medium" if pct >= 35 else "Low")
    return {
        "zone_id": zone_id, "name": info.get("name", f"Zone {zone_id}"),
        "borough": info.get("borough", ""), "trips": avg,
        "demand_pct": pct, "demand_level": level, "holiday": holiday,
    }


def get_zone_metadata() -> list:
    return _zones_df.to_dict("records")


def _get_demand_and_forecast(zone_id: int, hour: int, dow: int,
                              date: str = None, holiday: str = "regular"):
    query_date = pd.Timestamp(date) if date else REFERENCE_DATE
    is_future = query_date > REFERENCE_DATE
    current_row = _profile[
        (_profile["PULocationID"] == zone_id) & (_profile["hour"] == hour)
        & (_profile["dayofweek"] == dow) & (_profile["holiday_name"] == holiday)
    ]
    demand_current = float(current_row["avg"].iloc[0]) if len(current_row) > 0 else 0.0
    if not is_future:
        next_hour = (hour + 1) % 24
        next_row = _profile[
            (_profile["PULocationID"] == zone_id) & (_profile["hour"] == next_hour)
            & (_profile["dayofweek"] == dow) & (_profile["holiday_name"] == holiday)
        ]
        demand_next = float(next_row["avg"].iloc[0]) if len(next_row) > 0 else demand_current
        is_forecast_used = False
        trend_direction = "stable"
    else:
        preds = forecast_demand(zone_id, hour, dow, num_steps=4, date=date)
        demand_next = (
            sum(p["predicted_trips"] for p in preds) / len(preds) if preds else demand_current
        )
        is_forecast_used = True
        preds_4h = forecast_demand(zone_id, hour, dow, num_steps=16, date=date)
        if len(preds_4h) >= 16:
            avg_1h = sum(p["predicted_trips"] for p in preds_4h[0:4]) / 4
            avg_4h = sum(p["predicted_trips"] for p in preds_4h[12:16]) / 4
            trend_direction = "rising" if avg_4h > avg_1h * 1.1 else ("falling" if avg_4h < avg_1h * 0.9 else "stable")
        else:
            trend_direction = "stable"
    change_pct = round((demand_next - demand_current) / demand_current * 100) if demand_current > 0 else 0
    return {
        "demand_current": demand_current, "demand_next_hour": demand_next,
        "is_forecast": is_forecast_used, "change_pct": change_pct,
        "trend_direction": trend_direction,
    }


def get_operator_zones(hour: int, dow: int, date: str = None, holiday: str = "regular") -> list:
    if _profile is None:
        return []
    query_date = pd.Timestamp(date) if date else REFERENCE_DATE
    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday
    zones_data = []
    for zone_id in _profile["PULocationID"].unique():
        demand_data = _get_demand_and_forecast(zone_id, hour, dow, date, holiday)
        demand_now = demand_data["demand_current"]
        demand_next = demand_data["demand_next_hour"]
        change_pct = demand_data["change_pct"]
        is_using_forecast = demand_data["is_forecast"]
        trend_direction = demand_data["trend_direction"]
        zone_info = _zone_map.get(zone_id, {})
        p95_baseline = _zone_unmet_baselines.get(zone_id, 0)
        if p95_baseline > 0 and demand_now > p95_baseline:
            overflow = demand_now - p95_baseline
            overflow_ratio = overflow / p95_baseline
            unmet_pct = min(0.30, 0.10 + overflow_ratio * 0.20)
            unmet_est = max(0, round(overflow * unmet_pct))
        else:
            unmet_est = 0
        volatility_ratio = _zone_volatility.get((zone_id, hour), 0.0)
        volatility_label = "high" if volatility_ratio > 0.3 else ("medium" if volatility_ratio > 0.15 else "low")
        supply_status = "tight" if unmet_est > 0 else ("balanced" if demand_now > 30 else "light")
        base_revenue = demand_now * 13.5
        drivers_needed = max(1, round(demand_now / 25))
        action = None
        if supply_status == "tight" and volatility_label == "high":
            action = "High demand + volatile. Raise surge cautiously."
        elif supply_status == "tight" and volatility_label == "low":
            action = "High demand + stable. Can raise surge aggressively."
        elif trend_direction == "rising" and supply_status == "balanced":
            action = "Demand rising. Prepare extra drivers."
        elif trend_direction == "falling" and supply_status == "tight":
            action = "Demand falling. Monitor before surge."
        zones_data.append({
            "zone_id": int(zone_id), "name": zone_info.get("name", f"Zone {zone_id}"),
            "borough": zone_info.get("borough", ""),
            "demand_now": round(float(demand_now), 1),
            "demand_next_hour": round(float(demand_next), 1),
            "change_pct": int(change_pct), "unmet_demand": int(unmet_est),
            "drivers_needed": int(drivers_needed), "revenue_potential": int(round(base_revenue)),
            "supply_status": supply_status, "is_forecast": bool(is_using_forecast),
            "forecast_source": "lgbm" if is_using_forecast else "actual",
            "volatility_label": volatility_label,
            "volatility_score": round(float(volatility_ratio), 2),
            "trend_direction": trend_direction, "action": action,
        })
    zones_data.sort(key=lambda z: (z["unmet_demand"], z["demand_now"]), reverse=True)
    return zones_data


def get_forecast_heatmap(hours_ahead: int = 0) -> dict:
    if _lgbm_model is None or _full_demand is None:
        return {"demand": {}, "max": 0.0, "hours_ahead": hours_ahead, "is_forecast": True}
    now = pd.Timestamp.now()
    current_hour = now.hour
    current_dow = now.dayofweek
    steps_ahead = max(1, (hours_ahead * 60) // 15)
    all_zones = _full_demand["PULocationID"].unique()
    demand = {}
    max_val = 0.0
    for zone_id in all_zones:
        forecast = forecast_demand(int(zone_id), current_hour, current_dow, num_steps=steps_ahead)
        if forecast and len(forecast) >= steps_ahead:
            predicted = forecast[steps_ahead - 1]["predicted_trips"]
            demand[str(int(zone_id))] = round(predicted, 2)
            max_val = max(max_val, predicted)
    return {"demand": demand, "max": round(max_val, 2), "hours_ahead": hours_ahead, "is_forecast": True}


# ── Week 3 startup validation ──────────────────────────────────────────────────
# Run at startup — logs issues, never crashes the API.
_data_quality_result = check_and_log_data_quality()
