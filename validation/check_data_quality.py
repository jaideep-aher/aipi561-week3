"""
Data quality checks for demand_enriched_corrupted.parquet.

Split the dataframe at CUTOFF before calling validate():
    baseline  = df[df['time_bucket'] < CUTOFF]
    corrupted = df[df['time_bucket'] >= CUTOFF]
    result = DataQualityValidator(baseline).validate(corrupted)

CLI:
    python -m validation.check_data_quality [data_dir]
"""

from __future__ import annotations

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CUTOFF = pd.Timestamp("2026-01-16")

CRITICAL_COLUMNS = ["PULocationID", "hour", "dayofweek", "trip_count", "time_bucket"]

VALUE_RANGE_CHECKS = {
    "trip_count": (0, None),
    "hour":       (0, 23),
    "dayofweek":  (0, 6),
}

DIST_SHIFT_SIGMA_THRESHOLD = 3.0
OUTLIER_SIGMA = 5.0


def _split(df: pd.DataFrame):
    df = df.copy()
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    return df[df["time_bucket"] < CUTOFF], df[df["time_bucket"] >= CUTOFF]


class DataQualityValidator:
    def __init__(self, baseline_df: Optional[pd.DataFrame] = None):
        self.baseline = baseline_df
        self.issues: List[dict] = []

    def validate(self, df: pd.DataFrame) -> Dict:
        self.issues = []

        # schema first — if time_bucket is missing we can't split
        self.check_schema(df)

        if self.baseline is None:
            if "time_bucket" not in df.columns:
                baseline = pd.DataFrame(columns=df.columns)
                corrupted = df.copy()
            else:
                baseline, corrupted = _split(df)
        else:
            baseline = self.baseline.copy()
            corrupted = df.copy()
            if "time_bucket" in corrupted.columns:
                corrupted["time_bucket"] = pd.to_datetime(corrupted["time_bucket"])

        if corrupted.empty:
            return {"is_valid": True, "num_issues": 0, "issues": []}

        self.check_null_rates(corrupted, baseline)
        self.check_value_ranges(corrupted)
        self.check_duplicates(corrupted)
        self.check_distributions(corrupted, baseline)

        return {
            "is_valid": len(self.issues) == 0,
            "num_issues": len(self.issues),
            "issues": self.issues,
        }

    def check_schema(self, df: pd.DataFrame):
        missing = set(CRITICAL_COLUMNS) - set(df.columns)
        if missing:
            self._add_issue(
                "schema_missing_columns", "critical",
                f"Required columns missing: {sorted(missing)}",
                count=len(missing), columns=sorted(missing),
            )
        type_errors = [
            f"{col} is {df[col].dtype}"
            for col in ["hour", "dayofweek", "trip_count"]
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
        ]
        if type_errors:
            self._add_issue(
                "schema_wrong_dtype", "high",
                f"Unexpected column types: {type_errors}",
                count=len(type_errors), details=type_errors,
            )

    def check_null_rates(self, df: pd.DataFrame, baseline: pd.DataFrame):
        base_null = baseline.isna().mean()
        curr_null = df.isna().mean()
        for col in df.columns:
            base_rate = float(base_null.get(col, 0.0))
            curr_rate = float(curr_null.get(col, 0.0))
            delta = curr_rate - base_rate
            is_critical = col in CRITICAL_COLUMNS
            threshold = 0.0 if is_critical else 0.01
            if curr_rate > threshold and delta > threshold:
                sev = "critical" if is_critical else ("high" if curr_rate > 0.1 else "medium")
                self._add_issue(
                    "null_rate_spike", sev,
                    f"'{col}' null rate {base_rate:.1%} -> {curr_rate:.1%} (+{delta:.1%})",
                    count=int(df[col].isna().sum()),
                    column=col,
                    baseline_null_rate=round(base_rate, 4),
                    corrupted_null_rate=round(curr_rate, 4),
                    delta=round(delta, 4),
                )

    def check_value_ranges(self, df: pd.DataFrame):
        for col, (lo, hi) in VALUE_RANGE_CHECKS.items():
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if lo is not None:
                bad = series[series < lo]
                if len(bad):
                    self._add_issue(
                        "out_of_range", "high",
                        f"'{col}' has {len(bad):,} values below {lo} (min={bad.min():.2f})",
                        count=len(bad), column=col, bound=f">= {lo}",
                        min_observed=float(bad.min()),
                    )
            if hi is not None:
                bad = series[series > hi]
                if len(bad):
                    self._add_issue(
                        "out_of_range", "high",
                        f"'{col}' has {len(bad):,} values above {hi} (max={bad.max():.2f})",
                        count=len(bad), column=col, bound=f"<= {hi}",
                        max_observed=float(bad.max()),
                    )

        # outlier spike check against baseline
        if "trip_count" in df.columns and self.baseline is not None and "trip_count" in self.baseline.columns:
            base_mean = float(self.baseline["trip_count"].mean())
            base_std  = float(self.baseline["trip_count"].std())
            if base_std > 0:
                threshold = base_mean + OUTLIER_SIGMA * base_std
                outliers  = df[df["trip_count"] > threshold]
                if len(outliers):
                    self._add_issue(
                        "outlier_trip_count", "high",
                        f"trip_count has {len(outliers):,} values >{OUTLIER_SIGMA}σ above baseline "
                        f"(mean={base_mean:.1f}, threshold={threshold:.1f}, max={outliers['trip_count'].max():.1f})",
                        count=len(outliers), column="trip_count",
                        baseline_mean=round(base_mean, 2),
                        baseline_std=round(base_std, 2),
                        threshold=round(threshold, 2),
                        max_observed=float(outliers["trip_count"].max()),
                    )

    def check_duplicates(self, df: pd.DataFrame):
        n_dups = int(df.duplicated().sum())
        if n_dups:
            self._add_issue(
                "duplicate_rows", "high",
                f"{n_dups:,} exact duplicate rows",
                count=n_dups, dup_fraction=round(n_dups / len(df), 4),
            )

    def check_distributions(self, df: pd.DataFrame, baseline: pd.DataFrame):
        col = "trip_count"
        if col not in df.columns or col not in baseline.columns:
            return
        base_mean = float(baseline[col].mean())
        base_std  = float(baseline[col].std())
        curr_mean = float(df[col].mean())
        if base_std == 0:
            return
        z = abs(curr_mean - base_mean) / base_std
        if z > DIST_SHIFT_SIGMA_THRESHOLD:
            direction = "higher" if curr_mean > base_mean else "lower"
            self._add_issue(
                "distribution_shift",
                "critical" if z > 6 else "high",
                f"trip_count mean shifted {z:.1f}σ {direction}: "
                f"baseline={base_mean:.2f}, corrupted={curr_mean:.2f}",
                count=len(df), column=col,
                baseline_mean=round(base_mean, 2),
                corrupted_mean=round(curr_mean, 2),
                z_score=round(z, 2), direction=direction,
            )
        curr_std = float(df[col].std())
        std_ratio = curr_std / base_std if base_std > 0 else 1.0
        if std_ratio > 3.0:
            self._add_issue(
                "variance_explosion", "medium",
                f"trip_count std went from {base_std:.2f} to {curr_std:.2f} ({std_ratio:.1f}x)",
                count=len(df), column=col,
                baseline_std=round(base_std, 2),
                corrupted_std=round(curr_std, 2),
                std_ratio=round(std_ratio, 2),
            )

    def _add_issue(self, issue_type, severity, description, count=None, **details):
        self.issues.append({
            "type": issue_type,
            "severity": severity,
            "description": description,
            "count": count,
            **details,
        })


def compare_distributions(baseline: pd.Series, current: pd.Series,
                           threshold: float = DIST_SHIFT_SIGMA_THRESHOLD) -> bool:
    base_std = baseline.std()
    if base_std == 0:
        return False
    return abs(current.mean() - baseline.mean()) / base_std > threshold


def detect_outliers(series: pd.Series, baseline_series: pd.Series = None,
                    sigma: float = OUTLIER_SIGMA) -> pd.Series:
    ref = baseline_series if baseline_series is not None else series
    mean, std = ref.mean(), ref.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    return (series - mean).abs() > sigma * std


def _run_cli(data_dir: str = "."):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    data_path = Path(data_dir) / "data" / "demand_enriched_corrupted.parquet"
    if not data_path.exists():
        data_path = Path("week3") / "data" / "demand_enriched_corrupted.parquet"
    if not data_path.exists():
        logger.error("Parquet file not found at %s", data_path)
        sys.exit(1)

    logger.info("Loading %s", data_path)
    df = pd.read_parquet(data_path)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])

    baseline  = df[df["time_bucket"] < CUTOFF]
    corrupted = df[df["time_bucket"] >= CUTOFF]
    logger.info("Baseline: %s rows  |  Corrupted: %s rows", f"{len(baseline):,}", f"{len(corrupted):,}")

    result = DataQualityValidator(baseline_df=baseline).validate(corrupted)

    if result["is_valid"]:
        logger.info("Validation passed.")
        sys.exit(0)
    else:
        logger.warning("Validation FAILED — %d issue(s):", result["num_issues"])
        for i in result["issues"]:
            logger.warning("  [%s] %s: %s", i["severity"].upper(), i["type"], i["description"])
        sys.exit(1)


if __name__ == "__main__":
    _run_cli(sys.argv[1] if len(sys.argv) > 1 else ".")
