"""
Data Quality Validation — Week 3 AIPI-561

Checks the demand_enriched_corrupted.parquet file for issues introduced on
or after 2026-01-16.  Four issues are known to exist; this module detects
all four, but the API only requires at least two.

Issues identified:
  1. NULLS       — critical columns develop missing values post-cutoff
  2. OUTLIERS    — trip_count spikes >10× the baseline 99th-percentile
  3. DUPLICATES  — exact-row duplicates appear in the corrupted window
  4. DIST SHIFT  — mean trip_count shifts significantly (>3 σ from baseline)

Usage (CLI):
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

# ── Constants ─────────────────────────────────────────────────────────────────

CUTOFF = pd.Timestamp("2026-01-16")

# Columns that must never be null for the model to work
CRITICAL_COLUMNS = ["PULocationID", "hour", "dayofweek", "trip_count", "time_bucket"]

# Columns allowed to have *some* nulls in baseline (tolerance)
NULL_TOLERANCE: Dict[str, float] = {
    "is_holiday": 0.0,
    "trip_count": 0.0,
}

# trip_count must stay non-negative
VALUE_RANGE_CHECKS = {
    "trip_count": (0, None),          # >= 0, no upper bound enforced here
    "hour": (0, 23),
    "dayofweek": (0, 6),
}

# Distribution shift: flag if mean changes by more than this many baseline σ
DIST_SHIFT_SIGMA_THRESHOLD = 3.0

# Outlier: flag trip_count values beyond this many σ above baseline mean
OUTLIER_SIGMA = 5.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split(df: pd.DataFrame):
    """Return (baseline, corrupted) split on CUTOFF."""
    df = df.copy()
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    baseline = df[df["time_bucket"] < CUTOFF]
    corrupted = df[df["time_bucket"] >= CUTOFF]
    return baseline, corrupted


# ── Validator ─────────────────────────────────────────────────────────────────

class DataQualityValidator:
    """
    Runs a battery of quality checks and returns a structured results dict.

    Parameters
    ----------
    baseline_df : pd.DataFrame, optional
        Pre-cutoff reference data.  If None, the validator will derive it
        from the dataframe passed to ``validate()`` using CUTOFF.
    """

    def __init__(self, baseline_df: Optional[pd.DataFrame] = None):
        self.baseline = baseline_df
        self.issues: List[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Run all checks against *df* (should be the full or corrupted window).

        Returns
        -------
        {
            "is_valid": bool,
            "num_issues": int,
            "issues": list[dict],
        }
        """
        self.issues = []

        # Schema check first — if critical columns are missing we may not be
        # able to split or compute distributions, so run it before anything else.
        self.check_schema(df)
        missing_critical = {i["type"] for i in self.issues} == {"schema_missing_columns"}

        # Derive baseline if not provided
        if self.baseline is None:
            if "time_bucket" not in df.columns:
                # Cannot split — treat entire df as corrupted, skip time-based checks
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
            logger.info("No post-cutoff rows found — nothing to validate.")
            return {"is_valid": True, "num_issues": 0, "issues": []}

        # Run all checks (schema already checked above)
        self.check_null_rates(corrupted, baseline)
        self.check_value_ranges(corrupted)
        self.check_duplicates(corrupted)
        self.check_distributions(corrupted, baseline)

        return {
            "is_valid": len(self.issues) == 0,
            "num_issues": len(self.issues),
            "issues": self.issues,
        }

    # ── Check 1: Schema ───────────────────────────────────────────────────────

    def check_schema(self, df: pd.DataFrame):
        """Verify required columns are present with expected dtypes."""
        required = set(CRITICAL_COLUMNS)
        missing_cols = required - set(df.columns)
        if missing_cols:
            self._add_issue(
                issue_type="schema_missing_columns",
                severity="critical",
                description=f"Required columns missing: {sorted(missing_cols)}",
                count=len(missing_cols),
                columns=sorted(missing_cols),
            )

        # Check dtypes of present columns
        type_errors = []
        for col in ["hour", "dayofweek", "trip_count"]:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    type_errors.append(f"{col} is {df[col].dtype}")
        if type_errors:
            self._add_issue(
                issue_type="schema_wrong_dtype",
                severity="high",
                description=f"Unexpected column types: {type_errors}",
                count=len(type_errors),
                details=type_errors,
            )

    # ── Check 2: Null Rates ───────────────────────────────────────────────────

    def check_null_rates(self, df: pd.DataFrame, baseline: pd.DataFrame):
        """
        Issue 1 — Null Rate Spike

        Compare per-column null rates between baseline and corrupted windows.
        Flag columns where the null rate increases by more than 1 pp or any
        critical column has *any* nulls.
        """
        baseline_null = baseline.isna().mean()
        corrupted_null = df.isna().mean()

        for col in df.columns:
            base_rate = float(baseline_null.get(col, 0.0))
            curr_rate = float(corrupted_null.get(col, 0.0))
            delta = curr_rate - base_rate

            is_critical = col in CRITICAL_COLUMNS
            threshold = 0.0 if is_critical else 0.01  # 1 pp for non-critical

            if curr_rate > threshold and delta > threshold:
                null_count = int(df[col].isna().sum())
                sev = "critical" if is_critical else ("high" if curr_rate > 0.1 else "medium")
                self._add_issue(
                    issue_type="null_rate_spike",
                    severity=sev,
                    description=(
                        f"Column '{col}' null rate increased from "
                        f"{base_rate:.1%} → {curr_rate:.1%} "
                        f"(+{delta:.1%})"
                    ),
                    count=null_count,
                    column=col,
                    baseline_null_rate=round(base_rate, 4),
                    corrupted_null_rate=round(curr_rate, 4),
                    delta=round(delta, 4),
                )

    # ── Check 3: Value Ranges ────────────────────────────────────────────────

    def check_value_ranges(self, df: pd.DataFrame):
        """
        Issue 2 — Out-of-Range / Outlier Values

        Check columns against hard bounds and flag extreme outliers in
        trip_count (>OUTLIER_SIGMA σ above baseline mean, treated as spike).
        """
        for col, (lo, hi) in VALUE_RANGE_CHECKS.items():
            if col not in df.columns:
                continue
            series = df[col].dropna()

            # Hard lower bound
            if lo is not None:
                bad = series[series < lo]
                if len(bad) > 0:
                    self._add_issue(
                        issue_type="out_of_range",
                        severity="high",
                        description=f"Column '{col}' has {len(bad):,} values below {lo} (min={bad.min():.2f})",
                        count=len(bad),
                        column=col,
                        bound=f">= {lo}",
                        min_observed=float(bad.min()),
                    )

            # Hard upper bound
            if hi is not None:
                bad = series[series > hi]
                if len(bad) > 0:
                    self._add_issue(
                        issue_type="out_of_range",
                        severity="high",
                        description=f"Column '{col}' has {len(bad):,} values above {hi} (max={bad.max():.2f})",
                        count=len(bad),
                        column=col,
                        bound=f"<= {hi}",
                        max_observed=float(bad.max()),
                    )

        # Outlier spike detection for trip_count using baseline stats
        if "trip_count" in df.columns and self.baseline is not None and "trip_count" in self.baseline.columns:
            base_mean = float(self.baseline["trip_count"].mean())
            base_std = float(self.baseline["trip_count"].std())
            if base_std > 0:
                threshold = base_mean + OUTLIER_SIGMA * base_std
                outliers = df[df["trip_count"] > threshold]
                if len(outliers) > 0:
                    self._add_issue(
                        issue_type="outlier_trip_count",
                        severity="high",
                        description=(
                            f"trip_count has {len(outliers):,} extreme outliers "
                            f"(>{OUTLIER_SIGMA}σ above baseline mean={base_mean:.1f}); "
                            f"max observed={outliers['trip_count'].max():.1f}, threshold={threshold:.1f}"
                        ),
                        count=len(outliers),
                        column="trip_count",
                        baseline_mean=round(base_mean, 2),
                        baseline_std=round(base_std, 2),
                        threshold=round(threshold, 2),
                        max_observed=float(outliers["trip_count"].max()),
                    )

    # ── Check 4: Duplicates ───────────────────────────────────────────────────

    def check_duplicates(self, df: pd.DataFrame):
        """
        Issue 3 — Duplicate Rows

        Exact duplicate rows indicate data pipeline errors (double-ingestion).
        Key uniqueness constraint: (PULocationID, time_bucket) should be unique.
        """
        # Exact duplicates
        n_dups = int(df.duplicated().sum())
        if n_dups > 0:
            self._add_issue(
                issue_type="duplicate_rows",
                severity="high",
                description=f"{n_dups:,} exact duplicate rows found in corrupted window",
                count=n_dups,
                dup_fraction=round(n_dups / len(df), 4),
            )

        # Key-level duplicates (zone × time_bucket)
        key_cols = [c for c in ["PULocationID", "time_bucket"] if c in df.columns]
        if len(key_cols) == 2:
            key_dups = int(df.duplicated(subset=key_cols).sum())
            if key_dups > n_dups:  # only report if additional key-dups beyond exact
                self._add_issue(
                    issue_type="duplicate_keys",
                    severity="medium",
                    description=f"{key_dups:,} rows share the same (PULocationID, time_bucket) key",
                    count=key_dups,
                )

    # ── Check 5: Distribution Shift ───────────────────────────────────────────

    def check_distributions(self, df: pd.DataFrame, baseline: pd.DataFrame):
        """
        Issue 4 — Distribution Shift

        Compare mean and std of trip_count between baseline and corrupted window.
        Flag if the shift exceeds DIST_SHIFT_SIGMA_THRESHOLD baseline σ units.
        """
        col = "trip_count"
        if col not in df.columns or col not in baseline.columns:
            return

        base_mean = float(baseline[col].mean())
        base_std = float(baseline[col].std())
        curr_mean = float(df[col].mean())

        if base_std == 0:
            return

        z_score = abs(curr_mean - base_mean) / base_std
        if z_score > DIST_SHIFT_SIGMA_THRESHOLD:
            direction = "higher" if curr_mean > base_mean else "lower"
            self._add_issue(
                issue_type="distribution_shift",
                severity="critical" if z_score > 6 else "high",
                description=(
                    f"trip_count mean shifted significantly: "
                    f"baseline={base_mean:.2f}, corrupted={curr_mean:.2f} "
                    f"({direction} by {z_score:.1f}σ)"
                ),
                count=len(df),
                column=col,
                baseline_mean=round(base_mean, 2),
                corrupted_mean=round(curr_mean, 2),
                z_score=round(z_score, 2),
                direction=direction,
            )

        # Also check std explosion
        curr_std = float(df[col].std())
        std_ratio = curr_std / base_std if base_std > 0 else 1.0
        if std_ratio > 3.0:
            self._add_issue(
                issue_type="variance_explosion",
                severity="medium",
                description=(
                    f"trip_count std exploded: baseline={base_std:.2f}, "
                    f"corrupted={curr_std:.2f} ({std_ratio:.1f}× increase)"
                ),
                count=len(df),
                column=col,
                baseline_std=round(base_std, 2),
                corrupted_std=round(curr_std, 2),
                std_ratio=round(std_ratio, 2),
            )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _add_issue(self, issue_type: str, severity: str, description: str,
                   count: int = None, **details):
        issue = {
            "type": issue_type,
            "severity": severity,
            "description": description,
            "count": count,
            **details,
        }
        self.issues.append(issue)
        logger.debug("Issue found: %s — %s", issue_type, description)


# ── Utility functions (also importable) ───────────────────────────────────────

def compare_distributions(baseline: pd.Series, current: pd.Series,
                           threshold: float = DIST_SHIFT_SIGMA_THRESHOLD) -> bool:
    """Return True if distributions are significantly different (mean shifted > threshold σ)."""
    base_std = baseline.std()
    if base_std == 0:
        return False
    z = abs(current.mean() - baseline.mean()) / base_std
    return z > threshold


def detect_outliers(series: pd.Series, baseline_series: pd.Series = None,
                    sigma: float = OUTLIER_SIGMA) -> pd.Series:
    """Return boolean mask of outliers (True = outlier)."""
    ref = baseline_series if baseline_series is not None else series
    mean, std = ref.mean(), ref.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    return (series - mean).abs() > sigma * std


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def _run_cli(data_dir: str = "."):
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s  %(message)s")

    data_path = Path(data_dir) / "data" / "demand_enriched_corrupted.parquet"
    if not data_path.exists():
        # also try relative to cwd
        data_path = Path("week3") / "data" / "demand_enriched_corrupted.parquet"
    if not data_path.exists():
        logger.error("Parquet file not found. Searched: %s", data_path)
        sys.exit(1)

    logger.info("Loading %s …", data_path)
    df = pd.read_parquet(data_path)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])

    baseline = df[df["time_bucket"] < CUTOFF]
    corrupted = df[df["time_bucket"] >= CUTOFF]

    logger.info("Baseline rows: %s  |  Corrupted rows: %s",
                f"{len(baseline):,}", f"{len(corrupted):,}")

    validator = DataQualityValidator(baseline_df=baseline)
    result = validator.validate(corrupted)

    if result["is_valid"]:
        logger.info("✅  Validation PASSED — no issues found.")
        sys.exit(0)
    else:
        logger.warning("❌  Validation FAILED — %d issue(s) found:", result["num_issues"])
        for issue in result["issues"]:
            sev = issue["severity"].upper()
            logger.warning("  [%s] %s: %s (rows affected: %s)",
                           sev, issue["type"], issue["description"],
                           f"{issue['count']:,}" if issue["count"] is not None else "N/A")
        sys.exit(1)


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    _run_cli(data_dir)
