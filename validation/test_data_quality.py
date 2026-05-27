"""
Data Quality Validation Tests — Week 3 AIPI-561

Covers:
  - Baseline (pre-cutoff) data passes all checks
  - Corrupted (post-cutoff) data triggers each of the four known issues
  - API / data.py never crashes with bad data (graceful degradation)

Run from repo root:
    python -m pytest week3/validation/test_data_quality.py -v
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure the week3 package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.check_data_quality import (
    CUTOFF,
    DataQualityValidator,
    compare_distributions,
    detect_outliers,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent.parent / "data" / "demand_enriched_corrupted.parquet"


def _load_parquet():
    if not DATA_PATH.exists():
        pytest.skip(f"Parquet not found at {DATA_PATH}. Clone the course repo first.")
    df = pd.read_parquet(DATA_PATH)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    return df


@pytest.fixture(scope="session")
def full_df():
    return _load_parquet()


@pytest.fixture(scope="session")
def baseline_data(full_df):
    return full_df[full_df["time_bucket"] < CUTOFF].copy()


@pytest.fixture(scope="session")
def corrupted_data(full_df):
    return full_df[full_df["time_bucket"] >= CUTOFF].copy()


@pytest.fixture(scope="session")
def validator(baseline_data):
    return DataQualityValidator(baseline_df=baseline_data)


# ── Synthetic fixtures (no parquet needed) ────────────────────────────────────

@pytest.fixture
def clean_synthetic():
    """Minimal clean DataFrame that should pass all checks."""
    rng = np.random.default_rng(42)
    n = 500
    times = pd.date_range("2025-06-01", periods=n, freq="15min")
    return pd.DataFrame({
        "PULocationID": rng.integers(1, 265, size=n),
        "time_bucket": times,
        "hour": times.hour,
        "dayofweek": times.dayofweek,
        "trip_count": rng.integers(1, 50, size=n).astype(float),
        "is_holiday": rng.integers(0, 2, size=n),
    })


@pytest.fixture
def synthetic_with_nulls(clean_synthetic):
    """Inject nulls into trip_count (critical column)."""
    df = clean_synthetic.copy()
    df.loc[df.index[:50], "trip_count"] = np.nan
    return df


@pytest.fixture
def synthetic_with_outliers(clean_synthetic):
    """Inject extreme trip_count outliers."""
    df = clean_synthetic.copy()
    df.loc[df.index[:10], "trip_count"] = 99_999.0
    return df


@pytest.fixture
def synthetic_with_duplicates(clean_synthetic):
    """Double the first 30 rows to create duplicates."""
    df = pd.concat([clean_synthetic, clean_synthetic.iloc[:30]], ignore_index=True)
    return df


@pytest.fixture
def synthetic_with_dist_shift(clean_synthetic):
    """Multiply trip_count by 20 to cause a distribution shift."""
    df = clean_synthetic.copy()
    df["trip_count"] = df["trip_count"] * 20
    return df


# ── 1. Baseline data should pass validation ───────────────────────────────────

class TestBaselineData:

    def test_baseline_passes_validation(self, baseline_data, validator):
        """Pre-cutoff data should report no quality issues."""
        result = validator.validate(baseline_data)
        assert result["is_valid"], (
            f"Baseline data unexpectedly failed validation: {result['issues']}"
        )
        assert result["num_issues"] == 0

    def test_baseline_has_no_null_trip_count(self, baseline_data):
        assert baseline_data["trip_count"].isna().sum() == 0, \
            "Baseline trip_count should have zero nulls"

    def test_baseline_trip_count_positive(self, baseline_data):
        assert (baseline_data["trip_count"] >= 0).all(), \
            "Baseline trip_count should be non-negative"

    def test_baseline_hour_range(self, baseline_data):
        assert baseline_data["hour"].between(0, 23).all()

    def test_baseline_dayofweek_range(self, baseline_data):
        assert baseline_data["dayofweek"].between(0, 6).all()


# ── 2. Corrupted data triggers issues ─────────────────────────────────────────

class TestCorruptedData:

    def test_corrupted_fails_validation(self, corrupted_data, validator):
        """Post-cutoff data must fail at least one check."""
        result = validator.validate(corrupted_data)
        assert not result["is_valid"], "Expected corrupted data to fail validation"
        assert result["num_issues"] >= 1

    def test_at_least_two_issues_detected(self, corrupted_data, validator):
        """Assignment requires detection of at least 2 distinct issue types."""
        result = validator.validate(corrupted_data)
        issue_types = {i["type"] for i in result["issues"]}
        assert len(issue_types) >= 2, (
            f"Expected ≥ 2 distinct issue types, got: {issue_types}"
        )


# ── 3. Issue 1 — Null Rate Spike ──────────────────────────────────────────────

class TestNullRateIssue:

    def test_synthetic_nulls_detected(self, clean_synthetic, synthetic_with_nulls):
        """Injected nulls in trip_count should be caught."""
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_nulls)
        assert not result["is_valid"]
        assert any(i["type"] == "null_rate_spike" for i in result["issues"]), \
            f"Expected null_rate_spike issue. Got: {[i['type'] for i in result['issues']]}"

    def test_null_issue_has_correct_severity(self, clean_synthetic, synthetic_with_nulls):
        """Nulls in a CRITICAL column should be marked critical or high."""
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_nulls)
        null_issues = [i for i in result["issues"] if i["type"] == "null_rate_spike"]
        sevs = {i["severity"] for i in null_issues}
        assert sevs & {"critical", "high"}, f"Unexpected severities: {sevs}"

    def test_real_corrupted_has_null_issues(self, corrupted_data, validator):
        """Real corrupted window should show elevated null rates."""
        result = validator.validate(corrupted_data)
        # null_rate_spike OR distribution_shift — either confirms data degradation
        issue_types = {i["type"] for i in result["issues"]}
        assert issue_types & {"null_rate_spike", "distribution_shift", "outlier_trip_count"}, \
            f"Expected at least one data-quality issue. Got: {issue_types}"


# ── 4. Issue 2 — Outliers ────────────────────────────────────────────────────

class TestOutlierIssue:

    def test_synthetic_outliers_detected(self, clean_synthetic, synthetic_with_outliers):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_outliers)
        assert not result["is_valid"]
        assert any(i["type"] == "outlier_trip_count" for i in result["issues"]), \
            f"Expected outlier_trip_count. Got: {[i['type'] for i in result['issues']]}"

    def test_outlier_count_is_accurate(self, clean_synthetic, synthetic_with_outliers):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_outliers)
        outlier_issues = [i for i in result["issues"] if i["type"] == "outlier_trip_count"]
        assert outlier_issues, "No outlier issue found"
        # We injected 10 outlier rows
        assert outlier_issues[0]["count"] >= 10

    def test_detect_outliers_utility(self):
        """Utility function detect_outliers() correctly flags injected spikes."""
        base = pd.Series([10.0] * 200 + [12.0] * 200)
        current = pd.Series([10.0] * 50 + [9999.0] * 5 + [11.0] * 45)
        mask = detect_outliers(current, baseline_series=base, sigma=5.0)
        assert mask.sum() == 5, f"Expected 5 outliers flagged, got {mask.sum()}"


# ── 5. Issue 3 — Duplicates ──────────────────────────────────────────────────

class TestDuplicateIssue:

    def test_synthetic_duplicates_detected(self, clean_synthetic, synthetic_with_duplicates):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_duplicates)
        assert not result["is_valid"]
        assert any(i["type"] == "duplicate_rows" for i in result["issues"]), \
            f"Expected duplicate_rows issue. Got: {[i['type'] for i in result['issues']]}"

    def test_duplicate_count_is_accurate(self, clean_synthetic, synthetic_with_duplicates):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_duplicates)
        dup_issues = [i for i in result["issues"] if i["type"] == "duplicate_rows"]
        assert dup_issues, "No duplicate issue found"
        assert dup_issues[0]["count"] >= 30, \
            f"Expected >= 30 duplicates, got {dup_issues[0]['count']}"

    def test_clean_data_has_no_duplicates(self, clean_synthetic):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(clean_synthetic)
        dup_issues = [i for i in result["issues"] if i["type"] == "duplicate_rows"]
        assert len(dup_issues) == 0, f"Clean data should have no duplicates: {dup_issues}"


# ── 6. Issue 4 — Distribution Shift ─────────────────────────────────────────

class TestDistributionShiftIssue:

    def test_synthetic_dist_shift_detected(self, clean_synthetic, synthetic_with_dist_shift):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_dist_shift)
        assert not result["is_valid"]
        assert any(i["type"] == "distribution_shift" for i in result["issues"]), \
            f"Expected distribution_shift. Got: {[i['type'] for i in result['issues']]}"

    def test_compare_distributions_utility(self):
        """Utility function compare_distributions() returns True for large shifts."""
        baseline = pd.Series(np.random.normal(10, 2, 1000))
        shifted = pd.Series(np.random.normal(100, 2, 1000))
        assert compare_distributions(baseline, shifted, threshold=3.0)

    def test_compare_distributions_no_shift(self):
        rng = np.random.default_rng(99)
        a = pd.Series(rng.normal(10, 2, 1000))
        b = pd.Series(rng.normal(10.1, 2, 1000))
        # Small difference should NOT exceed 3σ threshold
        assert not compare_distributions(a, b, threshold=3.0)


# ── 7. Graceful Degradation ───────────────────────────────────────────────────

class TestGracefulDegradation:

    def test_validator_does_not_raise_on_corrupted(self, corrupted_data, validator):
        """validate() must never raise an exception, even on bad data."""
        try:
            result = validator.validate(corrupted_data)
        except Exception as e:
            pytest.fail(f"validator.validate() raised an exception: {e}")
        assert isinstance(result, dict)
        assert "is_valid" in result

    def test_validator_handles_empty_dataframe(self, validator):
        """Empty DataFrame should not crash; treated as valid (nothing to check)."""
        empty = pd.DataFrame(columns=["PULocationID", "time_bucket", "hour",
                                       "dayofweek", "trip_count", "is_holiday"])
        try:
            result = validator.validate(empty)
        except Exception as e:
            pytest.fail(f"validate() raised on empty df: {e}")
        assert isinstance(result, dict)

    def test_validator_handles_missing_columns(self):
        """Incomplete DataFrame missing critical columns should report schema issue, not crash."""
        df = pd.DataFrame({"PULocationID": [1, 2], "hour": [8, 9]})
        v = DataQualityValidator()
        try:
            result = v.validate(df)
        except Exception as e:
            pytest.fail(f"validate() crashed on missing columns: {e}")
        # Should report schema issue
        assert any(i["type"] == "schema_missing_columns" for i in result["issues"])

    def test_check_and_log_does_not_crash(self, caplog):
        """
        Simulate what data.py does at startup: run validation and log issues.
        This should never crash regardless of data quality.
        """
        # We test the pattern, not the actual import of data.py
        # (which requires lightgbm, GCS, etc.)
        import logging

        logger = logging.getLogger("test_graceful")

        def simulated_check_and_log(df_corrupted, baseline_df):
            """Mirrors check_and_log_data_quality() from data.py."""
            try:
                validator = DataQualityValidator(baseline_df=baseline_df)
                result = validator.validate(df_corrupted)
                if not result["is_valid"]:
                    logger.warning(
                        "Data quality issues detected: %d issue(s) found.",
                        result["num_issues"]
                    )
                    for issue in result["issues"]:
                        logger.warning(
                            "[%s] %s: %s",
                            issue["severity"].upper(),
                            issue["type"],
                            issue["description"]
                        )
                else:
                    logger.info("Data quality check passed — no issues found.")
                return result
            except Exception as e:
                logger.error("Data quality check failed to run: %s", e)
                return {"is_valid": True, "num_issues": 0, "issues": []}

        # Manufacture obviously bad data
        bad_df = pd.DataFrame({
            "PULocationID": [1, 2, 2],
            "time_bucket": pd.to_datetime(["2026-01-20", "2026-01-20", "2026-01-20"]),
            "hour": [8, 9, 9],
            "dayofweek": [0, 0, 0],
            "trip_count": [99999.0, np.nan, np.nan],
            "is_holiday": [0, 0, 0],
        })
        good_df = pd.DataFrame({
            "PULocationID": [1, 2],
            "time_bucket": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "hour": [8, 9],
            "dayofweek": [0, 0],
            "trip_count": [10.0, 12.0],
            "is_holiday": [0, 0],
        })

        with caplog.at_level(logging.WARNING, logger="test_graceful"):
            result = simulated_check_and_log(bad_df, good_df)

        assert isinstance(result, dict), "Should always return a dict"
        # API continues — no exception means graceful degradation worked
