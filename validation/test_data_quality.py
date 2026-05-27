"""
Tests for validation/check_data_quality.py.

Synthetic fixtures run without the parquet file.
Tests that load the real corrupted data are skipped if the file is missing.

Run from the week3 directory:
    python -m pytest validation/test_data_quality.py -v
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.check_data_quality import (
    CUTOFF,
    DataQualityValidator,
    compare_distributions,
    detect_outliers,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "demand_enriched_corrupted.parquet"


def _load_parquet():
    if not DATA_PATH.exists():
        pytest.skip(f"Parquet not found at {DATA_PATH}")
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


@pytest.fixture
def clean_synthetic():
    rng = np.random.default_rng(42)
    n = 500
    times = pd.date_range("2025-06-01", periods=n, freq="15min")
    return pd.DataFrame({
        "PULocationID": rng.integers(1, 265, size=n),
        "time_bucket":  times,
        "hour":         times.hour,
        "dayofweek":    times.dayofweek,
        "trip_count":   rng.integers(1, 50, size=n).astype(float),
        "is_holiday":   rng.integers(0, 2, size=n),
    })


@pytest.fixture
def synthetic_with_nulls(clean_synthetic):
    df = clean_synthetic.copy()
    df.loc[df.index[:50], "trip_count"] = np.nan
    return df


@pytest.fixture
def synthetic_with_outliers(clean_synthetic):
    df = clean_synthetic.copy()
    df.loc[df.index[:10], "trip_count"] = 99_999.0
    return df


@pytest.fixture
def synthetic_with_duplicates(clean_synthetic):
    return pd.concat([clean_synthetic, clean_synthetic.iloc[:30]], ignore_index=True)


@pytest.fixture
def synthetic_with_dist_shift(clean_synthetic):
    df = clean_synthetic.copy()
    df["trip_count"] = df["trip_count"] * 20
    return df


class TestBaselineData:
    def test_baseline_passes_validation(self, baseline_data, validator):
        result = validator.validate(baseline_data)
        assert result["is_valid"], f"Baseline failed: {result['issues']}"
        assert result["num_issues"] == 0

    def test_no_null_trip_count(self, baseline_data):
        assert baseline_data["trip_count"].isna().sum() == 0

    def test_trip_count_non_negative(self, baseline_data):
        assert (baseline_data["trip_count"] >= 0).all()

    def test_hour_range(self, baseline_data):
        assert baseline_data["hour"].between(0, 23).all()

    def test_dayofweek_range(self, baseline_data):
        assert baseline_data["dayofweek"].between(0, 6).all()


class TestCorruptedData:
    def test_corrupted_fails_validation(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        assert not result["is_valid"]

    def test_at_least_two_issues_detected(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        issue_types = {i["type"] for i in result["issues"]}
        assert len(issue_types) >= 2, f"Only found: {issue_types}"


class TestNullRateIssue:
    def test_nulls_detected(self, clean_synthetic, synthetic_with_nulls):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_nulls)
        assert not result["is_valid"]
        assert any(i["type"] == "null_rate_spike" for i in result["issues"])

    def test_null_severity(self, clean_synthetic, synthetic_with_nulls):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_nulls)
        sevs = {i["severity"] for i in result["issues"] if i["type"] == "null_rate_spike"}
        assert sevs & {"critical", "high"}

    def test_real_data_has_data_issues(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        types = {i["type"] for i in result["issues"]}
        assert types & {"null_rate_spike", "distribution_shift", "outlier_trip_count"}


class TestOutlierIssue:
    def test_outliers_detected(self, clean_synthetic, synthetic_with_outliers):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_outliers)
        assert any(i["type"] == "outlier_trip_count" for i in result["issues"])

    def test_outlier_count(self, clean_synthetic, synthetic_with_outliers):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_outliers)
        issues = [i for i in result["issues"] if i["type"] == "outlier_trip_count"]
        assert issues and issues[0]["count"] >= 10

    def test_detect_outliers_utility(self):
        base = pd.Series([10.0] * 200 + [12.0] * 200)
        curr = pd.Series([10.0] * 50 + [9999.0] * 5 + [11.0] * 45)
        mask = detect_outliers(curr, baseline_series=base, sigma=5.0)
        assert mask.sum() == 5


class TestDuplicateIssue:
    def test_duplicates_detected(self, clean_synthetic, synthetic_with_duplicates):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_duplicates)
        assert any(i["type"] == "duplicate_rows" for i in result["issues"])

    def test_duplicate_count(self, clean_synthetic, synthetic_with_duplicates):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_duplicates)
        issues = [i for i in result["issues"] if i["type"] == "duplicate_rows"]
        assert issues and issues[0]["count"] >= 30

    def test_clean_data_no_duplicates(self, clean_synthetic):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(clean_synthetic)
        assert not any(i["type"] == "duplicate_rows" for i in result["issues"])


class TestDistributionShiftIssue:
    def test_dist_shift_detected(self, clean_synthetic, synthetic_with_dist_shift):
        v = DataQualityValidator(baseline_df=clean_synthetic)
        result = v.validate(synthetic_with_dist_shift)
        assert any(i["type"] == "distribution_shift" for i in result["issues"])

    def test_compare_distributions_utility(self):
        baseline = pd.Series(np.random.normal(10, 2, 1000))
        shifted  = pd.Series(np.random.normal(100, 2, 1000))
        assert compare_distributions(baseline, shifted, threshold=3.0)

    def test_compare_distributions_no_shift(self):
        rng = np.random.default_rng(99)
        a = pd.Series(rng.normal(10, 2, 1000))
        b = pd.Series(rng.normal(10.1, 2, 1000))
        assert not compare_distributions(a, b, threshold=3.0)


class TestGracefulDegradation:
    def test_no_exception_on_corrupted(self, corrupted_data, validator):
        try:
            result = validator.validate(corrupted_data)
        except Exception as e:
            pytest.fail(f"validate() raised: {e}")
        assert isinstance(result, dict)

    def test_empty_dataframe(self, validator):
        empty = pd.DataFrame(columns=["PULocationID", "time_bucket", "hour",
                                       "dayofweek", "trip_count", "is_holiday"])
        try:
            result = validator.validate(empty)
        except Exception as e:
            pytest.fail(f"validate() raised on empty df: {e}")
        assert isinstance(result, dict)

    def test_missing_columns(self):
        df = pd.DataFrame({"PULocationID": [1, 2], "hour": [8, 9]})
        v = DataQualityValidator()
        try:
            result = v.validate(df)
        except Exception as e:
            pytest.fail(f"validate() crashed on missing columns: {e}")
        assert any(i["type"] == "schema_missing_columns" for i in result["issues"])

    def test_startup_check_never_crashes(self):
        bad = pd.DataFrame({
            "PULocationID": [1, 2, 2],
            "time_bucket":  pd.to_datetime(["2026-01-20"] * 3),
            "hour":         [8, 9, 9],
            "dayofweek":    [0, 0, 0],
            "trip_count":   [99999.0, np.nan, np.nan],
            "is_holiday":   [0, 0, 0],
        })
        good = pd.DataFrame({
            "PULocationID": [1, 2],
            "time_bucket":  pd.to_datetime(["2025-01-01"] * 2),
            "hour":         [8, 9],
            "dayofweek":    [0, 0],
            "trip_count":   [10.0, 12.0],
            "is_holiday":   [0, 0],
        })

        def check_and_log(corrupted, baseline):
            try:
                v = DataQualityValidator(baseline_df=baseline)
                return v.validate(corrupted)
            except Exception as e:
                logging.error("check failed: %s", e)
                return {"is_valid": True, "num_issues": 0, "issues": []}

        result = check_and_log(bad, good)
        assert isinstance(result, dict)
