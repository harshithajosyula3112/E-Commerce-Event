"""
src/quality/validator.py
-------------------------
Data quality rule engine — Great Expectations pattern, no dependency needed.

Defines typed, composable quality checks that run against any DataFrame.
Each check returns a QualityResult with pass/fail + details.
Thresholds are configurable per table.

Usage:
    python src/quality/validator.py
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from datetime import datetime

log = logging.getLogger(__name__)


@dataclass
class QualityResult:
    check_name: str
    table: str
    passed: bool
    expected: str
    actual: str
    severity: str = "error"  # "error" | "warning" | "info"
    rows_affected: int = 0
    details: dict = field(default_factory=dict)

    def __str__(self):
        status = "✅ PASS" if self.passed else ("⚠️  WARN" if self.severity == "warning" else "❌ FAIL")
        return f"{status} | {self.table}.{self.check_name} | {self.actual} (expected: {self.expected})"


class DataQualityValidator:
    """
    Runs a suite of data quality checks against a DataFrame.
    Mimics Great Expectations API pattern without the dependency.
    """

    def __init__(self, table_name: str):
        self.table_name = table_name
        self.results: list[QualityResult] = []

    def _add(self, result: QualityResult):
        self.results.append(result)
        if not result.passed:
            log.warning(str(result))
        return result

    # ── Core checks ───────────────────────────────────────────────────────────

    def expect_no_nulls(self, df: pd.DataFrame, column: str,
                        severity: str = "error") -> QualityResult:
        null_count = df[column].isna().sum()
        return self._add(QualityResult(
            check_name=f"no_nulls_{column}",
            table=self.table_name,
            passed=null_count == 0,
            expected="0 nulls",
            actual=f"{null_count} nulls ({null_count/len(df)*100:.2f}%)",
            severity=severity,
            rows_affected=null_count,
        ))

    def expect_null_rate_below(self, df: pd.DataFrame, column: str,
                                max_rate: float, severity: str = "warning") -> QualityResult:
        null_rate = df[column].isna().mean()
        return self._add(QualityResult(
            check_name=f"null_rate_{column}",
            table=self.table_name,
            passed=null_rate <= max_rate,
            expected=f"null rate ≤ {max_rate:.1%}",
            actual=f"null rate = {null_rate:.2%}",
            severity=severity,
            rows_affected=int(df[column].isna().sum()),
        ))

    def expect_no_duplicates(self, df: pd.DataFrame, column: str,
                              severity: str = "error") -> QualityResult:
        dup_count = df[column].duplicated().sum()
        return self._add(QualityResult(
            check_name=f"no_duplicates_{column}",
            table=self.table_name,
            passed=dup_count == 0,
            expected="0 duplicates",
            actual=f"{dup_count} duplicates",
            severity=severity,
            rows_affected=int(dup_count),
        ))

    def expect_values_in_set(self, df: pd.DataFrame, column: str,
                              valid_values: set, severity: str = "error") -> QualityResult:
        invalid = ~df[column].isin(valid_values)
        invalid_count = invalid.sum()
        return self._add(QualityResult(
            check_name=f"values_in_set_{column}",
            table=self.table_name,
            passed=invalid_count == 0,
            expected=f"values in {sorted(valid_values)[:5]}...",
            actual=f"{invalid_count} invalid values: {df.loc[invalid, column].unique()[:3].tolist()}",
            severity=severity,
            rows_affected=int(invalid_count),
        ))

    def expect_column_min(self, df: pd.DataFrame, column: str,
                           min_val: float, severity: str = "error") -> QualityResult:
        col_min = df[column].min()
        return self._add(QualityResult(
            check_name=f"min_{column}",
            table=self.table_name,
            passed=col_min >= min_val,
            expected=f"min ≥ {min_val}",
            actual=f"min = {col_min}",
            severity=severity,
            rows_affected=int((df[column] < min_val).sum()),
        ))

    def expect_column_max(self, df: pd.DataFrame, column: str,
                           max_val: float, severity: str = "warning") -> QualityResult:
        col_max = df[column].max()
        return self._add(QualityResult(
            check_name=f"max_{column}",
            table=self.table_name,
            passed=col_max <= max_val,
            expected=f"max ≤ {max_val}",
            actual=f"max = {col_max}",
            severity=severity,
        ))

    def expect_row_count_between(self, df: pd.DataFrame,
                                  min_rows: int, max_rows: int,
                                  severity: str = "error") -> QualityResult:
        n = len(df)
        return self._add(QualityResult(
            check_name="row_count",
            table=self.table_name,
            passed=min_rows <= n <= max_rows,
            expected=f"{min_rows:,} ≤ rows ≤ {max_rows:,}",
            actual=f"{n:,} rows",
            severity=severity,
        ))

    def expect_referential_integrity(self, df: pd.DataFrame, column: str,
                                      ref_df: pd.DataFrame, ref_column: str,
                                      severity: str = "warning") -> QualityResult:
        """Check that all values in column exist in ref_df[ref_column]."""
        valid_ids = set(ref_df[ref_column].dropna())
        invalid = ~df[column].isin(valid_ids)
        invalid_count = int(invalid.sum())
        return self._add(QualityResult(
            check_name=f"referential_integrity_{column}_to_{ref_column}",
            table=self.table_name,
            passed=invalid_count == 0,
            expected="all values in reference set",
            actual=f"{invalid_count} orphaned values",
            severity=severity,
            rows_affected=invalid_count,
        ))

    def expect_freshness(self, df: pd.DataFrame, timestamp_col: str,
                          max_hours_old: int = 26,
                          severity: str = "warning") -> QualityResult:
        """Data should not be stale (latest event within max_hours_old)."""
        latest = pd.to_datetime(df[timestamp_col]).max()
        hours_old = (datetime.utcnow() - latest.replace(tzinfo=None)).total_seconds() / 3600
        return self._add(QualityResult(
            check_name=f"freshness_{timestamp_col}",
            table=self.table_name,
            passed=hours_old <= max_hours_old,
            expected=f"data ≤ {max_hours_old}h old",
            actual=f"latest data is {hours_old:.1f}h old ({latest.date()})",
            severity=severity,
        ))

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        errors = sum(1 for r in self.results if not r.passed and r.severity == "error")
        warnings = sum(1 for r in self.results if not r.passed and r.severity == "warning")

        return {
            "table": self.table_name,
            "total_checks": total,
            "passed": passed,
            "failed_errors": errors,
            "failed_warnings": warnings,
            "pass_rate": round(passed / max(total, 1) * 100, 2),
            "overall_status": "PASS" if errors == 0 else "FAIL",
            "checked_at": datetime.utcnow().isoformat(),
        }


def validate_silver(df: pd.DataFrame) -> dict:
    """Run full quality suite on Silver events table."""
    v = DataQualityValidator("silver_events")

    v.expect_row_count_between(df, min_rows=1000, max_rows=10_000_000)
    v.expect_no_nulls(df, "event_id")
    v.expect_no_nulls(df, "user_id")
    v.expect_no_nulls(df, "timestamp")
    v.expect_no_duplicates(df, "event_id")
    v.expect_values_in_set(df, "event_type", {
        "page_view", "add_to_cart", "remove_from_cart",
        "checkout_start", "purchase", "search", "login", "logout",
    })
    v.expect_values_in_set(df, "device", {"mobile", "desktop", "tablet"})
    v.expect_column_min(df, "revenue", min_val=0.0)
    v.expect_column_max(df, "revenue", max_val=100_000)
    v.expect_null_rate_below(df, "product_id", max_rate=0.50)
    v.expect_freshness(df, "timestamp", max_hours_old=48)

    for r in v.results:
        print(r)

    return v.summary()


def validate_gold_revenue(df: pd.DataFrame) -> dict:
    """Quality checks on Gold daily_revenue table."""
    v = DataQualityValidator("gold_daily_revenue")

    v.expect_no_nulls(df, "event_date")
    v.expect_no_nulls(df, "gross_revenue")
    v.expect_column_min(df, "gross_revenue", min_val=0.0)
    v.expect_column_min(df, "orders", min_val=0)
    v.expect_column_min(df, "avg_order_value", min_val=0.0)

    for r in v.results:
        print(r)

    return v.summary()


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

    silver_parts = sorted(Path("data/silver").glob("event_date=*/data.parquet"))
    if silver_parts:
        df = pd.concat([pd.read_parquet(p) for p in silver_parts[:5]])
        result = validate_silver(df)
        print(f"\nSummary: {result}")
    else:
        print("No silver data found. Run the pipeline first.")
