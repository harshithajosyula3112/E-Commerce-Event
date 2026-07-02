"""
src/transformation/silver_transformer.py
-----------------------------------------
Silver layer: cleans, deduplicates, and enriches Bronze events.

Transformations:
  1. Deduplication — remove exact duplicate event_ids (idempotent)
  2. Null handling — fill per business rules, drop unfixable
  3. Type enrichment — add time features, session sequence numbers
  4. Session reconstruction — assign event order within sessions
  5. Business rules — e.g., revenue must = price × quantity for purchases

Output is analysis-ready, deduplicated, and enriched.

Usage:
    python src/transformation/silver_transformer.py
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)


class SilverTransformer:

    def __init__(self, bronze_dir: Path = BRONZE_DIR, silver_dir: Path = SILVER_DIR):
        self.bronze_dir = bronze_dir
        self.silver_dir = silver_dir

    def load_bronze(self) -> pd.DataFrame:
        """Load all Bronze partitions into a single DataFrame."""
        partitions = sorted(self.bronze_dir.glob("event_date=*/data.parquet"))
        if not partitions:
            raise FileNotFoundError(f"No Bronze partitions found in {self.bronze_dir}. "
                                    "Run ingestion first.")
        dfs = [pd.read_parquet(p) for p in partitions]
        df = pd.concat(dfs, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        log.info(f"Loaded Bronze: {len(df):,} events from {len(partitions)} partitions")
        return df

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate event_ids. Keep first occurrence by timestamp.
        This handles re-ingestion scenarios (idempotent pipeline).
        """
        n_before = len(df)
        df = df.sort_values("timestamp").drop_duplicates(subset=["event_id"], keep="first")
        n_after = len(df)
        n_dupes = n_before - n_after
        log.info(f"Deduplication: removed {n_dupes:,} duplicates "
                 f"({n_dupes/max(n_before,1)*100:.2f}% of records)")
        return df

    def handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply business rules for null handling."""
        df = df.copy()

        # Non-purchase events: null product fields are expected
        df["product_id"] = df["product_id"].fillna("NONE")
        df["category"] = df["category"].fillna("Unknown")
        df["order_id"] = df["order_id"].fillna("NONE")

        # Numeric defaults
        df["price"] = df["price"].fillna(0.0)
        df["quantity"] = df["quantity"].fillna(0.0)
        df["revenue"] = df["revenue"].fillna(0.0)
        df["rating"] = df["rating"].fillna(df["rating"].median())

        # Drop if critical fields null
        n_before = len(df)
        df = df.dropna(subset=["event_id", "user_id", "timestamp", "event_type"])
        n_dropped = n_before - len(df)
        if n_dropped:
            log.warning(f"Dropped {n_dropped} rows with null critical fields")

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with time-based features for downstream analytics."""
        df = df.copy()
        df["event_date"] = df["timestamp"].dt.date.astype(str)
        df["event_hour"] = df["timestamp"].dt.hour
        df["event_dow"] = df["timestamp"].dt.day_name()
        df["event_week"] = df["timestamp"].dt.isocalendar().week.astype(int)
        df["event_month"] = df["timestamp"].dt.month
        df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
        df["is_peak_hour"] = df["event_hour"].between(18, 23)
        return df

    def reconstruct_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Number events within each session and compute time-since-last-event.
        This enables session-level metrics (session length, time-to-purchase, etc.)
        """
        df = df.copy().sort_values(["user_id", "session_id", "timestamp"])

        # Event sequence number within session
        df["session_event_seq"] = df.groupby("session_id").cumcount() + 1

        # Time since last event in session (seconds)
        df["seconds_since_last_event"] = (
            df.groupby("session_id")["timestamp"]
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )

        # Session depth at time of purchase
        df["is_first_session_event"] = df["session_event_seq"] == 1

        log.info("Session reconstruction complete")
        return df

    def enforce_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix business-level data quality issues.
        Separate from schema validation — these are semantic rules.
        """
        df = df.copy()
        fixes = 0

        # Rule 1: Purchase revenue = price × quantity (recalculate if diverges)
        purchase_mask = df["event_type"] == "purchase"
        expected_revenue = df.loc[purchase_mask, "price"] * df.loc[purchase_mask, "quantity"]
        revenue_discrepancy = (
            (df.loc[purchase_mask, "revenue"] - expected_revenue).abs() > 0.01
        )
        if revenue_discrepancy.sum() > 0:
            df.loc[purchase_mask & revenue_discrepancy, "revenue"] = expected_revenue[revenue_discrepancy]
            fixes += revenue_discrepancy.sum()

        # Rule 2: Non-purchase events should have zero revenue
        non_purchase_mask = df["event_type"] != "purchase"
        df.loc[non_purchase_mask, "revenue"] = 0.0

        # Rule 3: Quantity must be positive for purchases
        invalid_qty = purchase_mask & (df["quantity"] <= 0)
        df.loc[invalid_qty, "quantity"] = 1.0
        fixes += invalid_qty.sum()

        if fixes:
            log.info(f"Business rule fixes applied: {fixes} corrections")

        return df

    def add_silver_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_silver_processed_at"] = datetime.utcnow().isoformat()
        df["_silver_version"] = "1.0.0"
        return df

    def write_silver(self, df: pd.DataFrame) -> None:
        """Write Silver as date-partitioned Parquet."""
        dates = df["event_date"].unique()
        for event_date in dates:
            partition_df = df[df["event_date"] == event_date]
            out_dir = self.silver_dir / f"event_date={event_date}"
            out_dir.mkdir(parents=True, exist_ok=True)
            partition_df.to_parquet(out_dir / "data.parquet", index=False)
        log.info(f"Silver written: {len(df):,} records across {len(dates)} partitions")

    def run(self) -> dict:
        log.info("=== SILVER TRANSFORMATION STARTED ===")
        start = datetime.utcnow()

        df = self.load_bronze()
        df = self.deduplicate(df)
        df = self.handle_nulls(df)
        df = self.add_time_features(df)
        df = self.reconstruct_sessions(df)
        df = self.enforce_business_rules(df)
        df = self.add_silver_metadata(df)
        self.write_silver(df)

        duration = (datetime.utcnow() - start).total_seconds()
        summary = {
            "status": "success",
            "silver_records": len(df),
            "unique_users": df["user_id"].nunique(),
            "unique_sessions": df["session_id"].nunique(),
            "total_revenue": round(df["revenue"].sum(), 2),
            "duration_seconds": round(duration, 2),
        }
        log.info(f"=== SILVER COMPLETE in {duration:.1f}s === {summary}")
        return summary


if __name__ == "__main__":
    transformer = SilverTransformer()
    transformer.run()
