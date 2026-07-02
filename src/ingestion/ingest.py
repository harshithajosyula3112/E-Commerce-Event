"""
src/ingestion/ingest.py
------------------------
Bronze layer ingestion: reads raw events, enforces schema,
partitions by date, and writes Parquet files.

Design principles:
  - Append-only (never mutate raw data)
  - Schema enforcement (reject malformed records, don't silently drop)
  - Idempotent: re-running same day produces identical output
  - Partition by event_date for query efficiency

Usage:
    python src/ingestion/ingest.py
    python src/ingestion/ingest.py --date 2024-01-15  # reprocess specific date
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RAW_DIR = Path("data/raw")
BRONZE_DIR = Path("data/bronze")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

# ── Schema definition ─────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "event_id", "event_type", "user_id", "session_id",
    "timestamp", "device", "traffic_source",
]

VALID_EVENT_TYPES = {
    "page_view", "add_to_cart", "remove_from_cart",
    "checkout_start", "purchase", "search", "login", "logout",
}

SCHEMA = {
    "event_id":       "string",
    "event_type":     "string",
    "user_id":        "string",
    "session_id":     "string",
    "product_id":     "string",
    "category":       "string",
    "timestamp":      "datetime64[ns]",
    "device":         "string",
    "traffic_source": "string",
    "price":          "float64",
    "quantity":       "float64",
    "revenue":        "float64",
    "order_id":       "string",
    "rating":         "float64",
}


class BronzeIngester:
    """
    Handles Bronze layer ingestion with schema enforcement and partitioning.
    """

    def __init__(self, raw_dir: Path = RAW_DIR, bronze_dir: Path = BRONZE_DIR):
        self.raw_dir = raw_dir
        self.bronze_dir = bronze_dir
        self.ingestion_stats = {}

    def load_raw(self) -> pd.DataFrame:
        """Load raw event files from data/raw/."""
        raw_file = self.raw_dir / "events_raw.parquet"
        if not raw_file.exists():
            log.warning(f"Raw file not found at {raw_file}. Running generator...")
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from event_generator import generate_events
            df_events, _, _ = generate_events()
            df_events.to_parquet(raw_file, index=False)

        df = pd.read_parquet(raw_file)
        log.info(f"Loaded {len(df):,} raw events from {raw_file}")
        return df

    def enforce_schema(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply schema to DataFrame.
        Returns (valid_df, rejected_df).
        """
        n_input = len(df)
        rejected_rows = []

        # Check required columns
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add missing optional columns with nulls
        for col in SCHEMA:
            if col not in df.columns:
                df[col] = None

        # Cast types
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for col in ["price", "quantity", "revenue", "rating"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Validation rules
        invalid_event_type = ~df["event_type"].isin(VALID_EVENT_TYPES)
        null_timestamp = df["timestamp"].isna()
        null_user = df["user_id"].isna() | (df["user_id"] == "")
        null_event_id = df["event_id"].isna() | (df["event_id"] == "")
        negative_revenue = (df["revenue"] < 0).fillna(False)

        invalid_mask = (
            invalid_event_type | null_timestamp | null_user |
            null_event_id | negative_revenue
        )

        valid_df = df[~invalid_mask].copy()
        rejected_df = df[invalid_mask].copy()

        n_rejected = len(rejected_df)
        rejection_rate = n_rejected / max(n_input, 1) * 100

        log.info(f"Schema enforcement: {n_input:,} input → "
                 f"{len(valid_df):,} valid, {n_rejected:,} rejected "
                 f"({rejection_rate:.2f}% rejection rate)")

        if rejection_rate > 5:
            log.warning(f"HIGH REJECTION RATE: {rejection_rate:.1f}% — investigate upstream")

        return valid_df, rejected_df

    def add_bronze_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ingestion metadata columns to Bronze records."""
        df = df.copy()
        df["_ingested_at"] = datetime.utcnow().isoformat()
        df["_source_file"] = "events_raw.parquet"
        df["_pipeline_version"] = "1.0.0"
        df["event_date"] = df["timestamp"].dt.date.astype(str)
        df["event_hour"] = df["timestamp"].dt.hour
        return df

    def write_partitioned(self, df: pd.DataFrame) -> dict[str, int]:
        """
        Write Parquet files partitioned by event_date.
        Returns dict of {date: row_count}.
        """
        partition_stats = {}
        dates = df["event_date"].unique()

        for event_date in dates:
            partition_df = df[df["event_date"] == event_date]
            partition_path = self.bronze_dir / f"event_date={event_date}"
            partition_path.mkdir(parents=True, exist_ok=True)
            out_file = partition_path / "data.parquet"
            partition_df.to_parquet(out_file, index=False)
            partition_stats[event_date] = len(partition_df)

        log.info(f"Written {len(dates)} date partitions to {self.bronze_dir}")
        return partition_stats

    def write_rejected(self, rejected_df: pd.DataFrame) -> None:
        """Save rejected records for investigation."""
        if len(rejected_df) == 0:
            return
        out = self.bronze_dir / "_rejected" / f"rejected_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        rejected_df.to_parquet(out, index=False)
        log.warning(f"Saved {len(rejected_df)} rejected records → {out}")

    def run(self, target_date: str | None = None) -> dict:
        """
        Full Bronze ingestion run.
        If target_date is set, only processes that partition.
        """
        log.info("=== BRONZE LAYER INGESTION STARTED ===")
        start = datetime.utcnow()

        df_raw = self.load_raw()

        if target_date:
            df_raw["_ts"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")
            df_raw = df_raw[df_raw["_ts"].dt.date.astype(str) == target_date].drop("_ts", axis=1)
            log.info(f"Filtered to target_date={target_date}: {len(df_raw):,} events")

        valid_df, rejected_df = self.enforce_schema(df_raw)
        valid_df = self.add_bronze_metadata(valid_df)
        partition_stats = self.write_partitioned(valid_df)
        self.write_rejected(rejected_df)

        duration = (datetime.utcnow() - start).total_seconds()

        summary = {
            "status": "success",
            "total_input": len(df_raw),
            "total_valid": len(valid_df),
            "total_rejected": len(rejected_df),
            "partitions_written": len(partition_stats),
            "duration_seconds": round(duration, 2),
        }

        log.info(f"=== BRONZE COMPLETE in {duration:.1f}s === {summary}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Bronze layer ingestion")
    parser.add_argument("--date", help="Process specific date (YYYY-MM-DD)", default=None)
    args = parser.parse_args()

    ingester = BronzeIngester()
    ingester.run(target_date=args.date)


if __name__ == "__main__":
    main()
