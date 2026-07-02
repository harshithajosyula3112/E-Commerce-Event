"""
tests/test_pipeline.py
-----------------------
Unit + integration tests for the e-commerce pipeline.

Run: pytest tests/ -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Event Generator Tests ────────────────────────────────────────────────────

class TestEventGenerator:

    def test_product_catalog_shape(self):
        from ingestion.event_generator import generate_product_catalog
        df = generate_product_catalog(100)
        assert len(df) == 100
        assert set(["product_id", "category", "price", "rating"]).issubset(df.columns)

    def test_product_prices_valid(self):
        from ingestion.event_generator import generate_product_catalog
        df = generate_product_catalog(200)
        assert (df["price"] > 0).all()
        assert (df["price"] < 10_000).all()

    def test_ratings_in_range(self):
        from ingestion.event_generator import generate_product_catalog
        df = generate_product_catalog(200)
        assert (df["rating"] >= 1.0).all()
        assert (df["rating"] <= 5.0).all()

    def test_users_generated(self):
        from ingestion.event_generator import generate_users
        df = generate_users(500)
        assert len(df) == 500
        assert "user_id" in df.columns
        assert "segment" in df.columns
        valid_segments = {"high_value", "regular", "occasional", "new"}
        assert set(df["segment"].unique()).issubset(valid_segments)

    def test_generate_events_small(self):
        from ingestion.event_generator import generate_events
        df, products, users = generate_events(n_days=3, target_events=500, seed=123)
        assert len(df) > 0
        assert "event_type" in df.columns
        assert "timestamp" in df.columns
        assert "user_id" in df.columns
        assert "revenue" in df.columns

    def test_events_have_valid_types(self):
        from ingestion.event_generator import generate_events
        df, _, _ = generate_events(n_days=2, target_events=300, seed=42)
        valid_types = {
            "page_view", "add_to_cart", "remove_from_cart",
            "checkout_start", "purchase", "search", "login", "logout",
        }
        assert set(df["event_type"].unique()).issubset(valid_types)

    def test_purchase_revenue_positive(self):
        from ingestion.event_generator import generate_events
        df, _, _ = generate_events(n_days=3, target_events=500, seed=42)
        purchases = df[df["event_type"] == "purchase"]
        if len(purchases) > 0:
            assert (purchases["revenue"] > 0).all()

    def test_timestamps_sorted(self):
        from ingestion.event_generator import generate_events
        df, _, _ = generate_events(n_days=2, target_events=200, seed=42)
        ts = pd.to_datetime(df["timestamp"])
        assert (ts.diff().dropna() >= pd.Timedelta(0)).all()

    def test_funnel_ordering(self):
        """More page views than purchases (realistic funnel drop-off)."""
        from ingestion.event_generator import generate_events
        df, _, _ = generate_events(n_days=5, target_events=2000, seed=42)
        views = (df["event_type"] == "page_view").sum()
        purchases = (df["event_type"] == "purchase").sum()
        assert views > purchases, f"Expected views > purchases, got {views} vs {purchases}"


# ── Bronze Ingester Tests ────────────────────────────────────────────────────

class TestBronzeIngester:

    def _make_sample_df(self, n=100):
        """Create minimal valid events DataFrame."""
        import uuid
        from datetime import datetime, timedelta
        rng = np.random.default_rng(0)
        now = datetime.utcnow()
        return pd.DataFrame({
            "event_id": [str(uuid.uuid4()) for _ in range(n)],
            "event_type": rng.choice(["page_view", "add_to_cart", "purchase"], n),
            "user_id": [f"U{i:04d}" for i in rng.integers(0, 50, n)],
            "session_id": [f"S{i:04d}" for i in rng.integers(0, 30, n)],
            "timestamp": [(now - timedelta(hours=int(h))).isoformat()
                          for h in rng.integers(0, 24, n)],
            "device": rng.choice(["mobile", "desktop"], n),
            "traffic_source": rng.choice(["organic", "paid"], n),
            "revenue": rng.uniform(0, 200, n),
            "product_id": [f"P{i:04d}" for i in rng.integers(0, 50, n)],
            "category": rng.choice(["Electronics", "Clothing"], n),
            "price": rng.uniform(10, 200, n),
            "quantity": rng.integers(1, 5, n).astype(float),
        })

    def test_schema_enforcement_valid(self):
        from ingestion.ingest import BronzeIngester
        ing = BronzeIngester()
        df = self._make_sample_df(100)
        valid, rejected = ing.enforce_schema(df)
        assert len(valid) + len(rejected) == len(df)
        assert len(rejected) < len(df) * 0.1  # Less than 10% rejected

    def test_schema_rejects_invalid_event_types(self):
        from ingestion.ingest import BronzeIngester
        ing = BronzeIngester()
        df = self._make_sample_df(10)
        df.loc[0, "event_type"] = "INVALID_TYPE"
        df.loc[1, "event_type"] = "ANOTHER_BAD_ONE"
        valid, rejected = ing.enforce_schema(df)
        assert len(rejected) >= 2

    def test_schema_rejects_null_event_id(self):
        from ingestion.ingest import BronzeIngester
        ing = BronzeIngester()
        df = self._make_sample_df(10)
        df.loc[0, "event_id"] = None
        valid, rejected = ing.enforce_schema(df)
        assert len(rejected) >= 1

    def test_metadata_columns_added(self):
        from ingestion.ingest import BronzeIngester
        ing = BronzeIngester()
        df = self._make_sample_df(10)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df_meta = ing.add_bronze_metadata(df)
        assert "_ingested_at" in df_meta.columns
        assert "_pipeline_version" in df_meta.columns
        assert "event_date" in df_meta.columns

    def test_deduplication_removes_dupes(self):
        from ingestion.ingest import BronzeIngester
        ing = BronzeIngester()
        df = self._make_sample_df(10)
        # Force duplicate event_id
        df.loc[5, "event_id"] = df.loc[0, "event_id"]
        valid, rejected = ing.enforce_schema(df)
        # After schema check, check dedup behavior in silver (which calls dedupe)


# ── Silver Transformer Tests ─────────────────────────────────────────────────

class TestSilverTransformer:

    def _make_bronze_df(self, n=100):
        import uuid
        from datetime import datetime, timedelta
        rng = np.random.default_rng(1)
        now = datetime.utcnow()
        return pd.DataFrame({
            "event_id": [str(uuid.uuid4()) for _ in range(n)],
            "event_type": rng.choice(["page_view", "add_to_cart", "purchase"], n),
            "user_id": [f"U{i:04d}" for i in rng.integers(0, 20, n)],
            "session_id": [f"S{i:04d}" for i in rng.integers(0, 15, n)],
            "timestamp": pd.date_range(now - timedelta(hours=n), periods=n, freq="1min"),
            "device": rng.choice(["mobile", "desktop", "tablet"], n),
            "traffic_source": rng.choice(["organic", "paid_search"], n),
            "product_id": [f"P{i:04d}" for i in rng.integers(0, 30, n)],
            "category": rng.choice(["Electronics", "Clothing"], n),
            "price": rng.uniform(10, 300, n),
            "quantity": rng.integers(1, 4, n).astype(float),
            "revenue": rng.uniform(10, 900, n),
            "order_id": [f"ORD{i}" for i in range(n)],
            "rating": rng.uniform(3, 5, n),
            "_ingested_at": [now.isoformat()] * n,
            "_pipeline_version": ["1.0.0"] * n,
            "event_date": [(now - timedelta(hours=i % 24)).strftime("%Y-%m-%d") for i in range(n)],
            "event_hour": rng.integers(0, 24, n),
        })

    def test_deduplication(self):
        from transformation.silver_transformer import SilverTransformer
        st = SilverTransformer()
        df = self._make_bronze_df(50)
        # Add duplicate
        dupe = df.iloc[0:1].copy()
        df = pd.concat([df, dupe], ignore_index=True)
        deduped = st.deduplicate(df)
        assert len(deduped) == 50

    def test_null_handling_fills_product_id(self):
        from transformation.silver_transformer import SilverTransformer
        st = SilverTransformer()
        df = self._make_bronze_df(20)
        df.loc[0, "product_id"] = None
        cleaned = st.handle_nulls(df)
        assert cleaned["product_id"].isna().sum() == 0

    def test_time_features_added(self):
        from transformation.silver_transformer import SilverTransformer
        st = SilverTransformer()
        df = self._make_bronze_df(20)
        enriched = st.add_time_features(df)
        assert "is_weekend" in enriched.columns
        assert "is_peak_hour" in enriched.columns
        assert "event_dow" in enriched.columns

    def test_session_sequence_numbers(self):
        from transformation.silver_transformer import SilverTransformer
        st = SilverTransformer()
        df = self._make_bronze_df(30)
        sessioned = st.reconstruct_sessions(df)
        assert "session_event_seq" in sessioned.columns
        # Minimum sequence should be 1
        assert sessioned["session_event_seq"].min() == 1

    def test_business_rules_revenue(self):
        from transformation.silver_transformer import SilverTransformer
        st = SilverTransformer()
        df = self._make_bronze_df(20)
        df["event_type"] = "purchase"
        df["price"] = 50.0
        df["quantity"] = 2.0
        df["revenue"] = 999.0  # Wrong — should be 100
        fixed = st.enforce_business_rules(df)
        # Revenue should be corrected to price * quantity
        assert (fixed["revenue"] - 100.0).abs().max() < 0.01


# ── Data Quality Tests ───────────────────────────────────────────────────────

class TestDataQuality:

    def _make_df(self):
        import uuid
        return pd.DataFrame({
            "event_id": [str(uuid.uuid4()) for _ in range(50)],
            "user_id": [f"U{i}" for i in range(50)],
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
            "event_type": ["page_view"] * 50,
            "device": ["mobile"] * 50,
            "revenue": [0.0] * 50,
        })

    def test_no_nulls_passes(self):
        from quality.validator import DataQualityValidator
        v = DataQualityValidator("test")
        df = self._make_df()
        result = v.expect_no_nulls(df, "event_id")
        assert result.passed

    def test_no_nulls_fails(self):
        from quality.validator import DataQualityValidator
        v = DataQualityValidator("test")
        df = self._make_df()
        df.loc[0, "event_id"] = None
        result = v.expect_no_nulls(df, "event_id")
        assert not result.passed

    def test_values_in_set_passes(self):
        from quality.validator import DataQualityValidator
        v = DataQualityValidator("test")
        df = self._make_df()
        result = v.expect_values_in_set(df, "device", {"mobile", "desktop", "tablet"})
        assert result.passed

    def test_values_in_set_fails(self):
        from quality.validator import DataQualityValidator
        v = DataQualityValidator("test")
        df = self._make_df()
        df.loc[0, "device"] = "smartwatch"
        result = v.expect_values_in_set(df, "device", {"mobile", "desktop", "tablet"})
        assert not result.passed

    def test_column_min_passes(self):
        from quality.validator import DataQualityValidator
        v = DataQualityValidator("test")
        df = self._make_df()
        result = v.expect_column_min(df, "revenue", 0.0)
        assert result.passed

    def test_column_min_fails(self):
        from quality.validator import DataQualityValidator
        v = DataQualityValidator("test")
        df = self._make_df()
        df.loc[0, "revenue"] = -50.0
        result = v.expect_column_min(df, "revenue", 0.0)
        assert not result.passed

    def test_summary_all_pass(self):
        from quality.validator import DataQualityValidator
        v = DataQualityValidator("test")
        df = self._make_df()
        v.expect_no_nulls(df, "event_id")
        v.expect_no_nulls(df, "user_id")
        v.expect_column_min(df, "revenue", 0.0)
        summary = v.summary()
        assert summary["passed"] == 3
        assert summary["overall_status"] == "PASS"

    def test_row_count_check(self):
        from quality.validator import DataQualityValidator
        v = DataQualityValidator("test")
        df = self._make_df()
        result = v.expect_row_count_between(df, 10, 1000)
        assert result.passed

    def test_no_duplicates(self):
        from quality.validator import DataQualityValidator
        import uuid
        v = DataQualityValidator("test")
        df = self._make_df()
        result = v.expect_no_duplicates(df, "event_id")
        assert result.passed


# ── Integration Test ─────────────────────────────────────────────────────────

class TestEndToEnd:

    def test_full_pipeline_small(self, tmp_path):
        """Run the entire pipeline with small data to verify integration."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        from ingestion.event_generator import generate_events
        from ingestion.ingest import BronzeIngester
        from transformation.silver_transformer import SilverTransformer
        from transformation.gold_aggregator import GoldAggregator

        # Set up temp directories
        bronze_dir = tmp_path / "bronze"
        silver_dir = tmp_path / "silver"
        gold_dir = tmp_path / "gold"
        raw_dir = tmp_path / "raw"

        raw_dir.mkdir()
        bronze_dir.mkdir()
        silver_dir.mkdir()
        gold_dir.mkdir()

        # Generate small dataset
        df_events, _, _ = generate_events(n_days=3, target_events=1000, seed=99)
        df_events.to_parquet(raw_dir / "events_raw.parquet", index=False)

        # Bronze
        ingester = BronzeIngester(raw_dir=raw_dir, bronze_dir=bronze_dir)
        bronze_result = ingester.run()
        assert bronze_result["status"] == "success"
        assert bronze_result["total_valid"] > 0

        # Silver
        transformer = SilverTransformer(bronze_dir=bronze_dir, silver_dir=silver_dir)
        silver_result = transformer.run()
        assert silver_result["status"] == "success"

        # Gold
        aggregator = GoldAggregator(silver_dir=silver_dir, gold_dir=gold_dir)
        gold_result = aggregator.run()
        assert gold_result["status"] == "success"

        # Verify Gold files exist and have data
        for table in ["daily_revenue", "funnel_metrics", "user_segments"]:
            path = gold_dir / f"{table}.parquet"
            assert path.exists(), f"Gold table missing: {table}"
            df = pd.read_parquet(path)
            assert len(df) > 0, f"Gold table empty: {table}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
