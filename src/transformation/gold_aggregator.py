"""
src/transformation/gold_aggregator.py
---------------------------------------
Gold layer: business-ready aggregates for analytics and BI.

Produces 5 Gold tables:
  1. daily_revenue       — Revenue KPIs by day, category, device
  2. funnel_metrics      — Daily conversion funnel rates
  3. user_segments       — User-level behavioral summary
  4. product_performance — Product-level sales + engagement metrics
  5. cohort_summary      — Weekly cohort acquisition + revenue

These power the dashboard and the SQL queries.

Usage:
    python src/transformation/gold_aggregator.py
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SILVER_DIR = Path("data/silver")
GOLD_DIR = Path("data/gold")
GOLD_DIR.mkdir(parents=True, exist_ok=True)


class GoldAggregator:

    def __init__(self, silver_dir=SILVER_DIR, gold_dir=GOLD_DIR):
        self.silver_dir = silver_dir
        self.gold_dir = gold_dir

    def load_silver(self) -> pd.DataFrame:
        parts = sorted(self.silver_dir.glob("event_date=*/data.parquet"))
        if not parts:
            raise FileNotFoundError(f"No Silver data in {self.silver_dir}")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        log.info(f"Loaded Silver: {len(df):,} events from {len(parts)} partitions")
        return df

    # ── Gold Table 1: Daily Revenue ───────────────────────────────────────────

    def build_daily_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """Revenue KPIs by date, category, and device."""
        purchases = df[df["event_type"] == "purchase"].copy()

        # By date + category
        daily = purchases.groupby(["event_date", "category", "device"]).agg(
            orders=("order_id", "nunique"),
            units_sold=("quantity", "sum"),
            gross_revenue=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            unique_buyers=("user_id", "nunique"),
        ).reset_index()

        daily["gross_revenue"] = daily["gross_revenue"].round(2)
        daily["avg_order_value"] = daily["avg_order_value"].round(2)

        # Day-over-day revenue change
        daily_total = daily.groupby("event_date")["gross_revenue"].sum().reset_index()
        daily_total["revenue_dod_pct"] = daily_total["gross_revenue"].pct_change() * 100
        daily_total["revenue_7d_avg"] = (
            daily_total["gross_revenue"].rolling(7, min_periods=1).mean().round(2)
        )
        daily_total["revenue_dod_pct"] = daily_total["revenue_dod_pct"].round(2)

        log.info(f"Daily revenue table: {len(daily):,} rows")
        return daily, daily_total

    # ── Gold Table 2: Funnel Metrics ──────────────────────────────────────────

    def build_funnel_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Daily conversion funnel: views → cart → checkout → purchase."""

        funnel = df.groupby(["event_date", "device"]).agg(
            page_views=("event_id", lambda x: (df.loc[x.index, "event_type"] == "page_view").sum()),
            add_to_carts=("event_id", lambda x: (df.loc[x.index, "event_type"] == "add_to_cart").sum()),
            checkouts=("event_id", lambda x: (df.loc[x.index, "event_type"] == "checkout_start").sum()),
            purchases=("event_id", lambda x: (df.loc[x.index, "event_type"] == "purchase").sum()),
            unique_sessions=("session_id", "nunique"),
        ).reset_index()

        funnel["cart_rate"] = (funnel["add_to_carts"] / funnel["page_views"].clip(lower=1) * 100).round(2)
        funnel["checkout_rate"] = (funnel["checkouts"] / funnel["add_to_carts"].clip(lower=1) * 100).round(2)
        funnel["purchase_rate"] = (funnel["purchases"] / funnel["checkouts"].clip(lower=1) * 100).round(2)
        funnel["overall_conversion"] = (funnel["purchases"] / funnel["page_views"].clip(lower=1) * 100).round(3)

        log.info(f"Funnel metrics table: {len(funnel):,} rows")
        return funnel

    # ── Gold Table 3: User Segments ───────────────────────────────────────────

    def build_user_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """User-level behavioral summary for segmentation analysis."""

        purchases = df[df["event_type"] == "purchase"]

        user_stats = df.groupby("user_id").agg(
            total_events=("event_id", "count"),
            total_sessions=("session_id", "nunique"),
            total_page_views=("event_type", lambda x: (x == "page_view").sum()),
            total_cart_adds=("event_type", lambda x: (x == "add_to_cart").sum()),
            first_event=("timestamp", "min"),
            last_event=("timestamp", "max"),
            devices_used=("device", "nunique"),
            primary_device=("device", lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown"),
            primary_category=("category", lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown"),
        ).reset_index()

        purchase_stats = purchases.groupby("user_id").agg(
            total_orders=("order_id", "nunique"),
            total_revenue=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            total_units=("quantity", "sum"),
        ).reset_index()

        user_df = user_stats.merge(purchase_stats, on="user_id", how="left")
        user_df["total_orders"] = user_df["total_orders"].fillna(0).astype(int)
        user_df["total_revenue"] = user_df["total_revenue"].fillna(0.0).round(2)
        user_df["avg_order_value"] = user_df["avg_order_value"].fillna(0.0).round(2)
        user_df["total_units"] = user_df["total_units"].fillna(0).astype(int)

        # Days active
        user_df["days_active"] = (
            (user_df["last_event"] - user_df["first_event"]).dt.total_seconds() / 86400
        ).round(1)

        # RFM-style segment — use rank-based scoring to avoid qcut bin edge issues
        rev_rank = user_df["total_revenue"].rank(pct=True).fillna(0)
        ord_rank = user_df["total_orders"].rank(pct=True).fillna(0)
        rfm_score = (rev_rank * 0.5 + ord_rank * 0.5) * 5  # scale 0–5

        conditions = [rfm_score >= 4, rfm_score >= 3, rfm_score >= 2]
        labels = ["High Value", "Medium Value", "Low Value"]
        user_df["rfm_segment"] = np.select(conditions, labels, default="Inactive")

        log.info(f"User segments table: {len(user_df):,} users")
        return user_df

    # ── Gold Table 4: Product Performance ────────────────────────────────────

    def build_product_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Product-level sales and engagement metrics."""
        product_events = df[df["product_id"] != "NONE"].copy()

        product_stats = product_events.groupby(["product_id", "category"]).agg(
            page_views=("event_type", lambda x: (x == "page_view").sum()),
            cart_adds=("event_type", lambda x: (x == "add_to_cart").sum()),
            purchases=("event_type", lambda x: (x == "purchase").sum()),
            total_revenue=("revenue", "sum"),
            total_units=("quantity", "sum"),
            unique_viewers=("user_id", "nunique"),
            avg_price=("price", "mean"),
            avg_rating=("rating", "mean"),
        ).reset_index()

        product_stats["conversion_rate"] = (
            product_stats["purchases"] / product_stats["page_views"].clip(lower=1) * 100
        ).round(2)
        product_stats["cart_rate"] = (
            product_stats["cart_adds"] / product_stats["page_views"].clip(lower=1) * 100
        ).round(2)
        product_stats["total_revenue"] = product_stats["total_revenue"].round(2)
        product_stats["avg_price"] = product_stats["avg_price"].round(2)
        product_stats["avg_rating"] = product_stats["avg_rating"].round(2)

        log.info(f"Product performance table: {len(product_stats):,} products")
        return product_stats

    # ── Gold Table 5: Cohort Summary ──────────────────────────────────────────

    def build_cohort_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Weekly acquisition cohort — users by first week, revenue over time."""

        first_seen = df.groupby("user_id")["timestamp"].min().reset_index()
        first_seen.columns = ["user_id", "first_event"]
        first_seen["acquisition_week"] = (
            first_seen["first_event"].dt.to_period("W").astype(str)
        )

        df_cohort = df.merge(first_seen[["user_id", "acquisition_week"]], on="user_id")
        df_cohort["activity_week"] = df_cohort["timestamp"].dt.to_period("W").astype(str)

        purchases = df_cohort[df_cohort["event_type"] == "purchase"]

        cohort = purchases.groupby(["acquisition_week", "activity_week"]).agg(
            active_buyers=("user_id", "nunique"),
            cohort_revenue=("revenue", "sum"),
            orders=("order_id", "nunique"),
        ).reset_index()

        # Cohort size (total users acquired that week)
        cohort_sizes = first_seen.groupby("acquisition_week")["user_id"].count().reset_index()
        cohort_sizes.columns = ["acquisition_week", "cohort_size"]
        cohort = cohort.merge(cohort_sizes, on="acquisition_week")
        cohort["retention_rate"] = (cohort["active_buyers"] / cohort["cohort_size"] * 100).round(2)
        cohort["cohort_revenue"] = cohort["cohort_revenue"].round(2)

        log.info(f"Cohort summary table: {len(cohort):,} rows")
        return cohort

    def write_gold(self, name: str, df: pd.DataFrame) -> None:
        out = self.gold_dir / f"{name}.parquet"
        df.to_parquet(out, index=False)
        log.info(f"Gold/{name}: {len(df):,} rows → {out}")

    def run(self) -> dict:
        log.info("=== GOLD AGGREGATION STARTED ===")
        start = datetime.utcnow()

        df = self.load_silver()

        daily, daily_total = self.build_daily_revenue(df)
        funnel = self.build_funnel_metrics(df)
        users = self.build_user_segments(df)
        products = self.build_product_performance(df)
        cohort = self.build_cohort_summary(df)

        self.write_gold("daily_revenue", daily)
        self.write_gold("daily_revenue_total", daily_total)
        self.write_gold("funnel_metrics", funnel)
        self.write_gold("user_segments", users)
        self.write_gold("product_performance", products)
        self.write_gold("cohort_summary", cohort)

        duration = (datetime.utcnow() - start).total_seconds()
        summary = {
            "status": "success",
            "gold_tables": 5,
            "duration_seconds": round(duration, 2),
            "total_revenue": round(daily["gross_revenue"].sum(), 2),
        }
        log.info(f"=== GOLD COMPLETE in {duration:.1f}s ===")
        return summary


if __name__ == "__main__":
    agg = GoldAggregator()
    agg.run()
