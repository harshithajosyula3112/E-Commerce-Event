"""
sql/run_queries.py
------------------
Runs all analytics SQL queries against the Gold/Silver Parquet files
using DuckDB (no server required — reads Parquet directly).

Usage:
    python sql/run_queries.py
"""

import logging
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


def run_with_duckdb():
    """Run queries using DuckDB against Parquet files."""
    con = duckdb.connect()

    # Register parquet files as virtual tables
    silver_dir = Path("data/silver")
    gold_dir = Path("data/gold")

    silver_parts = list(silver_dir.glob("event_date=*/data.parquet"))
    if silver_parts:
        con.execute(f"""
            CREATE VIEW silver_events AS
            SELECT * FROM read_parquet('{silver_dir}/event_date=*/data.parquet',
                                       hive_partitioning=true)
        """)
        log.info(f"Registered silver_events view ({len(silver_parts)} partitions)")

    for table in ["daily_revenue", "funnel_metrics", "user_segments",
                  "product_performance", "cohort_summary", "daily_revenue_total"]:
        path = gold_dir / f"{table}.parquet"
        if path.exists():
            con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')")
            log.info(f"Registered {table} view")

    return con


def run_session_analysis(con) -> pd.DataFrame:
    """Session analysis — window functions on silver events."""
    sql = """
    WITH session_summary AS (
        SELECT
            session_id,
            device,
            traffic_source,
            COUNT(*) AS total_events,
            SUM(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) AS page_views,
            SUM(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS cart_adds,
            SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases,
            SUM(revenue) AS session_revenue
        FROM silver_events
        GROUP BY session_id, device, traffic_source
    )
    SELECT
        device,
        traffic_source,
        COUNT(*) AS total_sessions,
        ROUND(AVG(total_events), 2) AS avg_session_depth,
        ROUND(SUM(CASE WHEN page_views = 1 AND cart_adds = 0 THEN 1.0 ELSE 0 END)
              / COUNT(*) * 100, 2) AS bounce_rate_pct,
        ROUND(SUM(CASE WHEN purchases > 0 THEN 1.0 ELSE 0 END)
              / COUNT(*) * 100, 3) AS conversion_rate_pct,
        ROUND(SUM(session_revenue), 2) AS total_revenue
    FROM session_summary
    GROUP BY device, traffic_source
    ORDER BY conversion_rate_pct DESC
    LIMIT 20
    """
    return con.execute(sql).df()


def run_funnel_analysis(con) -> pd.DataFrame:
    """Conversion funnel with rolling averages."""
    sql = """
    SELECT
        event_date,
        SUM(page_views) AS total_views,
        SUM(add_to_carts) AS total_cart_adds,
        SUM(checkouts) AS total_checkouts,
        SUM(purchases) AS total_purchases,
        ROUND(AVG(cart_rate), 2) AS avg_cart_rate,
        ROUND(AVG(checkout_rate), 2) AS avg_checkout_rate,
        ROUND(AVG(purchase_rate), 2) AS avg_purchase_rate,
        ROUND(AVG(overall_conversion), 3) AS avg_overall_conversion
    FROM funnel_metrics
    GROUP BY event_date
    ORDER BY event_date
    """
    return con.execute(sql).df()


def run_revenue_by_category(con) -> pd.DataFrame:
    """Revenue breakdown by category and device."""
    sql = """
    SELECT
        category,
        device,
        SUM(gross_revenue) AS total_revenue,
        SUM(orders) AS total_orders,
        ROUND(AVG(avg_order_value), 2) AS avg_order_value,
        SUM(unique_buyers) AS unique_buyers
    FROM daily_revenue
    GROUP BY category, device
    ORDER BY total_revenue DESC
    """
    return con.execute(sql).df()


def run_top_products(con) -> pd.DataFrame:
    """Top performing products by revenue and conversion."""
    sql = """
    SELECT
        product_id,
        category,
        total_revenue,
        purchases,
        page_views,
        ROUND(conversion_rate, 2) AS conversion_rate,
        ROUND(avg_price, 2) AS avg_price
    FROM product_performance
    ORDER BY total_revenue DESC
    LIMIT 20
    """
    return con.execute(sql).df()


def run_user_segment_analysis(con) -> pd.DataFrame:
    """User segment distribution and revenue."""
    sql = """
    SELECT
        rfm_segment,
        COUNT(*) AS user_count,
        ROUND(AVG(total_revenue), 2) AS avg_revenue,
        ROUND(AVG(total_orders), 2) AS avg_orders,
        ROUND(SUM(total_revenue), 2) AS segment_revenue,
        primary_device
    FROM user_segments
    GROUP BY rfm_segment, primary_device
    ORDER BY avg_revenue DESC
    """
    return con.execute(sql).df()


def main():
    if not DUCKDB_AVAILABLE:
        log.error("duckdb not installed. Run: pip install duckdb")
        return

    silver_parts = list(Path("data/silver").glob("event_date=*/data.parquet"))
    gold_files = list(Path("data/gold").glob("*.parquet"))

    if not silver_parts and not gold_files:
        log.error("No data found. Run the pipeline first:\n"
                  "  python src/ingestion/event_generator.py\n"
                  "  python src/ingestion/ingest.py\n"
                  "  python src/transformation/silver_transformer.py\n"
                  "  python src/transformation/gold_aggregator.py")
        return

    con = run_with_duckdb()

    queries = {
        "Session Analysis": run_session_analysis,
        "Funnel Analysis": run_funnel_analysis,
        "Revenue by Category": run_revenue_by_category,
        "Top Products": run_top_products,
        "User Segments": run_user_segment_analysis,
    }

    results = {}
    for name, fn in queries.items():
        try:
            df = fn(con)
            results[name] = df
            print(f"\n{'='*60}")
            print(f"  {name} ({len(df)} rows)")
            print(f"{'='*60}")
            print(df.to_string(index=False))
        except Exception as e:
            log.warning(f"Query '{name}' failed: {e}")

    return results


if __name__ == "__main__":
    main()
