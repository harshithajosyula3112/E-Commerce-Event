"""
dags/ecommerce_pipeline_dag.py
--------------------------------
Airflow DAG for the daily e-commerce data pipeline.

DAG: ecommerce_daily_pipeline
Schedule: Daily at 2 AM UTC (after midnight data cutoff)
SLA: All Gold tables ready by 6 AM UTC

Task graph:
    generate_events
         │
    ingest_bronze
         │
    validate_bronze
         │
    transform_silver
         │
    validate_silver
         │
    ┌────┴────────────────┐
    │                     │
aggregate_gold    run_quality_report
    │
validate_gold
    │
notify_success

Airflow setup:
    pip install apache-airflow
    airflow db init
    airflow dags trigger ecommerce_daily_pipeline
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
import logging

log = logging.getLogger(__name__)

# ── Default args ──────────────────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "sla": timedelta(hours=4),  # 4-hour SLA from DAG start
}


# ── Task functions ────────────────────────────────────────────────────────────

def task_generate_events(**context):
    """Generate synthetic event data (replace with real source in production)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ingestion"))
    from event_generator import generate_events
    
    execution_date = context["ds"]
    log.info(f"Generating events for {execution_date}")
    df_events, df_products, df_users = generate_events(n_days=1)
    log.info(f"Generated {len(df_events):,} events")
    return {"events_count": len(df_events)}


def task_ingest_bronze(**context):
    """Run Bronze layer ingestion."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ingestion"))
    from ingest import BronzeIngester

    execution_date = context["ds"]
    ingester = BronzeIngester()
    result = ingester.run(target_date=execution_date)

    if result["status"] != "success":
        raise ValueError(f"Bronze ingestion failed: {result}")

    log.info(f"Bronze: {result['total_valid']:,} records written")
    return result


def task_validate_bronze(**context):
    """Validate Bronze layer data quality."""
    import sys
    import pandas as pd
    from pathlib import Path

    execution_date = context["ds"]
    bronze_path = Path(f"data/bronze/event_date={execution_date}/data.parquet")

    if not bronze_path.exists():
        log.warning(f"Bronze partition not found for {execution_date}, skipping validation")
        return {"status": "skipped"}

    df = pd.read_parquet(bronze_path)
    log.info(f"Validating Bronze partition: {len(df):,} rows")

    null_rate = df["event_id"].isna().mean()
    dup_rate = df["event_id"].duplicated().mean()

    assert null_rate < 0.01, f"Bronze null rate too high: {null_rate:.2%}"
    assert dup_rate < 0.05, f"Bronze duplicate rate too high: {dup_rate:.2%}"

    return {"status": "pass", "rows": len(df)}


def task_transform_silver(**context):
    """Transform Bronze → Silver."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "transformation"))
    from silver_transformer import SilverTransformer

    transformer = SilverTransformer()
    result = transformer.run()

    if result["status"] != "success":
        raise ValueError(f"Silver transformation failed: {result}")

    return result


def task_validate_silver(**context):
    """Validate Silver layer data quality gates."""
    import sys
    import pandas as pd
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "quality"))
    from validator import validate_silver

    silver_parts = sorted(Path("data/silver").glob("event_date=*/data.parquet"))
    if not silver_parts:
        raise FileNotFoundError("No Silver data found")

    df = pd.concat([pd.read_parquet(p) for p in silver_parts[-7:]])  # last 7 days
    result = validate_silver(df)

    if result["overall_status"] == "FAIL":
        raise ValueError(f"Silver quality gate FAILED: {result}")

    log.info(f"Silver quality: {result['pass_rate']}% pass rate")
    return result


def task_aggregate_gold(**context):
    """Aggregate Silver → Gold business tables."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "transformation"))
    from gold_aggregator import GoldAggregator

    agg = GoldAggregator()
    result = agg.run()

    if result["status"] != "success":
        raise ValueError(f"Gold aggregation failed: {result}")

    return result


def task_quality_report(**context):
    """Generate and save full quality report."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "quality"))
    from quality_report import generate_report
    generate_report()
    return {"status": "complete"}


def task_validate_gold(**context):
    """Final Gold-layer quality validation before downstream systems consume."""
    import pandas as pd
    from pathlib import Path

    gold_tables = {
        "daily_revenue": Path("data/gold/daily_revenue.parquet"),
        "funnel_metrics": Path("data/gold/funnel_metrics.parquet"),
        "user_segments": Path("data/gold/user_segments.parquet"),
    }

    for table_name, path in gold_tables.items():
        if not path.exists():
            raise FileNotFoundError(f"Gold table missing: {path}")
        df = pd.read_parquet(path)
        if len(df) == 0:
            raise ValueError(f"Gold table empty: {table_name}")
        log.info(f"Gold/{table_name}: {len(df):,} rows ✅")

    return {"status": "all_gold_validated"}


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="ecommerce_daily_pipeline",
    description="Daily e-commerce Bronze→Silver→Gold medallion pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * *",  # 2 AM UTC daily
    catchup=False,
    max_active_runs=1,
    tags=["ecommerce", "data-engineering", "medallion"],
) as dag:

    start = EmptyOperator(task_id="pipeline_start")

    generate = PythonOperator(
        task_id="generate_events",
        python_callable=task_generate_events,
    )

    bronze = PythonOperator(
        task_id="ingest_bronze",
        python_callable=task_ingest_bronze,
    )

    validate_bronze = PythonOperator(
        task_id="validate_bronze",
        python_callable=task_validate_bronze,
    )

    silver = PythonOperator(
        task_id="transform_silver",
        python_callable=task_transform_silver,
    )

    validate_silver_task = PythonOperator(
        task_id="validate_silver",
        python_callable=task_validate_silver,
    )

    gold = PythonOperator(
        task_id="aggregate_gold",
        python_callable=task_aggregate_gold,
    )

    quality_report = PythonOperator(
        task_id="quality_report",
        python_callable=task_quality_report,
        trigger_rule=TriggerRule.ALL_DONE,  # run even if upstream fails
    )

    validate_gold_task = PythonOperator(
        task_id="validate_gold",
        python_callable=task_validate_gold,
    )

    end = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ── Task dependencies ─────────────────────────────────────────────────────
    (
        start
        >> generate
        >> bronze
        >> validate_bronze
        >> silver
        >> validate_silver_task
        >> [gold, quality_report]
    )
    gold >> validate_gold_task >> end
    quality_report >> end
