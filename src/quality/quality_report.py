"""
src/quality/quality_report.py
------------------------------
Generates a full data quality report across all pipeline layers.
Called by the Airflow DAG after Gold aggregation.

Usage:
    python src/quality/quality_report.py
"""

import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPORTS_DIR = Path("data/quality_reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def check_layer(layer_name: str, data_dir: Path) -> dict:
    """Check a single pipeline layer: file count, row count, size."""
    partitions = list(data_dir.glob("**/*.parquet")) if data_dir.exists() else []
    if not partitions:
        return {"layer": layer_name, "status": "MISSING", "files": 0, "rows": 0, "size_mb": 0}

    total_rows = 0
    total_size = 0
    for p in partitions:
        try:
            df = pd.read_parquet(p)
            total_rows += len(df)
            total_size += p.stat().st_size
        except Exception:
            pass

    return {
        "layer": layer_name,
        "status": "OK",
        "files": len(partitions),
        "rows": total_rows,
        "size_mb": round(total_size / 1e6, 2),
    }


def check_gold_tables() -> list[dict]:
    """Validate all Gold tables exist and have data."""
    gold_dir = Path("data/gold")
    expected = [
        "daily_revenue", "daily_revenue_total", "funnel_metrics",
        "user_segments", "product_performance", "cohort_summary",
    ]
    results = []
    for table in expected:
        path = gold_dir / f"{table}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            results.append({
                "table": table, "status": "OK",
                "rows": len(df), "columns": len(df.columns),
                "null_pct": round(df.isnull().mean().mean() * 100, 2),
            })
        else:
            results.append({"table": table, "status": "MISSING", "rows": 0})
    return results


def check_silver_quality() -> dict:
    """Run basic quality checks on Silver data."""
    silver_parts = sorted(Path("data/silver").glob("event_date=*/data.parquet"))
    if not silver_parts:
        return {"status": "MISSING"}

    # Sample last 3 partitions for speed
    sample_parts = silver_parts[-3:]
    df = pd.concat([pd.read_parquet(p) for p in sample_parts], ignore_index=True)

    valid_types = {"page_view", "add_to_cart", "remove_from_cart",
                   "checkout_start", "purchase", "search", "login", "logout"}

    return {
        "status": "OK",
        "sampled_rows": len(df),
        "duplicate_event_ids": int(df["event_id"].duplicated().sum()),
        "null_user_id_pct": round(df["user_id"].isna().mean() * 100, 3),
        "invalid_event_types": int((~df["event_type"].isin(valid_types)).sum()),
        "negative_revenue_rows": int((df["revenue"] < 0).sum()),
        "unique_users": df["user_id"].nunique(),
        "unique_sessions": df["session_id"].nunique(),
        "event_type_counts": df["event_type"].value_counts().to_dict(),
    }


def generate_report() -> dict:
    """Build and save the full quality report."""
    log.info("Generating data quality report...")

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "pipeline_version": "1.0.0",
        "layers": {
            "bronze": check_layer("bronze", Path("data/bronze")),
            "silver": check_layer("silver", Path("data/silver")),
            "gold":   check_layer("gold",   Path("data/gold")),
        },
        "gold_tables": check_gold_tables(),
        "silver_quality": check_silver_quality(),
    }

    # Overall pass/fail
    layer_ok = all(v["status"] == "OK" for v in report["layers"].values())
    gold_ok  = all(t["status"] == "OK" for t in report["gold_tables"])
    silver_q = report["silver_quality"]
    silver_ok = (
        silver_q.get("status") == "OK"
        and silver_q.get("duplicate_event_ids", 1) == 0
        and silver_q.get("negative_revenue_rows", 1) == 0
        and silver_q.get("invalid_event_types", 1) == 0
    )

    report["overall_status"] = "PASS" if (layer_ok and gold_ok and silver_ok) else "WARN"

    # Save JSON report
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"quality_report_{ts}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(f"Quality report saved → {out}")
    log.info(f"Overall status: {report['overall_status']}")

    # Print summary to stdout
    print(f"\n{'='*55}")
    print(f"  PIPELINE QUALITY REPORT — {report['overall_status']}")
    print(f"{'='*55}")
    for layer, stats in report["layers"].items():
        print(f"  {layer.upper():8s}: {stats['status']:8s} | "
              f"{stats['rows']:>8,} rows | {stats['size_mb']:.1f} MB")
    print(f"\n  Gold Tables:")
    for t in report["gold_tables"]:
        print(f"  {'✅' if t['status']=='OK' else '❌'} {t['table']:<30} {t.get('rows',0):>6,} rows")

    sq = report["silver_quality"]
    if sq.get("status") == "OK":
        print(f"\n  Silver Quality:")
        print(f"  Duplicate event IDs : {sq['duplicate_event_ids']}")
        print(f"  Invalid event types : {sq['invalid_event_types']}")
        print(f"  Negative revenue    : {sq['negative_revenue_rows']}")
        print(f"  Null user_id %      : {sq['null_user_id_pct']}%")
    print(f"{'='*55}\n")

    return report


if __name__ == "__main__":
    generate_report()
