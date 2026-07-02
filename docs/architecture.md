# Architecture & Design Decisions

## Overview

This pipeline implements the **medallion architecture** (Bronze → Silver → Gold),
the industry-standard pattern used at Databricks, Meta, Amazon, and most large-scale
data engineering teams.

## Layer Design

### Bronze — Raw Ingestion
**What**: Append-only storage of raw events with minimal transformation.  
**Why**: Preserves raw data for re-processing. If a downstream bug corrupts Silver,
we can replay from Bronze without re-fetching source data.

Design decisions:
- **Parquet over CSV**: 10-30x compression, 100x faster analytical queries via columnar storage
- **Date partitioning**: `event_date=2024-01-15/data.parquet` enables predicate pushdown
  — queries filtered by date scan only relevant files (critical at 10M+ events/day)
- **Schema enforcement at ingestion**: Reject invalid records immediately with audit trail
  rather than letting bad data propagate downstream and corrupt business metrics

### Silver — Cleaned & Enriched
**What**: Deduplicated, validated, enriched events ready for analytics.  
**Why**: Separates "we have the data" from "the data is trustworthy."

Design decisions:
- **Idempotent transformations**: Re-running Silver on the same Bronze produces identical output.
  Critical for failure recovery — if a DAG run fails mid-way, we re-run safely.
- **Deduplication by event_id**: Events can be delivered more than once (at-least-once delivery
  semantics in real streaming systems). We always deduplicate at Silver.
- **Session reconstruction here, not Gold**: Session data is used by multiple Gold tables.
  Computing it once in Silver avoids redundant computation.

### Gold — Business Aggregates
**What**: Pre-aggregated, business-ready tables serving BI tools and ML feature stores.  
**Why**: BI dashboards can't query 100M raw events in real time. Pre-aggregation
enables sub-second dashboard queries.

Design decisions:
- **5 purpose-built tables**: Each table serves a specific business domain (revenue, funnel,
  users, products, cohorts). Avoids the "god table" anti-pattern.
- **Star schema**: Gold tables have clear fact/dimension separation, compatible with
  any BI tool (Tableau, Power BI, Looker, QuickSight).
- **Incremental aggregation opportunity**: Current design reprocesses all Silver.
  In production, we'd use `MERGE INTO` or dbt incremental models to only process
  new/changed Silver records — reducing daily runtime from 45s to ~5s.

## Scalability Path

| Current (local) | Production scale |
|---|---|
| pandas DataFrames | Apache Spark on EMR / Databricks |
| Local Parquet files | S3 / GCS data lake |
| Single-node DuckDB | Redshift / BigQuery / Snowflake |
| Manual Airflow trigger | Managed Airflow (MWAA / Cloud Composer) |
| ~100K events/day | 1B+ events/day |

The code structure supports this migration path: all transformations are written
as pandas-compatible operations that map directly to PySpark equivalents.

## Data Quality Strategy

Three-tier quality enforcement:

1. **Schema quality** (Bronze ingestion): reject malformed records
2. **Business rules** (Silver transformation): fix/flag semantic errors
3. **Statistical quality** (Silver validation DAG task): anomaly detection on distributions

This mirrors the approach used at Airbnb (Minerva), Uber (Databook), and
LinkedIn (DataHub).

## Trade-offs Documented

| Decision | Chose | Over | Because |
|---|---|---|---|
| Storage format | Parquet | CSV, JSON | 10-30x faster analytical queries |
| Partitioning | By date | By user_id | Date is most common filter in analytics |
| Dedup strategy | event_id first-seen | Last-seen | Preserves original event time |
| Gold aggregation | Pre-computed | Ad-hoc SQL | Sub-second dashboard response times |
| Quality framework | Custom rules | Great Expectations | No external dependency, fully transparent |

## Interview Talking Points

When asked about this project:

- **"What would you do differently at 100x scale?"** → Move to Spark, add streaming
  (Kafka + Spark Structured Streaming for real-time Gold), implement change-data-capture
  on Bronze to enable incremental Silver processing.

- **"How do you handle late-arriving events?"** → Bronze is append-only so late events
  are always ingested. Silver dedup handles re-processing. Gold would use a watermark
  window (allow events up to 2 hours late before finalizing daily aggregates).

- **"How would you add real-time capability?"** → Add Kafka topic for event ingestion,
  Spark Streaming consumer to maintain rolling Silver, use micro-batch (5-minute) Gold
  for near-real-time dashboard. Batch Gold at midnight for final numbers.

- **"How do you prevent duplicate counting in dashboards?"** → Deduplication in Silver
  ensures event_id uniqueness. Order-level metrics use order_id dedup in Gold.
  Idempotent DAG design means re-runs don't create double-counting.
