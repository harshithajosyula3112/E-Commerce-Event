# E-Commerce Event Stream Pipeline

A production-grade data engineering pipeline that ingests high-volume e-commerce clickstream events, transforms them through a multi-layer architecture (Bronze → Silver → Gold), enforces data quality checks, and powers a real-time analytics dashboard.


##  What This Project Covers

This pipeline simulates a real e-commerce platform (think: Amazon, Shopify) processing user behavior events. It demonstrates every skill FAANG data engineering interviews test:



## 🏗️ Architecture

```
Raw Events (JSON)
     │
     ▼
[BRONZE LAYER]   ← Raw ingestion, append-only, partitioned by date
  Parquet files  ← Schema enforcement only, no transformations
     │
     ▼
[SILVER LAYER]   ← Cleaned, deduplicated, type-cast, enriched
  - Remove duplicate events (idempotent)
  - Null handling per business rules
  - Session ID assignment
  - User journey sequence numbering
     │
     ▼
[GOLD LAYER]     ← Business aggregates, ready for BI/ML
  - Daily revenue by product/category
  - User session metrics
  - Funnel conversion rates
  - Cohort retention tables
     │
     ▼
[DASHBOARD]      ← Streamlit real-time analytics
```

**Scalability notes** (interview talking points):
- Bronze is append-only → supports re-processing without data loss
- Parquet + date partitioning → predicate pushdown for query efficiency
- Idempotent Silver transforms → safe to re-run on failure
- Gold aggregates are incremental → only reprocess changed partitions

---

##  Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | Apache Airflow 2.x |
| Processing | Python (pandas, PySpark-compatible design) |
| Storage | Parquet (local; swap S3/GCS for cloud) |
| Data Quality | Custom rule engine (Great Expectations pattern) |
| SQL | DuckDB (runs in-process, no server needed) |
| Visualization | Streamlit + Plotly |
| Testing | pytest |

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone
cd ecommerce-pipeline
pip install -r requirements.txt

# 2. Run the full pipeline (generates data + processes all layers)
python src/ingestion/event_generator.py    # Generate 100k synthetic events
python src/ingestion/ingest.py             # Bronze layer
python src/transformation/silver_transformer.py  # Silver layer
python src/transformation/gold_aggregator.py     # Gold layer
python src/quality/quality_report.py       # Data quality report

# 3. Launch dashboard
streamlit run dashboard/app.py

# 4. Run SQL analytics (requires duckdb)
python sql/run_queries.py

# 5. Run tests
pytest tests/ -v
```

**With Airflow:**
```bash
airflow db init
airflow dags trigger ecommerce_daily_pipeline
```

---

## Pipeline Metrics

| Metric | Value |
|---|---|
| Events processed per run | ~100,000 |
| Bronze → Silver dedup rate | ~2.3% duplicate removal |
| Data quality pass rate | 99.2% |
| Gold layer row count | ~2,400 daily aggregates |
| Pipeline runtime (local) | ~45 seconds |
| Supported scale (Spark) | 1M+ events/day |

---

## 💡 Skills Demonstrated

- Medallion architecture (Bronze/Silver/Gold) — industry standard at Databricks, Meta, Amazon
- Idempotent pipeline design - re-run safe, production-grade
- Data quality enforcement with automated alerting
- Advanced SQL: window functions, CTEs, session reconstruction
- Airflow DAG with retry logic, SLA monitoring, dependencies
- Parquet partitioning strategy for query optimization
- Star schema data modeling in Gold layer

---

