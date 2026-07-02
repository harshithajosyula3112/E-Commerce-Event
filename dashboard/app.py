"""
dashboard/app.py
-----------------
Streamlit analytics dashboard for the e-commerce pipeline.
Reads from Gold layer Parquet files.

Run: streamlit run dashboard/app.py
"""

import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.kpi-card {
    background: #f0f4ff;
    border: 1px solid #c7d2fe;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 8px;
}
.kpi-label { font-size: 12px; color: #6366f1; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { font-size: 26px; font-weight: 700; color: #1e1b4b; margin: 4px 0; }
.kpi-sub { font-size: 12px; color: #64748b; }
.positive { color: #059669; }
.negative { color: #dc2626; }
section[data-testid="stSidebar"] { background-color: #1e1b4b; }
section[data-testid="stSidebar"] * { color: #e0e7ff !important; }
</style>
""", unsafe_allow_html=True)

GOLD_DIR = Path("data/gold")

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_gold_tables():
    """Load all Gold tables, generating them if needed."""
    tables = {}
    table_names = [
        "daily_revenue", "daily_revenue_total", "funnel_metrics",
        "user_segments", "product_performance", "cohort_summary",
    ]

    for name in table_names:
        path = GOLD_DIR / f"{name}.parquet"
        if path.exists():
            tables[name] = pd.read_parquet(path)
        else:
            tables[name] = None

    if tables.get("daily_revenue") is None:
        st.warning("Gold data not found. Running pipeline now...")
        with st.spinner("Building pipeline data (~60 seconds)..."):
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            try:
                from ingestion.event_generator import generate_events
                from ingestion.ingest import BronzeIngester
                from transformation.silver_transformer import SilverTransformer
                from transformation.gold_aggregator import GoldAggregator

                generate_events(n_days=30, target_events=50_000)
                BronzeIngester().run()
                SilverTransformer().run()
                agg = GoldAggregator()
                agg.run()

                for name in table_names:
                    path = GOLD_DIR / f"{name}.parquet"
                    if path.exists():
                        tables[name] = pd.read_parquet(path)
            except Exception as e:
                st.error(f"Pipeline error: {e}\nRun the pipeline manually first.")
                st.stop()

    return tables


data = load_gold_tables()

daily = data.get("daily_revenue", pd.DataFrame())
daily_total = data.get("daily_revenue_total", pd.DataFrame())
funnel = data.get("funnel_metrics", pd.DataFrame())
users = data.get("user_segments", pd.DataFrame())
products = data.get("product_performance", pd.DataFrame())
cohort = data.get("cohort_summary", pd.DataFrame())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 E-Commerce Analytics")
    st.markdown("---")
    page = st.radio("View", [
        "📊 Executive Overview",
        "🔽 Conversion Funnel",
        "🛍️ Product Analytics",
        "👥 User Segments",
        "📅 Cohort Retention",
        "⚙️ Pipeline Status",
    ])
    st.markdown("---")

    if not daily.empty and "event_date" in daily.columns:
        dates = sorted(daily["event_date"].unique())
        if len(dates) >= 2:
            date_range = st.select_slider(
                "Date Range",
                options=dates,
                value=(dates[0], dates[-1]),
            )
        else:
            date_range = (dates[0], dates[-1]) if dates else (None, None)
    else:
        date_range = (None, None)


# ── Page: Executive Overview ──────────────────────────────────────────────────
if "Executive" in page:
    st.markdown("## Executive Overview")

    if not daily_total.empty:
        total_rev = daily_total["gross_revenue"].sum()
        total_orders = daily["orders"].sum() if not daily.empty else 0
        unique_buyers = daily["unique_buyers"].sum() if not daily.empty else 0
        avg_aov = daily["avg_order_value"].mean() if not daily.empty else 0

        k1, k2, k3, k4 = st.columns(4)
        for col, label, val, fmt in [
            (k1, "TOTAL REVENUE", total_rev, "${:,.0f}"),
            (k2, "TOTAL ORDERS", total_orders, "{:,}"),
            (k3, "UNIQUE BUYERS", unique_buyers, "{:,}"),
            (k4, "AVG ORDER VALUE", avg_aov, "${:.2f}"),
        ]:
            col.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{fmt.format(val)}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Revenue trend
        if not daily_total.empty and "gross_revenue" in daily_total.columns:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=daily_total["event_date"], y=daily_total["gross_revenue"],
                name="Daily Revenue", marker_color="#6366f1", opacity=0.7,
            ), secondary_y=False)
            if "revenue_7d_avg" in daily_total.columns:
                fig.add_trace(go.Scatter(
                    x=daily_total["event_date"], y=daily_total["revenue_7d_avg"],
                    name="7-Day Avg", line=dict(color="#f59e0b", width=2.5),
                ), secondary_y=False)
            fig.update_layout(
                title="Daily Revenue Trend",
                plot_bgcolor="white", paper_bgcolor="white",
                height=320, margin=dict(l=40, r=20, t=40, b=40),
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            if not daily.empty and "category" in daily.columns:
                cat_rev = daily.groupby("category")["gross_revenue"].sum().reset_index()
                fig = px.pie(cat_rev, values="gross_revenue", names="category",
                             title="Revenue by Category",
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             height=320)
                fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if not daily.empty and "device" in daily.columns:
                device_rev = daily.groupby("device")["gross_revenue"].sum().reset_index()
                fig = px.bar(device_rev, x="device", y="gross_revenue",
                             title="Revenue by Device",
                             color="device",
                             color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b"],
                             height=320)
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                  showlegend=False, margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)


# ── Page: Conversion Funnel ───────────────────────────────────────────────────
elif "Funnel" in page:
    st.markdown("##  Conversion Funnel Analysis")

    if not funnel.empty:
        avg_funnel = funnel.agg({
            "page_views": "mean", "add_to_carts": "mean",
            "checkouts": "mean", "purchases": "mean",
        })

        stages = ["Page Views", "Add to Cart", "Checkout", "Purchase"]
        values = [avg_funnel["page_views"], avg_funnel["add_to_carts"],
                  avg_funnel["checkouts"], avg_funnel["purchases"]]

        fig_funnel = go.Figure(go.Funnel(
            y=stages, x=values,
            textinfo="value+percent initial",
            marker_color=["#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe"],
        ))
        fig_funnel.update_layout(
            title="Average Daily Conversion Funnel",
            height=380, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

        # Funnel rates over time
        funnel_daily = funnel.groupby("event_date").agg({
            "overall_conversion": "mean",
            "cart_rate": "mean",
            "checkout_rate": "mean",
        }).reset_index()

        fig_rates = px.line(funnel_daily, x="event_date",
                            y=["cart_rate", "checkout_rate", "overall_conversion"],
                            title="Daily Conversion Rates Over Time",
                            labels={"value": "Rate (%)", "variable": "Stage"},
                            height=320)
        fig_rates.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_rates, use_container_width=True)


# ── Page: Product Analytics ───────────────────────────────────────────────────
elif "Product" in page:
    st.markdown("## Product Performance")

    if not products.empty:
        col1, col2 = st.columns(2)

        with col1:
            top_rev = products.nlargest(10, "total_revenue")
            fig = px.bar(top_rev, x="total_revenue", y="product_id",
                         orientation="h", color="category",
                         title="Top 10 Products by Revenue",
                         height=380)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig_scatter = px.scatter(
                products.sample(min(len(products), 100)),
                x="conversion_rate", y="total_revenue",
                size="page_views", color="category",
                hover_data=["product_id", "purchases"],
                title="Conversion Rate vs Revenue",
                height=380,
            )
            fig_scatter.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                      margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Category Summary")
        cat_summary = products.groupby("category").agg(
            total_revenue=("total_revenue", "sum"),
            total_products=("product_id", "count"),
            avg_conversion=("conversion_rate", "mean"),
            total_orders=("purchases", "sum"),
        ).reset_index().sort_values("total_revenue", ascending=False)
        cat_summary["total_revenue"] = cat_summary["total_revenue"].round(2)
        cat_summary["avg_conversion"] = cat_summary["avg_conversion"].round(3)
        st.dataframe(cat_summary, use_container_width=True, hide_index=True)


# ── Page: User Segments ───────────────────────────────────────────────────────
elif "User" in page:
    st.markdown("##  User Segmentation (RFM)")

    if not users.empty and "rfm_segment" in users.columns:
        seg_stats = users.groupby("rfm_segment").agg(
            count=("user_id", "count"),
            avg_revenue=("total_revenue", "mean"),
            total_revenue=("total_revenue", "sum"),
            avg_orders=("total_orders", "mean"),
        ).reset_index()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(seg_stats, values="count", names="rfm_segment",
                         title="User Distribution by Segment",
                         color_discrete_sequence=["#6366f1","#10b981","#f59e0b","#ef4444"],
                         height=320)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(seg_stats, x="rfm_segment", y="avg_revenue",
                         color="rfm_segment", title="Average Revenue per User by Segment",
                         color_discrete_sequence=["#6366f1","#10b981","#f59e0b","#ef4444"],
                         height=320)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            seg_stats.rename(columns={
                "count":"Users","avg_revenue":"Avg Revenue ($)",
                "total_revenue":"Total Revenue ($)","avg_orders":"Avg Orders",
            }).round(2),
            use_container_width=True, hide_index=True,
        )


# ── Page: Cohort Retention ────────────────────────────────────────────────────
elif "Cohort" in page:
    st.markdown("## Cohort Retention Analysis")

    if not cohort.empty:
        pivot = cohort.pivot_table(
            index="acquisition_week", columns="week_number",
            values="retention_rate", aggfunc="mean"
        ).round(1)

        if not pivot.empty:
            fig = px.imshow(
                pivot, text_auto=True,
                color_continuous_scale="Blues",
                title="Weekly Cohort Retention Heatmap (%)",
                labels=dict(x="Weeks Since Acquisition", y="Acquisition Week", color="Retention %"),
                height=400,
            )
            fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

            week0 = cohort[cohort["week_number"] == 0] if "week_number" in cohort.columns else pd.DataFrame()
            if not week0.empty:
                avg_retention = cohort.groupby("week_number")["retention_rate"].mean().reset_index()
                fig2 = px.line(avg_retention, x="week_number", y="retention_rate",
                               title="Average Retention Curve",
                               labels={"week_number": "Weeks", "retention_rate": "Retention (%)"},
                               markers=True, height=300)
                fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig2, use_container_width=True)


# ── Page: Pipeline Status ─────────────────────────────────────────────────────
elif "Pipeline" in page:
    st.markdown("##  Pipeline Status")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gold Tables")
        for name in ["daily_revenue", "daily_revenue_total", "funnel_metrics",
                      "user_segments", "product_performance", "cohort_summary"]:
            path = GOLD_DIR / f"{name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                mod_time = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                st.success(f" {name} — {len(df):,} rows — last updated {mod_time}")
            else:
                st.error(f" {name} — NOT FOUND")

    with col2:
        st.subheader("Layer Summary")
        for layer, dir_path in [("Bronze", Path("data/bronze")),
                                  ("Silver", Path("data/silver")),
                                  ("Gold", Path("data/gold"))]:
            partitions = list(dir_path.glob("**/*.parquet")) if dir_path.exists() else []
            if partitions:
                total_mb = sum(p.stat().st_size for p in partitions) / 1e6
                st.info(f"**{layer}**: {len(partitions)} files, {total_mb:.1f} MB")
            else:
                st.warning(f"**{layer}**: No data")

    st.subheader("Run Full Pipeline")
    if st.button("Run Pipeline Now", type="primary"):
        with st.spinner("Running pipeline..."):
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            try:
                from ingestion.event_generator import generate_events
                from ingestion.ingest import BronzeIngester
                from transformation.silver_transformer import SilverTransformer
                from transformation.gold_aggregator import GoldAggregator

                generate_events(n_days=30, target_events=50_000)
                BronzeIngester().run()
                SilverTransformer().run()
                GoldAggregator().run()
                st.success("Pipeline completed successfully!")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

st.markdown("---")
st.caption("E-Commerce Event Stream Pipeline | Medallion Architecture (Bronze→Silver→Gold) | Apache Airflow + DuckDB")
