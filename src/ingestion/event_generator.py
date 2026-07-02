"""
src/ingestion/event_generator.py
---------------------------------
Generates realistic synthetic e-commerce clickstream events.
Simulates a 30-day event history for ~5,000 users across 500 products.

Event types (mirrors real platform event taxonomy):
  - page_view     : User views a product page
  - add_to_cart   : User adds item to cart
  - remove_from_cart
  - checkout_start
  - purchase      : Completed transaction (with revenue)
  - search        : User searches for a product
  - login / logout

Realistic behaviors modeled:
  - User funnel drop-off (most views → fewer purchases)
  - Session clustering (events happen in bursts)
  - Product popularity follows power law (Pareto)
  - Revenue follows log-normal distribution
  - Peak traffic on weekday evenings + weekends

Usage:
    python src/ingestion/event_generator.py
"""

import json
import uuid
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Master data ───────────────────────────────────────────────────────────────

CATEGORIES = {
    "Electronics":      {"price_range": (29, 1499), "weight": 0.22},
    "Clothing":         {"price_range": (12, 299),  "weight": 0.25},
    "Home & Kitchen":   {"price_range": (8, 499),   "weight": 0.18},
    "Books":            {"price_range": (5, 59),    "weight": 0.12},
    "Sports":           {"price_range": (10, 399),  "weight": 0.10},
    "Beauty":           {"price_range": (8, 199),   "weight": 0.08},
    "Toys":             {"price_range": (5, 149),   "weight": 0.05},
}

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_WEIGHTS = [0.55, 0.37, 0.08]

TRAFFIC_SOURCES = ["organic", "paid_search", "email", "social", "direct", "affiliate"]
TRAFFIC_WEIGHTS = [0.30, 0.22, 0.18, 0.15, 0.10, 0.05]

US_STATES = [
    "CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
]
STATE_WEIGHTS = [0.12, 0.09, 0.08, 0.07, 0.05, 0.04, 0.04, 0.04, 0.04, 0.03,
                 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]


def generate_product_catalog(n: int = 500) -> pd.DataFrame:
    """Generate a product catalog with realistic attributes."""
    rng = np.random.default_rng(42)
    categories = list(CATEGORIES.keys())
    cat_weights = [CATEGORIES[c]["weight"] for c in categories]

    chosen_cats = rng.choice(categories, size=n, p=cat_weights)

    records = []
    for i, cat in enumerate(chosen_cats):
        lo, hi = CATEGORIES[cat]["price_range"]
        price = round(float(rng.lognormal(np.log((lo + hi) / 2), 0.4)), 2)
        price = max(lo, min(hi, price))
        rating = round(float(rng.normal(4.1, 0.6)), 1)
        rating = max(1.0, min(5.0, rating))

        records.append({
            "product_id": f"P{i+1:05d}",
            "product_name": f"{cat} Product {i+1}",
            "category": cat,
            "price": price,
            "rating": rating,
            "inventory": int(rng.integers(0, 500)),
        })

    return pd.DataFrame(records)


def generate_users(n: int = 5000) -> pd.DataFrame:
    """Generate a user base with realistic segments."""
    rng = np.random.default_rng(7)
    states = np.array(US_STATES)
    state_w = np.array(STATE_WEIGHTS) / sum(STATE_WEIGHTS)
    chosen_states = rng.choice(states, size=n, p=state_w)

    segments = rng.choice(
        ["high_value", "regular", "occasional", "new"],
        size=n, p=[0.10, 0.40, 0.35, 0.15]
    )

    signup_days = rng.integers(1, 1095, size=n)  # up to 3 years old
    records = [{
        "user_id": f"U{i+1:06d}",
        "state": chosen_states[i],
        "segment": segments[i],
        "signup_days_ago": int(signup_days[i]),
    } for i in range(n)]

    return pd.DataFrame(records)


def simulate_session(
    user_id: str,
    products: pd.DataFrame,
    session_start: datetime,
    segment: str,
    device: str,
    source: str,
) -> list[dict]:
    """
    Simulate a single browsing session for one user.
    Session depth and purchase probability vary by user segment.
    """
    # Segment-based behavior parameters
    params = {
        "high_value":  {"n_views": (3, 12), "purchase_prob": 0.45, "add_prob": 0.40},
        "regular":     {"n_views": (2, 8),  "purchase_prob": 0.18, "add_prob": 0.28},
        "occasional":  {"n_views": (1, 5),  "purchase_prob": 0.08, "add_prob": 0.15},
        "new":         {"n_views": (1, 4),  "purchase_prob": 0.05, "add_prob": 0.12},
    }
    p = params.get(segment, params["regular"])

    session_id = str(uuid.uuid4())[:16]
    events = []
    current_time = session_start
    cart = []

    # Product popularity: power-law — top 20% of products get 80% of views
    n_products = len(products)
    popularity_weights = np.array([1 / (i + 1) ** 0.7 for i in range(n_products)])
    popularity_weights /= popularity_weights.sum()

    # Optional: start with search
    if random.random() < 0.35:
        events.append(_make_event("search", user_id, session_id, None, None,
                                   current_time, device, source, {}))
        current_time += timedelta(seconds=random.randint(5, 30))

    # Page views
    n_views = random.randint(*p["n_views"])
    viewed = np.random.choice(products.index, size=min(n_views, len(products)),
                              replace=False, p=popularity_weights).tolist()

    for idx in viewed:
        prod = products.iloc[idx]

        events.append(_make_event(
            "page_view", user_id, session_id,
            prod["product_id"], prod["category"],
            current_time, device, source,
            {"price": prod["price"], "rating": prod["rating"]},
        ))
        current_time += timedelta(seconds=random.randint(20, 180))

        # Add to cart?
        if random.random() < p["add_prob"]:
            qty = random.choices([1, 2, 3], weights=[0.75, 0.20, 0.05])[0]
            cart.append({"product_id": prod["product_id"],
                         "category": prod["category"],
                         "price": prod["price"],
                         "quantity": qty})
            events.append(_make_event(
                "add_to_cart", user_id, session_id,
                prod["product_id"], prod["category"],
                current_time, device, source,
                {"price": prod["price"], "quantity": qty},
            ))
            current_time += timedelta(seconds=random.randint(5, 30))

    # Checkout + purchase?
    if cart and random.random() < p["purchase_prob"]:
        events.append(_make_event("checkout_start", user_id, session_id,
                                   None, None, current_time, device, source, {}))
        current_time += timedelta(seconds=random.randint(30, 120))

        order_id = str(uuid.uuid4())[:12]
        for item in cart:
            revenue = round(item["price"] * item["quantity"], 2)
            events.append(_make_event(
                "purchase", user_id, session_id,
                item["product_id"], item["category"],
                current_time, device, source,
                {
                    "price": item["price"],
                    "quantity": item["quantity"],
                    "revenue": revenue,
                    "order_id": order_id,
                },
            ))
        current_time += timedelta(seconds=random.randint(5, 20))

    return events


def _make_event(
    event_type: str, user_id: str, session_id: str,
    product_id: str | None, category: str | None,
    timestamp: datetime, device: str, source: str,
    extra: dict,
) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "user_id": user_id,
        "session_id": session_id,
        "product_id": product_id,
        "category": category,
        "timestamp": timestamp.isoformat(),
        "device": device,
        "traffic_source": source,
        **extra,
    }


def generate_events(
    n_days: int = 30,
    target_events: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate the full event dataset.
    Returns a DataFrame of clickstream events.
    """
    random.seed(seed)
    np.random.seed(seed)

    log.info("Generating product catalog and users...")
    products = generate_product_catalog(500)
    users = generate_users(5000)

    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=n_days)

    all_events = []
    events_per_day = target_events // n_days

    log.info(f"Simulating {n_days} days of events (~{events_per_day:,}/day)...")

    for day_offset in range(n_days):
        current_day = start_date + timedelta(days=day_offset)
        day_events = []
        events_today = 0

        # Traffic spike on weekends (1.3x) and Mon/Tue dips (0.85x)
        day_of_week = current_day.weekday()
        day_multiplier = 1.3 if day_of_week >= 5 else (0.85 if day_of_week < 2 else 1.0)
        target_today = int(events_per_day * day_multiplier)

        while events_today < target_today:
            user = users.sample(1).iloc[0]

            # Session time: peak 7–11 PM, low 3–7 AM
            hour = random.choices(range(24),
                                  weights=[1,1,1,1,1,2,3,4,5,6,7,8,
                                           8,8,7,6,7,9,11,13,12,10,7,4])[0]
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            session_start = current_day.replace(hour=hour, minute=minute, second=second)

            device = random.choices(DEVICES, weights=DEVICE_WEIGHTS)[0]
            source = random.choices(TRAFFIC_SOURCES, weights=TRAFFIC_WEIGHTS)[0]

            session_events = simulate_session(
                user["user_id"], products, session_start,
                user["segment"], device, source,
            )

            day_events.extend(session_events)
            events_today += len(session_events)

        all_events.extend(day_events)

        if (day_offset + 1) % 5 == 0:
            log.info(f"  Day {day_offset+1}/{n_days} — total events: {len(all_events):,}")

    df = pd.DataFrame(all_events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    log.info(f"Generated {len(df):,} events | "
             f"{df['user_id'].nunique():,} users | "
             f"{df[df['event_type']=='purchase']['revenue'].sum():,.0f} total revenue")

    return df, products, users


if __name__ == "__main__":
    df_events, df_products, df_users = generate_events(n_days=30, target_events=100_000)

    out_events = OUTPUT_DIR / "events_raw.parquet"
    out_products = OUTPUT_DIR / "products.parquet"
    out_users = OUTPUT_DIR / "users.parquet"

    df_events.to_parquet(out_events, index=False)
    df_products.to_parquet(out_products, index=False)
    df_users.to_parquet(out_users, index=False)

    log.info(f"Saved → {out_events}")
    log.info(f"Saved → {out_products}")
    log.info(f"Saved → {out_users}")
    log.info(f"\nEvent breakdown:\n{df_events['event_type'].value_counts().to_string()}")
