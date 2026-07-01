-- =============================================================================
-- sql/01_session_analysis.sql
-- User session reconstruction with window functions
--
-- Answers: What is average session depth, time-to-purchase, and
--          bounce rate by device and traffic source?
-- =============================================================================

WITH session_events AS (
    SELECT
        session_id,
        user_id,
        device,
        traffic_source,
        event_type,
        timestamp,
        revenue,
        -- Rank events within each session
        ROW_NUMBER() OVER (
            PARTITION BY session_id
            ORDER BY timestamp
        ) AS event_rank,
        -- Time since session start
        EXTRACT(EPOCH FROM (
            timestamp - FIRST_VALUE(timestamp) OVER (
                PARTITION BY session_id ORDER BY timestamp
            )
        )) / 60.0 AS minutes_into_session
    FROM silver_events
),

session_summary AS (
    SELECT
        session_id,
        user_id,
        MAX(device) AS device,
        MAX(traffic_source) AS traffic_source,
        COUNT(*) AS total_events,
        MAX(event_rank) AS session_depth,
        MAX(minutes_into_session) AS session_duration_min,
        SUM(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) AS page_views,
        SUM(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS cart_adds,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases,
        SUM(revenue) AS session_revenue,
        -- Time to first purchase
        MIN(CASE WHEN event_type = 'purchase'
            THEN minutes_into_session END) AS minutes_to_purchase
    FROM session_events
    GROUP BY session_id, user_id
),

session_flags AS (
    SELECT
        *,
        CASE WHEN page_views = 1 AND cart_adds = 0 THEN 1 ELSE 0 END AS is_bounce,
        CASE WHEN purchases > 0 THEN 1 ELSE 0 END AS converted
    FROM session_summary
)

SELECT
    device,
    traffic_source,
    COUNT(*) AS total_sessions,
    ROUND(AVG(session_depth), 2) AS avg_session_depth,
    ROUND(AVG(session_duration_min), 2) AS avg_duration_min,
    ROUND(SUM(is_bounce) * 100.0 / COUNT(*), 2) AS bounce_rate_pct,
    ROUND(SUM(converted) * 100.0 / COUNT(*), 2) AS conversion_rate_pct,
    ROUND(AVG(CASE WHEN converted = 1 THEN minutes_to_purchase END), 2) AS avg_minutes_to_purchase,
    ROUND(SUM(session_revenue), 2) AS total_revenue
FROM session_flags
GROUP BY device, traffic_source
ORDER BY conversion_rate_pct DESC;


-- =============================================================================
-- sql/02_funnel_analysis.sql
-- Conversion funnel with step-by-step drop-off rates
-- =============================================================================

WITH daily_events AS (
    SELECT
        event_date,
        user_id,
        session_id,
        -- Flag each funnel stage per session
        MAX(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) AS reached_view,
        MAX(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS reached_cart,
        MAX(CASE WHEN event_type = 'checkout_start' THEN 1 ELSE 0 END) AS reached_checkout,
        MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS reached_purchase
    FROM silver_events
    GROUP BY event_date, user_id, session_id
),

funnel_daily AS (
    SELECT
        event_date,
        SUM(reached_view) AS sessions_viewed,
        SUM(reached_cart) AS sessions_carted,
        SUM(reached_checkout) AS sessions_checked_out,
        SUM(reached_purchase) AS sessions_purchased
    FROM daily_events
    GROUP BY event_date
)

SELECT
    event_date,
    sessions_viewed,
    sessions_carted,
    sessions_checked_out,
    sessions_purchased,
    -- Step-to-step rates
    ROUND(sessions_carted * 100.0 / NULLIF(sessions_viewed, 0), 2) AS view_to_cart_pct,
    ROUND(sessions_checked_out * 100.0 / NULLIF(sessions_carted, 0), 2) AS cart_to_checkout_pct,
    ROUND(sessions_purchased * 100.0 / NULLIF(sessions_checked_out, 0), 2) AS checkout_to_purchase_pct,
    -- Overall conversion
    ROUND(sessions_purchased * 100.0 / NULLIF(sessions_viewed, 0), 3) AS overall_conversion_pct,
    -- 7-day rolling average conversion
    ROUND(AVG(sessions_purchased * 100.0 / NULLIF(sessions_viewed, 0))
        OVER (ORDER BY event_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 3
    ) AS rolling_7d_conversion_pct
FROM funnel_daily
ORDER BY event_date;


-- =============================================================================
-- sql/03_cohort_retention.sql
-- Weekly cohort retention analysis
-- =============================================================================

WITH user_first_purchase AS (
    SELECT
        user_id,
        DATE_TRUNC('week', MIN(timestamp)) AS cohort_week
    FROM silver_events
    WHERE event_type = 'purchase'
    GROUP BY user_id
),

user_activity_weeks AS (
    SELECT DISTINCT
        e.user_id,
        ufp.cohort_week,
        DATE_TRUNC('week', e.timestamp) AS activity_week
    FROM silver_events e
    JOIN user_first_purchase ufp ON e.user_id = ufp.user_id
    WHERE e.event_type = 'purchase'
),

cohort_sizes AS (
    SELECT
        cohort_week,
        COUNT(DISTINCT user_id) AS cohort_size
    FROM user_first_purchase
    GROUP BY cohort_week
),

retention_matrix AS (
    SELECT
        ua.cohort_week,
        cs.cohort_size,
        DATEDIFF('week', ua.cohort_week, ua.activity_week) AS week_number,
        COUNT(DISTINCT ua.user_id) AS active_users
    FROM user_activity_weeks ua
    JOIN cohort_sizes cs ON ua.cohort_week = cs.cohort_week
    GROUP BY ua.cohort_week, cs.cohort_size, week_number
)

SELECT
    cohort_week,
    cohort_size,
    week_number,
    active_users,
    ROUND(active_users * 100.0 / cohort_size, 2) AS retention_rate_pct
FROM retention_matrix
WHERE week_number <= 8  -- 8-week retention window
ORDER BY cohort_week, week_number;


-- =============================================================================
-- sql/04_product_affinity.sql
-- Market basket analysis — which products are bought together?
-- Uses self-join on order_id to find co-purchase pairs.
-- =============================================================================

WITH order_products AS (
    SELECT DISTINCT
        order_id,
        product_id,
        category
    FROM silver_events
    WHERE event_type = 'purchase'
      AND order_id != 'NONE'
      AND product_id != 'NONE'
),

product_pairs AS (
    SELECT
        a.product_id AS product_a,
        b.product_id AS product_b,
        a.category AS category_a,
        b.category AS category_b,
        COUNT(DISTINCT a.order_id) AS co_purchase_count
    FROM order_products a
    JOIN order_products b
        ON a.order_id = b.order_id
        AND a.product_id < b.product_id  -- avoid self-pairs and duplicate pairs
    GROUP BY a.product_id, b.product_id, a.category, b.category
    HAVING COUNT(DISTINCT a.order_id) >= 3  -- minimum support threshold
),

product_totals AS (
    SELECT
        product_id,
        COUNT(DISTINCT order_id) AS total_orders
    FROM order_products
    GROUP BY product_id
)

SELECT
    pp.product_a,
    pp.product_b,
    pp.category_a,
    pp.category_b,
    pp.co_purchase_count,
    -- Lift = P(A∩B) / (P(A) × P(B))
    ROUND(
        pp.co_purchase_count * 1.0
        / (pt_a.total_orders * pt_b.total_orders)
        * (SELECT COUNT(DISTINCT order_id) FROM order_products),
        4
    ) AS lift_score,
    -- Confidence: P(B|A)
    ROUND(pp.co_purchase_count * 100.0 / pt_a.total_orders, 2) AS confidence_pct
FROM product_pairs pp
JOIN product_totals pt_a ON pp.product_a = pt_a.product_id
JOIN product_totals pt_b ON pp.product_b = pt_b.product_id
ORDER BY lift_score DESC
LIMIT 50;


-- =============================================================================
-- sql/05_revenue_attribution.sql
-- Multi-touch revenue attribution by traffic source
-- First-touch, last-touch, and linear attribution models
-- =============================================================================

WITH user_sessions AS (
    SELECT
        user_id,
        session_id,
        traffic_source,
        MIN(timestamp) AS session_start,
        SUM(revenue) AS session_revenue,
        MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS has_purchase
    FROM silver_events
    GROUP BY user_id, session_id, traffic_source
),

user_journeys AS (
    SELECT
        user_id,
        session_id,
        traffic_source,
        session_start,
        session_revenue,
        has_purchase,
        -- Session rank per user (journey order)
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY session_start) AS journey_position,
        COUNT(*) OVER (PARTITION BY user_id) AS total_touchpoints
    FROM user_sessions
    WHERE session_revenue > 0 OR has_purchase = 1
),

attributions AS (
    SELECT
        traffic_source,
        -- First-touch: 100% credit to first session
        SUM(CASE WHEN journey_position = 1 THEN session_revenue ELSE 0 END)
            AS first_touch_revenue,
        -- Last-touch: 100% credit to last session
        SUM(CASE WHEN journey_position = total_touchpoints THEN session_revenue ELSE 0 END)
            AS last_touch_revenue,
        -- Linear: equal credit split across all touchpoints
        SUM(session_revenue / NULLIF(total_touchpoints, 0))
            AS linear_attribution_revenue,
        COUNT(DISTINCT user_id) AS unique_users,
        SUM(has_purchase) AS conversions
    FROM user_journeys
    GROUP BY traffic_source
),

total_revenue AS (
    SELECT SUM(linear_attribution_revenue) AS total FROM attributions
)

SELECT
    a.traffic_source,
    a.unique_users,
    a.conversions,
    ROUND(a.first_touch_revenue, 2) AS first_touch_revenue,
    ROUND(a.last_touch_revenue, 2) AS last_touch_revenue,
    ROUND(a.linear_attribution_revenue, 2) AS linear_attribution_revenue,
    ROUND(a.linear_attribution_revenue * 100.0 / tr.total, 2) AS revenue_share_pct,
    ROUND(a.linear_attribution_revenue / NULLIF(a.conversions, 0), 2) AS revenue_per_conversion
FROM attributions a
CROSS JOIN total_revenue tr
ORDER BY linear_attribution_revenue DESC;
