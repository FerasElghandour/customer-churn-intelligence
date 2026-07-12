"""
BOX 2 — UNDERSTANDING  (Data -> Understanding)

Turn a pile of invoices into one row per customer, describing who they are as of
the cutoff date. Every feature is computed ONLY from the feature period (left of
the cutoff). Nothing from the label window is allowed to touch this file.

THE FEATURE FAMILIES
--------------------
RFM — the classic retail trio, and still the backbone:
    Recency    days since their last order          (the single strongest signal)
    Frequency  how many times they've ordered
    Monetary   what they've spent, total and per order

CADENCE — the interesting part, and our answer to the fixed-window problem:
    own_mean_gap    this customer's personal rhythm between orders
    recency_ratio   recency / own_mean_gap   <-- THE KEY FEATURE

    A customer 80 days silent whose rhythm is 30 days is in trouble (ratio 2.7).
    A customer 80 days silent whose rhythm is 120 days is fine (ratio 0.7).
    Raw recency cannot tell these two apart. The ratio can.

    This is Project 1's `sensor_drift` idea, verbatim: measure the unit against
    ITS OWN baseline, never against a population constant. Health is a deviation,
    not a value. Customers turn out to work like jet engines.

TREND — is their behaviour decaying?
    A customer who spent £5,000 lifetime but £0 in the last 90 days is a
    different animal from one who spent £5,000 and £2,000 of it last quarter.
    Level tells you who they were. Trend tells you where they're going.

FRICTION — returns. A customer who keeps sending things back is telling you
    something, and it isn't good.

MISSING CADENCE, HANDLED HONESTLY
    23% of customers ordered exactly once — they have NO gap, so no personal
    rhythm exists. We fill their mean gap with the population median and set
    `has_cadence = 0`, so the model can learn "we don't actually know this
    person's rhythm" as a fact in its own right, rather than us quietly pretending
    the median is their truth.
"""

import numpy as np
import pandas as pd

# Recent-activity windows, in days before the cutoff.
# 365 is here for the VALUATION in Box 4 (what did they actually spend last year),
# not as a model feature — see FEATURE_COLS at the bottom.
TREND_WINDOWS = (30, 90, 180, 365)


def build_features(feature_period: pd.DataFrame,
                   returns: pd.DataFrame,
                   cutoff: pd.Timestamp,
                   customer_ids) -> pd.DataFrame:
    """One row per customer. State of the world as of `cutoff`."""
    p = feature_period[feature_period["Customer ID"].isin(customer_ids)].copy()

    # An "order" = one customer on one day. Invoice numbers split baskets.
    orders = (p.groupby(["Customer ID", "date"], as_index=False)
                .agg(revenue=("revenue", "sum"),
                     items=("Quantity", "sum"),
                     lines=("StockCode", "size"))
                .sort_values(["Customer ID", "date"]))

    g = orders.groupby("Customer ID")
    f = pd.DataFrame(index=g.size().index)

    # ---- RFM ----------------------------------------------------------------
    last = g["date"].max()
    first = g["date"].min()
    f["recency"] = (cutoff - last).dt.days
    f["frequency"] = g.size()
    f["monetary_total"] = g["revenue"].sum()
    f["monetary_avg"] = g["revenue"].mean()          # average order value
    f["monetary_max"] = g["revenue"].max()
    f["tenure"] = (cutoff - first).dt.days            # how long we've known them

    # ---- CADENCE: the customer's own rhythm ---------------------------------
    orders["gap"] = g["date"].diff().dt.days
    gaps = orders.dropna(subset=["gap"]).groupby("Customer ID")["gap"]
    f["own_mean_gap"] = gaps.mean()
    f["own_std_gap"] = gaps.std()
    f["has_cadence"] = f["own_mean_gap"].notna().astype(int)

    # Customers with 1 order have no gap at all. Fill with the population median
    # and FLAG it, rather than pretending we know something we don't.
    pop_median_gap = f["own_mean_gap"].median()
    f["own_mean_gap"] = f["own_mean_gap"].fillna(pop_median_gap)
    f["own_std_gap"] = f["own_std_gap"].fillna(0.0)

    # THE KEY FEATURE: how overdue are they, in units of their own normal rhythm?
    f["recency_ratio"] = f["recency"] / f["own_mean_gap"].clip(lower=1)

    # ---- TREND: is their behaviour decaying? --------------------------------
    for w in TREND_WINDOWS:
        recent = orders[orders["date"] > cutoff - pd.Timedelta(days=w)]
        rg = recent.groupby("Customer ID")
        f[f"orders_last_{w}d"] = rg.size().reindex(f.index).fillna(0)
        f[f"revenue_last_{w}d"] = rg["revenue"].sum().reindex(f.index).fillna(0.0)

    # what share of their lifetime spend happened recently? 0 = gone quiet.
    f["revenue_share_last_90d"] = (f["revenue_last_90d"] /
                                   f["monetary_total"].clip(lower=0.01))
    f["orders_share_last_90d"] = f["orders_last_90d"] / f["frequency"].clip(lower=1)

    # spend momentum: last 90d vs the 90d before that
    prev = orders[(orders["date"] <= cutoff - pd.Timedelta(days=90)) &
                  (orders["date"] > cutoff - pd.Timedelta(days=180))]
    prev_rev = prev.groupby("Customer ID")["revenue"].sum().reindex(f.index).fillna(0.0)
    f["revenue_momentum"] = ((f["revenue_last_90d"] - prev_rev) /
                             (prev_rev + f["revenue_last_90d"]).clip(lower=0.01))

    # ---- BASKET -------------------------------------------------------------
    f["avg_items_per_order"] = g["items"].mean()
    f["avg_lines_per_order"] = g["lines"].mean()
    f["distinct_products"] = (p.groupby("Customer ID")["StockCode"].nunique())

    # ---- FRICTION: returns --------------------------------------------------
    r = returns[(returns["Customer ID"].isin(customer_ids)) &
                (returns["date"] <= cutoff)]
    rg = r.groupby("Customer ID")
    f["return_orders"] = rg["Invoice"].nunique().reindex(f.index).fillna(0)
    f["return_value"] = rg["revenue"].sum().abs().reindex(f.index).fillna(0.0)
    f["return_rate"] = (f["return_value"] / f["monetary_total"].clip(lower=0.01)).clip(0, 5)

    # ---- CONTEXT ------------------------------------------------------------
    country = p.groupby("Customer ID")["Country"].agg(lambda s: s.mode().iloc[0])
    f["is_uk"] = (country == "United Kingdom").astype(int)

    f = f.reset_index().rename(columns={"index": "Customer ID"})
    return f


# NOTE: orders_last_365d / revenue_last_365d are computed above but deliberately
# left OUT of FEATURE_COLS. They exist for Box 4's customer valuation ("what did
# this person actually spend last year"), not for the model. Keeping them out
# means the Box 3 comparison is unaffected by the Box 4 valuation fix.
FEATURE_COLS = [
    # RFM
    "recency", "frequency", "monetary_total", "monetary_avg", "monetary_max", "tenure",
    # cadence
    "own_mean_gap", "own_std_gap", "has_cadence", "recency_ratio",
    # trend
    "orders_last_30d", "revenue_last_30d",
    "orders_last_90d", "revenue_last_90d",
    "orders_last_180d", "revenue_last_180d",
    "revenue_share_last_90d", "orders_share_last_90d", "revenue_momentum",
    # basket
    "avg_items_per_order", "avg_lines_per_order", "distinct_products",
    # friction
    "return_orders", "return_value", "return_rate",
    # context
    "is_uk",
]
