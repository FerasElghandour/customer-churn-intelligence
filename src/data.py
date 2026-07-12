"""
BOX 1 — DATA + TARGET  (Reality -> Data)

The hard part of this project is right here, and it is NOT the modelling.

A jet engine tells you when it died. A customer never does. Nobody clicks
"cancel" — they just quietly stop buying. So before we can predict churn, we
have to INVENT the label. There is no answer key in this dataset. We are
manufacturing the ground truth, and every downstream number inherits that choice.

=============================================================================
HOW CHURN IS FORMALLY DEFINED HERE  (and why)
=============================================================================
CONTRACTUAL vs NON-CONTRACTUAL SETTINGS
    In a CONTRACTUAL setting (SaaS, gym membership, insurance) churn is an
    OBSERVED EVENT: the customer cancels, and you have a timestamp for it.
    Online Retail II is NON-CONTRACTUAL (Fader & Hardie, Marketing Science):
    customers never announce their departure. They "silently attrite."
    Churn here cannot be observed at all — it can only be INFERRED from the
    transaction history. That inference is a modelling decision, and it is ours.

WHAT WE USE: A FIXED-WINDOW DEFINITION
    churned = no purchase within 90 days, among still-active customers.
    This is the standard textbook approach for retail churn; fixed thresholds of
    3 / 6 / 12 months are the common convention in the applied literature.

ITS KNOWN LIMITATION — STATED, NOT HIDDEN
    A fixed window mislabels slow-but-alive buyers whose natural purchase gap is
    simply longer than the window. Measured in THIS data: ~19% of all real
    inter-purchase gaps exceed 90 days. So roughly one in five genuine repeat
    buyers gets called a churner. That is the price of the choice.

WHAT WE DELIBERATELY DID NOT USE (choice, not ignorance)
    (a) DYNAMIC PER-CUSTOMER THRESHOLDS — define churn as a multiple of each
        customer's OWN average inter-purchase time. Directly fixes the slow-buyer
        problem. We measured the cost of doing this and rejected it for two
        concrete reasons, both visible in this dataset:
          - COVERAGE: 40% of our eligible customers have fewer than 3 orders, so
            they have no stable personal cadence. For them a "dynamic" threshold
            degenerates to the global median — i.e. the fixed rule again.
          - OBSERVABILITY: at 2x each customer's own mean gap, the median implied
            window is 126 days and the 90th percentile is 251 days. Our label
            window is only 90 days long, because that is all the future the data
            contains. We cannot observe 126 days of silence that isn't there.
            Fixing it means moving the cutoff back ~250 days, which costs a third
            of the feature history and shifts the study into a different season.
    (b) PROBABILISTIC "BUY-TILL-YOU-DIE" MODELS — Pareto/NBD and BG/NBD
        (Fader & Hardie). These infer P(customer is still alive) from recency and
        frequency with NO arbitrary cutoff, and are the gold standard for CLV in
        non-contractual settings. They are the principled answer. They are also a
        different project: they replace the classifier rather than feed it.

HOW WE RECOVER MOST OF THE BENEFIT ANYWAY
    We put purchase cadence into the FEATURES instead of the label (see Box 2):
    `recency / this customer's own mean gap`. A customer 80 days silent whose
    rhythm is 30 days is in trouble; one 80 days silent whose rhythm is 120 days
    is fine. The model learns that distinction itself — without us mangling the
    label or throwing away calendar. Same idea as Project 1's `sensor_drift`:
    measure the unit against ITS OWN baseline, not a population constant.
=============================================================================

THE TWO DECISIONS, AND THE EVIDENCE BEHIND THEM
-----------------------------------------------
1) CHURN WINDOW = 90 days of silence.

   Measured from the real purchase cadence in this data:
       median gap between orders ..... 32 days
       75th percentile ............... 70 days
       90th percentile .............. 148 days

   A 90-day window covers 80.6% of all real inter-purchase gaps. Which means
   ~19% of genuinely-alive-but-slow buyers get mislabelled as churned. That is
   the honest cost of this choice. A 180-day window would cut that to 7% — but
   you would learn a customer churned six months after the fact, far too late to
   send them anything. We chose actionability over label purity, deliberately.

2) ELIGIBLE POPULATION = customers who bought within 12 months before the cutoff.

   Because we found this:
       last bought <90d before cutoff ... 31% churn
       last bought 180-365d ago ......... 69% churn
       last bought OVER A YEAR AGO ...... 89% churn   <-- these people are gone

   Those 954 long-dormant customers aren't "at risk" — they already left. Leave
   them in and the model learns "recency = churn", which is true, trivial, and
   commercially useless. It would score beautifully and tell you nothing.
   A retention budget should not be aimed at corpses.

THE SETUP (a classic feature/label split across time — no leakage possible)
   |<------ FEATURE PERIOD (21 months) ------>|<-- LABEL WINDOW (90d) -->|
  2009-12-01                              2011-09-10                 2011-12-09
                                          = CUTOFF T
   Everything we KNOW is computed left of T. Whether they came back is measured
   right of T. The model can never peek at the future because the future is a
   separate slice of calendar.

TRANSFERS FROM PROJECT 1: engine -> customer, flight cycle -> day,
sensor reading -> purchase behaviour, "cycles until failure" -> "will they
come back within 90 days". Same skeleton. The label is the thing that changed:
we had to build it ourselves.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# ---- The locked decisions (see the module docstring for the evidence) --------
CHURN_WINDOW_DAYS = 90      # silence for this long = churned
ELIGIBILITY_DAYS = 365      # must have bought within this long before cutoff

# Stock codes that are not products. Postage is not a purchase.
ADMIN_CODES = {
    "POST", "DOT", "M", "C2", "BANK CHARGES", "AMAZONFEE", "ADJUST",
    "TEST001", "TEST002", "PADS", "B", "CRUK", "D", "S", "SP1002", "GIFT",
}


def load_raw() -> pd.DataFrame:
    """Read the source workbook. Cached to parquet — the .xlsx takes ~11s to parse."""
    pq = RAW / "online_retail_II.parquet"
    if pq.exists():
        return pd.read_parquet(pq)

    xl = RAW / "online_retail_II.xlsx"
    if not xl.exists():
        raise FileNotFoundError(
            f"Missing {xl}.\nDownload 'Online Retail II' from the UCI ML Repository "
            "(or Kaggle: mashlyn/online-retail-ii-uci) and place online_retail_II.xlsx there."
        )
    try:
        frames = [pd.read_excel(xl, sheet_name=s, engine="calamine")
                  for s in ["Year 2009-2010", "Year 2010-2011"]]
    except Exception:                                    # calamine not installed
        frames = [pd.read_excel(xl, sheet_name=s)
                  for s in ["Year 2009-2010", "Year 2010-2011"]]
    df = pd.concat(frames, ignore_index=True)
    for c in ["Invoice", "StockCode", "Description", "Country"]:
        df[c] = df[c].astype(str)
    df.to_parquet(pq, index=False)
    return df


def clean(df: pd.DataFrame):
    """
    Returns (purchases, returns).

    Returns are NOT thrown away — a customer who sends things back is telling you
    something about their satisfaction. They come back as a feature in Box 2.
    """
    df = df[df["Customer ID"].notna()].copy()            # unattributable -> unusable
    is_cancel = df["Invoice"].str.startswith("C")
    returns = df[is_cancel].copy()

    p = df[~is_cancel]
    p = p[(p["Quantity"] > 0) & (p["Price"] > 0)]
    p = p[~p["StockCode"].str.upper().isin(ADMIN_CODES)]

    for d in (p, returns):
        d["revenue"] = d["Quantity"] * d["Price"]
        d["date"] = d["InvoiceDate"].dt.normalize()
    return p.reset_index(drop=True), returns.reset_index(drop=True)


def build_target(purchases: pd.DataFrame,
                 churn_window: int = CHURN_WINDOW_DAYS,
                 eligibility_days: int = ELIGIBILITY_DAYS):
    """
    Split time at the cutoff, then ask one question of every eligible customer:
    did they buy again in the 90 days after the cutoff?  No = churned.
    """
    end = purchases["date"].max()
    cutoff = end - pd.Timedelta(days=churn_window)

    feature_period = purchases[purchases["date"] <= cutoff]
    label_period = purchases[purchases["date"] > cutoff]

    last_seen = feature_period.groupby("Customer ID")["date"].max()
    recency_at_cutoff = (cutoff - last_seen).dt.days

    # eligibility: alive enough to be worth a retention budget
    eligible = recency_at_cutoff[recency_at_cutoff <= eligibility_days].index

    came_back = set(label_period["Customer ID"].unique())
    target = pd.DataFrame({"Customer ID": eligible})
    target["churned"] = (~target["Customer ID"].isin(came_back)).astype(int)

    meta = dict(
        data_start=purchases["date"].min(),
        data_end=end,
        cutoff=cutoff,
        churn_window=churn_window,
        eligibility_days=eligibility_days,
        n_eligible=len(target),
        n_churned=int(target["churned"].sum()),
        churn_rate=float(target["churned"].mean()),
    )
    return target, feature_period, label_period, meta


if __name__ == "__main__":
    raw = load_raw()
    purchases, returns = clean(raw)
    target, feat, lab, meta = build_target(purchases)

    print(f"raw rows              {len(raw):,}")
    print(f"clean purchase rows   {len(purchases):,}  ({len(purchases)/len(raw)*100:.0f}% kept)")
    print(f"returns held aside    {len(returns):,}")
    print(f"customers in data     {purchases['Customer ID'].nunique():,}")
    print(f"total revenue         £{purchases['revenue'].sum():,.0f}")
    print()
    print(f"data spans            {meta['data_start'].date()} -> {meta['data_end'].date()}")
    print(f"cutoff T              {meta['cutoff'].date()}")
    print(f"feature period        {(meta['cutoff']-meta['data_start']).days} days of history")
    print(f"label window          {meta['churn_window']} days after T")
    print()
    print(f"ELIGIBLE CUSTOMERS    {meta['n_eligible']:,}  (bought within {meta['eligibility_days']}d before T)")
    print(f"CHURNED               {meta['n_churned']:,}")
    print(f"CHURN RATE            {meta['churn_rate']*100:.1f}%")
