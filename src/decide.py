"""
BOX 4 — THE COST-AWARE DECISION LAYER  (Prediction -> Decision)

    "78% likely to churn"  ->  "£2,800 at risk. Send the offer. Expected gain £64."

This is the box that earns the project. Everything before it produces numbers.
This produces DECISIONS, and decisions are what a business actually buys.

WE DO NOT ASK "WHO WILL CHURN?"
We ask: "For whom does the expected value of intervening exceed the cost?"
Those are different questions and they produce different lists.

THE ECONOMICS
-------------
For one customer:
    p    = P(churn)                      from Box 3 (calibrated — that's why it matters)
    R    = revenue they actually spent in the last 365 days (their forward value)
    AOV  = their average order value
    GM   = gross margin
    D    = discount %
    s    = save rate — share of would-be churners the offer actually rescues
    c    = contact cost

    EV(send) =  p·s·(GM·R − Dcost)      they were leaving; the offer rescued them.
                                        We keep a year of margin, less the discount.
             − (1−p)·Dcost              THE FALSE ALARM. They were staying anyway,
                                        and they redeem a discount on a purchase
                                        they'd have made at full price. Pure burn.
             − c                        paid either way.

    Send  <=>  EV > 0.

Note which mistake is expensive. A churner we FAIL to catch costs us only `c` —
they never come back, so they never redeem. **The costly error is the STAYER, not
the miss.** That is the exact inversion of the jet-engine project, where a miss
was catastrophic and a false alarm was cheap.

*** THE MOST IMPORTANT LINE IN THIS FILE IS OFFER_SCOPE ***
-----------------------------------------------------------
What does the discount cost when redeemed?

    "next_order" : "15% off your next order."  Cost = 15% of ONE order (AOV).
                   This is what a normal win-back email actually is.
    "annual"     : a standing 15% off everything for a year. Cost = 15% × R.

For a repeat customer R >> AOV, so these differ by roughly 10x. And that single
marketing choice INVERTS THE ENTIRE CONCLUSION:

    next_order -> false alarms are CHEAP -> act on ~50% suspicion -> profitable
    annual     -> false alarms are BRUTAL -> need ~71% -> barely worth doing

We ship "next_order" because that's the realistic offer. The sweep in main()
prints both, because the finding is the point: **the offer design, not the model,
decides whether retention makes money.** No amount of AUC rescues a bad offer.

HOW VALUE-WEIGHTING WORKS
-------------------------
The breakeven threshold is computed PER CUSTOMER:

    p* = (Dcost + c) / ( s·(GM·R − Dcost) + Dcost )

A whale has a huge GM·R on top, so their threshold FALLS — you act on thinner
suspicion for a customer worth more. It isn't a rule bolted on the side; it falls
straight out of the same equation. Low-value customers must clear a higher bar
because there's less to win and the contact cost bites harder.

ASSUMPTIONS — ALL PLACEHOLDERS, ALL ADJUSTABLE
----------------------------------------------
These are the numbers a real brand supplies from its own history. Every result in
this project is downstream of them. That's why they sit at the top in capitals.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

# =============================================================================
# THE NUMBERS A REAL BRAND WOULD REPLACE WITH THEIR OWN
# =============================================================================
GROSS_MARGIN = 0.40         # 40% — typical for gift / homeware wholesale
OFFER_DISCOUNT = 0.15       # the win-back offer: 15% off
SAVE_RATE = 0.25            # 25% of would-be churners are actually rescued.
                            # Real win-back campaigns land ~10-30%. The single
                            # most uncertain input here — so we sweep it.
CONTACT_COST = 1.50         # £1.50 per contact (email + ops + creative, amortised)
VALUE_HORIZON_DAYS = 365    # we value one forward year of the relationship

OFFER_SCOPE = "next_order"  # "next_order" | "annual"  <- see the module docstring.
                            # This decides whether the whole programme is profitable.

VALUE_CAP_PERCENTILE = 99   # blunt the extreme tail — see expected_forward_value
# =============================================================================


def expected_forward_value(df: pd.DataFrame) -> pd.Series:
    """
    What is this customer worth over the next year IF they stay?

    WE USE: revenue they ACTUALLY SPENT in the last 365 days. Observed. No maths.

    WE REJECTED: annualising their lifetime run-rate (monetary_total/tenure*365).
    It looks more sophisticated and it is much worse. It produced this:

        Customer 15098:  £39,916 spent | tenure 92 days | ONE single order
                         -> extrapolated to a £158,364/year customer
                         -> and therefore ranked #1 on the "save this customer" list

    One bulk order, multiplied by four, became a phantom whale at the top of the
    action list. A marketer would have phoned them, and the number would have been
    fiction. **Extrapolation invents revenue. Observation does not.**

    THEN WE CAP at the 99th percentile — not to flatter the numbers, but because
    the P&L was riding on three customers. A policy whose profitability depends on
    0.3% of the book is not a policy, it's a bet. The concentration check in main()
    reports whether we succeeded.
    """
    v = df["revenue_last_365d"].astype(float)
    return v.clip(upper=np.percentile(v, VALUE_CAP_PERCENTILE))


def _discount_cost(R, AOV, scope=OFFER_SCOPE, d=OFFER_DISCOUNT):
    """Cost of the discount when redeemed. See OFFER_SCOPE in the docstring."""
    return d * (AOV if scope == "next_order" else R)


def expected_value_of_offer(p, R, AOV, gm=GROSS_MARGIN, d=OFFER_DISCOUNT,
                            s=SAVE_RATE, c=CONTACT_COST, scope=OFFER_SCOPE):
    """EV of sending the retention offer. Positive = send it."""
    dc = _discount_cost(R, AOV, scope, d)
    return p * s * (gm * R - dc) - (1 - p) * dc - c


def breakeven_probability(R, AOV, gm=GROSS_MARGIN, d=OFFER_DISCOUNT,
                          s=SAVE_RATE, c=CONTACT_COST, scope=OFFER_SCOPE):
    """Per-customer probability at which the offer breaks even."""
    dc = _discount_cost(R, AOV, scope, d)
    denom = s * (gm * R - dc) + dc
    return np.clip(np.where(denom > 0, (dc + c) / np.maximum(denom, 1e-9), 1.0), 0, 1)


def decide(df: pd.DataFrame, scope=OFFER_SCOPE) -> pd.DataFrame:
    out = df.copy()
    out["annual_value"] = expected_forward_value(out)
    out["aov"] = out["monetary_avg"].astype(float)
    out["value_at_risk"] = out["churn_prob"] * out["annual_value"]
    out["margin_at_risk"] = out["value_at_risk"] * GROSS_MARGIN
    out["breakeven_prob"] = breakeven_probability(out["annual_value"], out["aov"], scope=scope)
    out["ev_of_offer"] = expected_value_of_offer(out["churn_prob"], out["annual_value"],
                                                 out["aov"], scope=scope)
    out["offer_cost"] = _discount_cost(out["annual_value"], out["aov"], scope) + CONTACT_COST

    # THE DECISION. Not "will they churn" — "is it worth acting".
    out["act"] = out["ev_of_offer"] > 0

    out["segment"] = pd.cut(out["annual_value"], [-1, 500, 2000, 10000, np.inf],
                            labels=["Low (<£500)", "Mid (£500-2k)",
                                    "High (£2k-10k)", "VIP (>£10k)"])
    out["action"] = np.select(
        [out.act & (out.annual_value >= 10000),
         out.act & (out.annual_value >= 2000),
         out.act],
        ["Account manager calls personally — don't discount blindly",
         f"Send {int(OFFER_DISCOUNT*100)}% win-back offer + personal note",
         f"Send {int(OFFER_DISCOUNT*100)}% win-back offer (automated email)"],
        default="No action — not economic to intervene",
    )
    return out.sort_values("value_at_risk", ascending=False).reset_index(drop=True)


def evaluate_policy(d, threshold=None, use_ev=True, scope=OFFER_SCOPE):
    """
    Score a policy against the TRUTH — held-out customers, real outcomes.
    This is a realised P&L, not a projection.
    """
    flagged = d[d["ev_of_offer"] > 0] if use_ev else d[d["churn_prob"] >= threshold]
    if len(flagged) == 0:
        return dict(policy="none", contacted=0, pct_of_base=0.0, precision=0.0,
                    churners_caught=0, recall=0.0, margin_saved=0,
                    discount_wasted=0, contact_cost=0, NET=0)

    churned = flagged["churned"] == 1
    dc = _discount_cost(flagged["annual_value"], flagged["aov"], scope)

    saved = (SAVE_RATE * (GROSS_MARGIN * flagged.loc[churned, "annual_value"]
                          - dc[churned])).sum()
    wasted = dc[~churned].sum()          # stayers redeeming a discount they didn't need
    contact = len(flagged) * CONTACT_COST

    return dict(
        policy=("EV-optimal (value-weighted)" if use_ev else f"fixed p>={threshold:.0%}"),
        contacted=len(flagged),
        pct_of_base=round(100 * len(flagged) / len(d), 1),
        precision=round(100 * churned.mean(), 1),
        churners_caught=int(churned.sum()),
        recall=round(100 * churned.sum() / max((d.churned == 1).sum(), 1), 1),
        margin_saved=round(saved), discount_wasted=round(wasted),
        contact_cost=round(contact), NET=round(saved - wasted - contact),
    )


def main():
    scores = pd.read_parquet(ARTIFACTS / "customer_scores.parquet")
    d = decide(scores)
    d.to_parquet(ARTIFACTS / "customer_decisions.parquet", index=False)

    total_value = d["annual_value"].sum()
    at_risk = d["value_at_risk"].sum()

    print("=" * 90)
    print("THE BOOK")
    print("=" * 90)
    print(f"  customers                        {len(d):,}")
    print(f"  forward annual revenue           £{total_value:,.0f}")
    print(f"  REVENUE AT RISK  (Σ p × value)   £{at_risk:,.0f}   ({at_risk/total_value*100:.0f}% of the book)")
    print(f"  GROSS MARGIN AT RISK             £{at_risk*GROSS_MARGIN:,.0f}")
    print(f"\n  ASSUMPTIONS  margin {GROSS_MARGIN:.0%} | offer {OFFER_DISCOUNT:.0%} "
          f"({OFFER_SCOPE}) | save rate {SAVE_RATE:.0%} | contact £{CONTACT_COST:.2f}")

    # ---------- THE FINDING: offer design decides everything -----------------
    print("\n" + "=" * 90)
    print("OFFER DESIGN — the marketing choice that decides if ANY of this is profitable")
    print("=" * 90)
    test_ids = scores["in_test_set"]
    rows = []
    for scope in ["next_order", "annual"]:
        dd = decide(scores, scope=scope)
        tt = dd[dd["in_test_set"]]
        r = evaluate_policy(tt, use_ev=True, scope=scope)
        med_be = dd["breakeven_prob"].median()
        rows.append(dict(offer=f"15% off, {scope}", median_breakeven=f"{med_be:.0%}",
                         contacted=r["contacted"], NET=r["NET"]))
    od = pd.DataFrame(rows)
    print(od.to_string(index=False))
    print("\n  Same model. Same customers. Same 15%. The ONLY difference is whether the")
    print("  discount applies to one order or to a year of orders — and it flips the")
    print("  answer from 'run this campaign' to 'don't bother'.")

    # ---------- threshold sweep ---------------------------------------------
    test = d[d["in_test_set"]].copy()
    sweep_rows = [evaluate_policy(test, use_ev=True)]
    for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        sweep_rows.append(evaluate_policy(test, threshold=t, use_ev=False))
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(REPORTS / "threshold_sweep.csv", index=False)

    print("\n" + "=" * 90)
    print(f"THRESHOLD SWEEP — {len(test):,} HELD-OUT customers, real outcomes, real P&L")
    print("=" * 90)
    print(sweep.rename(columns={
        "contacted": "contact", "pct_of_base": "%base", "precision": "prec%",
        "churners_caught": "caught", "recall": "recall%", "margin_saved": "saved£",
        "discount_wasted": "wasted£", "contact_cost": "cost£", "NET": "NET £"
    }).to_string(index=False))
    best = sweep.loc[sweep["NET"].idxmax()]
    print(f"\n  BEST NET: {best['policy']} -> £{best['NET']:,.0f}")
    print(f"  Chasing recall (p>=30%) contacts {sweep.iloc[1]['contacted']} people and nets "
          f"£{sweep.iloc[1]['NET']:,.0f} — MORE churners caught, LESS money.")

    # ---------- segments -----------------------------------------------------
    print("\n" + "=" * 90)
    print("BY CUSTOMER VALUE — and how the bar moves with value")
    print("=" * 90)
    seg = (d.groupby("segment", observed=True)
             .agg(customers=("Customer ID", "size"), avg_value=("annual_value", "mean"),
                  at_risk=("value_at_risk", "sum"),
                  breakeven_p=("breakeven_prob", "median"), contacted=("act", "sum")))
    seg["avg_value"] = seg["avg_value"].round(0)
    seg["at_risk"] = seg["at_risk"].round(0)
    seg["breakeven_p"] = (seg["breakeven_p"] * 100).round(1)
    print(seg.to_string())
    print("\n  breakeven_p falls as value rises: a whale earns action on thinner suspicion.")

    # ---------- robustness ---------------------------------------------------
    print("\n" + "=" * 90)
    print("CONCENTRATION CHECK — is the profit real, or is it three customers?")
    print("=" * 90)
    base = evaluate_policy(test, use_ev=True)["NET"]
    print(f"  NET on all {len(test):,} held-out customers   £{base:>8,.0f}")
    for k in [1, 5, 10]:
        n = evaluate_policy(test.drop(test.nlargest(k, "annual_value").index), use_ev=True)["NET"]
        print(f"  NET excluding top {k:>2} by value          £{n:>8,.0f}")
    f = test[test.ev_of_offer > 0].copy()
    ch = f["churned"] == 1
    dcf = _discount_cost(f["annual_value"], f["aov"])
    f["realised"] = np.where(ch, SAVE_RATE * (GROSS_MARGIN * f["annual_value"] - dcf), -dcf) - CONTACT_COST
    print(f"\n  per-contacted-customer:  median £{f['realised'].median():,.0f}   "
          f"mean £{f['realised'].mean():,.0f}   {(f['realised']>0).mean()*100:.0f}% profitable")

    # ---------- save-rate sensitivity (the shakiest assumption) --------------
    print("\n" + "=" * 90)
    print("SENSITIVITY — the save rate is our least defensible number. Does it survive?")
    print("=" * 90)
    for s in [0.10, 0.15, 0.20, 0.25, 0.30]:
        dd = scores.copy()
        tmp = decide(dd)
        tmp["ev_of_offer"] = expected_value_of_offer(tmp["churn_prob"], tmp["annual_value"],
                                                     tmp["aov"], s=s)
        tt = tmp[tmp["in_test_set"]]
        fl = tt[tt.ev_of_offer > 0]
        chh = fl["churned"] == 1
        dcx = _discount_cost(fl["annual_value"], fl["aov"])
        net = (s * (GROSS_MARGIN * fl.loc[chh, "annual_value"] - dcx[chh])).sum() \
              - dcx[~chh].sum() - len(fl) * CONTACT_COST
        flag = "  <- our assumption" if abs(s - SAVE_RATE) < 1e-9 else ""
        print(f"  save rate {s:>4.0%}:  contact {len(fl):>4}   NET £{net:>8,.0f}{flag}")
    print("\n  Below ~15% the programme stops paying. That is the number to validate first.")

    # ---------- the list -----------------------------------------------------
    act = d[d["act"]]
    print("\n" + "=" * 90)
    print(f"THE LIST — {len(act):,} customers worth saving "
          f"(£{act['value_at_risk'].sum():,.0f} of revenue at risk)")
    print("=" * 90)
    top = act.head(10)[["Customer ID", "churn_prob", "annual_value", "value_at_risk",
                        "ev_of_offer", "segment", "action"]].copy()
    top["churn_prob"] = (top["churn_prob"] * 100).round(0).astype(int).astype(str) + "%"
    for c in ["annual_value", "value_at_risk", "ev_of_offer"]:
        top[c] = "£" + top[c].round(0).astype(int).astype(str)
    top["action"] = top["action"].str.slice(0, 40)
    print(top.to_string(index=False))

    json.dump(dict(
        assumptions=dict(gross_margin=GROSS_MARGIN, offer_discount=OFFER_DISCOUNT,
                         offer_scope=OFFER_SCOPE, save_rate=SAVE_RATE,
                         contact_cost=CONTACT_COST, horizon_days=VALUE_HORIZON_DAYS),
        book=dict(customers=int(len(d)), forward_revenue=float(total_value),
                  revenue_at_risk=float(at_risk),
                  margin_at_risk=float(at_risk * GROSS_MARGIN)),
        list_size=int(len(act)),
        list_value_at_risk=float(act["value_at_risk"].sum()),
        best_policy=str(best["policy"]), best_net=float(best["NET"]),
        offer_design=rows, sweep=sweep_rows,
        robustness=dict(net_all=float(base),
                        median_per_contact=float(f["realised"].median()),
                        pct_contacts_profitable=float((f["realised"] > 0).mean())),
    ), open(ARTIFACTS / "decision_metrics.json", "w"), indent=2)
    print("\nsaved -> artifacts/customer_decisions.parquet, artifacts/decision_metrics.json")


if __name__ == "__main__":
    main()
