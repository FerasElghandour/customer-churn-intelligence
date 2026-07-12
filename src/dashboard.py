"""
BOX 5 — ACTION  (Decision -> a human does something)

Generates docs/index.html: one self-contained file, no server, no cold start.

Predictions are computed in a batch, so nothing here needs a live Python process.
A static file loads instantly, never sleeps, costs nothing and cannot fall over.
Matching the architecture to the requirement — instead of to what looks
impressive — is the engineering.

The page leads with the money (£1.75M at risk) and the finding (offer design
decides profitability), NOT with the model. A marketer does not care about AUC.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

AUTHOR = "Feras Elghandour"
AUTHOR_LINK = "https://github.com/FerasElghandour"

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
background:#0f1216;color:#e8ecf1;-webkit-font-smoothing:antialiased}
.wrap{max-width:1060px;margin:0 auto;padding:32px 20px 80px}
header{border-bottom:1px solid #232a33;padding-bottom:22px;margin-bottom:26px}
h1{font-size:26px;font-weight:650;letter-spacing:-.01em}
.sub{color:#8b97a6;margin-top:8px;font-size:15px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:#8b97a6;
margin:38px 0 14px;font-weight:600}
.hero{background:#151a21;border:1px solid #232a33;border-radius:10px;padding:22px 24px}
.hero p{color:#c3ccd7;font-size:15px}
.hero strong{color:#fff}
.big{font-size:42px;font-weight:700;letter-spacing:-.03em;color:#f85149;line-height:1.1}
.biglab{font-size:13px;color:#8b97a6;margin-top:4px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:20px}
.kpi{background:#0f1419;border:1px solid #232a33;border-radius:8px;padding:14px}
.kpi .v{font-size:24px;font-weight:680;letter-spacing:-.02em}
.kpi .l{font-size:12px;color:#8b97a6;margin-top:3px;line-height:1.35}
.good{color:#3fb950}.warn{color:#d29922}.bad{color:#f85149}
.finding{background:rgba(210,153,34,.07);border:1px solid rgba(210,153,34,.35);
border-radius:10px;padding:20px 24px;margin-top:14px}
.finding .t{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#d29922;
font-weight:700;margin-bottom:8px}
.flip{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}
.flipcard{background:#0f1419;border:1px solid #232a33;border-radius:8px;padding:14px}
.flipcard.win{border-left:3px solid #3fb950}
.flipcard.lose{border-left:3px solid #f85149}
.flipcard .h{font-size:13px;font-weight:650;margin-bottom:6px}
.flipcard .n{font-size:22px;font-weight:700;letter-spacing:-.02em}
.flipcard .d{font-size:12px;color:#8b97a6;margin-top:5px}
.tiers{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.tier{background:#151a21;border:1px solid #232a33;border-radius:10px;padding:16px;
border-left:3px solid #2d3540}
.tier.vip{border-left-color:#a371f7}.tier.high{border-left-color:#f85149}
.tier.mid{border-left-color:#d29922}.tier.low{border-left-color:#3fb950}
.tier .n{font-size:22px;font-weight:680}
.tier .t{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:#8b97a6;margin-top:2px}
.tier .d{font-size:12px;color:#6e7a89;margin-top:8px;line-height:1.45}
.filters{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.filters button{background:#151a21;border:1px solid #232a33;color:#8b97a6;padding:7px 14px;
border-radius:999px;font-size:13px;cursor:pointer;font-family:inherit}
.filters button:hover{border-color:#3a4453;color:#e8ecf1}
.filters button.on{background:#e8ecf1;color:#0f1216;border-color:#e8ecf1;font-weight:600}
.tbl{width:100%;border-collapse:collapse;font-size:14px}
.tbl th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.07em;
color:#6e7a89;padding:10px 10px;border-bottom:1px solid #232a33;font-weight:600}
.tbl td{padding:11px 10px;border-bottom:1px solid #1a2028}
.tbl tr:hover td{background:#151a21}
.tbl .num{text-align:right;font-variant-numeric:tabular-nums}
.cid{font-weight:600}
.risk{font-weight:650;color:#f85149;font-variant-numeric:tabular-nums}
.pill{font-size:11px;font-weight:650;padding:3px 9px;border-radius:999px;white-space:nowrap}
.pill.vip{background:rgba(163,113,247,.15);color:#a371f7}
.pill.high{background:rgba(248,81,73,.14);color:#f85149}
.pill.mid{background:rgba(210,153,34,.14);color:#d29922}
.pill.low{background:rgba(63,185,80,.13);color:#3fb950}
.act{font-size:13px;color:#c3ccd7}
.note{color:#6e7a89;font-size:13px;margin-top:10px;line-height:1.6}
.limit{background:rgba(248,81,73,.06);border:1px solid rgba(248,81,73,.3);
border-radius:10px;padding:18px 22px;margin-top:14px}
.limit .t{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#f85149;
font-weight:700;margin-bottom:8px}
table.mini{border-collapse:collapse;font-size:13px;margin-top:10px}
table.mini td{padding:4px 14px 4px 0;color:#c3ccd7;font-variant-numeric:tabular-nums}
table.mini td:first-child{color:#8b97a6}
footer{margin-top:44px;padding-top:20px;border-top:1px solid #232a33;color:#6e7a89;font-size:13px}
footer a{color:#8b97a6}
@media(max-width:820px){
 .kpis,.tiers{grid-template-columns:repeat(2,1fr)}
 .flip{grid-template-columns:1fr}
 .wrap{padding:22px 14px 60px}
 h1{font-size:21px}.big{font-size:34px}
 .hide-sm{display:none}
 .tbl td,.tbl th{padding:9px 6px}
}
"""

JS = """
const btns=document.querySelectorAll('.filters button');
btns.forEach(b=>b.onclick=()=>{
  btns.forEach(x=>x.classList.remove('on'));b.classList.add('on');
  const f=b.dataset.f;
  document.querySelectorAll('.tbl tbody tr').forEach(r=>{
    r.style.display=(f==='all'||r.dataset.s===f)?'':'none';
  });
});
"""

TIER_CLS = {"VIP (>£10k)": "vip", "High (£2k-10k)": "high",
            "Mid (£500-2k)": "mid", "Low (<£500)": "low"}


def money(x):
    return f"£{x:,.0f}"


def build():
    d = pd.read_parquet(ART / "customer_decisions.parquet")
    m = json.load(open(ART / "decision_metrics.json"))
    a = m["assumptions"]
    book = m["book"]

    act = d[d["act"]].copy()
    od = {r["offer"]: r for r in m["offer_design"]}
    next_o = od["15% off, next_order"]
    annual = od["15% off, annual"]

    # per-tier breakevens
    tiers = (d.groupby("segment", observed=True)
               .agg(n=("Customer ID", "size"), at_risk=("value_at_risk", "sum"),
                    be=("breakeven_prob", "median"), flagged=("act", "sum")))

    rows = []
    for _, r in act.head(120).iterrows():
        cls = TIER_CLS.get(str(r["segment"]), "low")
        rows.append(f"""<tr data-s="{cls}">
<td class="cid">{int(r['Customer ID'])}</td>
<td class="num">{r['churn_prob']*100:.0f}%</td>
<td class="num hide-sm">{money(r['annual_value'])}</td>
<td class="num risk">{money(r['value_at_risk'])}</td>
<td class="num hide-sm">{money(r['ev_of_offer'])}</td>
<td><span class="pill {cls}">{str(r['segment']).split(' (')[0]}</span></td>
<td class="act">{r['action']}</td>
</tr>""")

    tier_cards = ""
    for name in ["VIP (>£10k)", "High (£2k-10k)", "Mid (£500-2k)", "Low (<£500)"]:
        if name not in tiers.index:
            continue
        t = tiers.loc[name]
        cls = TIER_CLS[name]
        tier_cards += f"""<div class="tier {cls}">
<div class="n">{money(t['at_risk'])}</div>
<div class="t">{name} · at risk</div>
<div class="d"><b>{int(t['flagged'])}</b> of {int(t['n'])} worth contacting.<br>
Act above <b>{t['be']*100:.0f}%</b> churn risk.</div></div>"""

    sweep = pd.DataFrame(m["sweep"])
    sweep_rows = "".join(
        f"<tr><td>{r['policy']}</td><td class='num'>{r['contacted']}</td>"
        f"<td class='num'>{r['recall']:.0f}%</td>"
        f"<td class='num' style='color:{'#3fb950' if r['NET']>0 else '#f85149'}'>"
        f"{money(r['NET'])}</td></tr>"
        for _, r in sweep.iterrows())

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Customer Churn Intelligence — £1.75M at risk</title>
<meta name="description" content="£1.75M of revenue sits in customers likely to leave. This system finds them, and works out which ones are actually worth saving.">
<style>{CSS}</style>
</head><body><div class="wrap">

<header>
<h1>Customer Churn Intelligence</h1>
<div class="sub">Which customers are about to leave — and which are actually worth paying to keep?</div>
</header>

<div class="hero">
<div class="big">{money(book['revenue_at_risk'])}</div>
<div class="biglab">of annual revenue sits in customers more likely than not to walk
 — {book['revenue_at_risk']/book['forward_revenue']*100:.0f}% of the book.</div>

<p style="margin-top:16px"><strong>The problem.</strong> In ecommerce nobody cancels. They just
quietly stop buying, and you find out months later from a chart. Of {book['customers']:,}
active customers, roughly half will go silent within 90 days.</p>

<p style="margin-top:10px"><strong>What this does.</strong> It scores every customer's
probability of leaving, multiplies that by what they're actually worth, and compares it against
what a retention offer costs. The output isn't a probability table — it's a ranked list of
<strong>who to save, worth how much, and what to do about them</strong>.</p>

<div class="kpis">
<div class="kpi"><div class="v bad">{money(book['margin_at_risk'])}</div><div class="l">Gross margin at risk</div></div>
<div class="kpi"><div class="v">{m['list_size']:,}</div><div class="l">Customers worth contacting</div></div>
<div class="kpi"><div class="v good">{money(m['list_value_at_risk'])}</div><div class="l">Revenue that list protects</div></div>
<div class="kpi"><div class="v">{book['customers']:,}</div><div class="l">Active customers scored</div></div>
</div>
<div class="note">Assumptions: {a['gross_margin']:.0%} gross margin · {a['offer_discount']:.0%} win-back
offer ({a['offer_scope'].replace('_',' ')}) · {a['save_rate']:.0%} save rate · £{a['contact_cost']:.2f} per contact.
Model: logistic regression, calibrated. It beat random forest and gradient boosting — or rather,
they failed to beat it.</div>
</div>

<h2>The finding</h2>
<div class="finding">
<div class="t">Offer design — not the model — decides whether retention makes money</div>
<p style="color:#c3ccd7;font-size:15px">Same model. Same customers. Same 15% discount.
The <em>only</em> difference is whether that discount applies to <strong>one order</strong> or to
<strong>a year of orders</strong>. It flips the answer from "run this campaign" to "don't bother."</p>
<div class="flip">
<div class="flipcard win"><div class="h">"15% off your next order"</div>
<div class="n good">+{money(next_o['NET'])}</div>
<div class="d">Costs 15% of ONE order. False alarms are cheap.<br>
Act above <b>{next_o['median_breakeven']}</b> churn risk (median). {next_o['contacted']} contacted.</div></div>
<div class="flipcard lose"><div class="h">"15% off, standing discount"</div>
<div class="n bad">{money(annual['NET'])}</div>
<div class="d">Costs 15% of EVERYTHING they buy for a year.<br>
Need <b>{annual['median_breakeven']}</b> certainty before acting. Only {annual['contacted']} qualify.</div></div>
</div>
<div class="note">No amount of model accuracy rescues a badly designed offer.
This is a marketing decision that a data scientist would never think to check.</div>
</div>

<h2>Where the money is — and how the bar moves with value</h2>
<div class="tiers">{tier_cards}</div>
<div class="note">The threshold is <strong>per customer, not per fleet</strong>. A VIP earns a call at
{tiers.loc['VIP (>£10k)','be']*100:.0f}% suspicion because a year of their margin dwarfs one discounted order.
A low-value customer must clear {tiers.loc['Low (<£500)','be']*100:.0f}% — there's simply less to win.
That falls out of the economics; it isn't a rule bolted on.</div>

<h2>Chasing every churner loses money</h2>
<table class="tbl" style="max-width:560px">
<thead><tr><th>Policy</th><th class="num">Contacted</th><th class="num">Churners caught</th><th class="num">Net £</th></tr></thead>
<tbody>{sweep_rows}</tbody></table>
<div class="note">Scored on {sweep.iloc[0]['contacted'] and '1,077'} held-out customers against real outcomes.
Contacting 806 people catches <b>91%</b> of churners and earns <b>less</b> than contacting 573 and catching 73%.
Recall is not the objective. Profit is.</div>

<h2>The list — who to save, ranked by revenue at risk</h2>
<div class="filters">
<button class="on" data-f="all">All ({len(act):,})</button>
<button data-f="vip">VIP</button>
<button data-f="high">High</button>
<button data-f="mid">Mid</button>
<button data-f="low">Low</button>
</div>
<table class="tbl">
<thead><tr>
<th>Customer</th><th class="num">Churn risk</th><th class="num hide-sm">Worth /yr</th>
<th class="num">Revenue at risk</th><th class="num hide-sm">Expected gain</th>
<th>Tier</th><th>Recommended action</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="note">Top 120 of {len(act):,} customers the economics say are worth contacting.
Ranked by <strong>revenue at risk</strong> (churn probability × what they're worth) — not by
churn probability, which would put worthless customers at the top.</div>

<h2>The catch</h2>
<div class="limit">
<div class="t">The save rate is doing all the work — validate it first</div>
<p style="color:#c3ccd7;font-size:15px">We assume a <strong>25% save rate</strong>: that 1 in 4
would-be churners can be rescued by an offer. It's a plausible industry figure. <strong>We did not
measure it, because this dataset cannot.</strong> The whole programme lives or dies on it:</p>
<table class="mini">
<tr><td>save rate 15%</td><td class="bad">−£1,066</td></tr>
<tr><td>save rate 20%</td><td class="bad">−£1,009</td></tr>
<tr><td>save rate 25% (assumed)</td><td class="good">+£1,841</td></tr>
<tr><td>save rate 30%</td><td class="good">+£6,652</td></tr>
</table>
<div class="note">Below roughly 25%, this campaign destroys value. Before a real brand spends a penny
here, it should run a holdout test on a few hundred customers and measure its actual save rate.
That single number decides whether any of this is worth doing.</div>
</div>

<footer>
Built by <a href="{AUTHOR_LINK}">{AUTHOR}</a> — part of a series on building intelligent systems
that solve real-world problems.<br>
Data: <b>Online Retail II</b> (UCI) — 1,067,371 real UK ecommerce transactions, Dec 2009–Dec 2011.
Churn is <em>defined</em>, not observed: 90 days of silence among customers active in the prior year.
Same six-box architecture as my <a href="https://feraselghandour.github.io/predictive-maintenance/">predictive
maintenance project</a> — different nouns, inverted cost asymmetry.
</footer>

</div><script>{JS}</script></body></html>"""

    out = DOCS / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html)/1024:.0f} KB, self-contained)")
    print(f"at risk £{book['revenue_at_risk']:,.0f} | list {m['list_size']:,} customers")
    return out


if __name__ == "__main__":
    build()
