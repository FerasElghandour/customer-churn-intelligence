# Customer Churn Intelligence — £1.75M of Revenue at Risk

### 🔗 **[View the live dashboard →](https://feraselghandour.github.io/customer-churn-intelligence/)**

---

**The problem.** In ecommerce, nobody cancels. Customers just quietly stop buying, and you find out months later from a chart that's already bent. Across 4,306 active customers of a real UK retailer, **roughly half go silent within 90 days.**

**What this system does.** It scores every customer's probability of leaving, multiplies it by what that customer is actually worth, and compares the result against what a retention offer costs. The output is not a probability table — it's a ranked list of **who to save, worth how much, and what to do about them.**

**The result.**

| | |
|---|---|
| **£1,754,168** | of annual revenue sits in customers more likely than not to leave — **26% of the book** |
| **£701,667** | of that is gross margin |
| **2,300** | customers where the economics say intervening is worth it |
| **£1,142,638** | of revenue that list protects |

**But the headline finding isn't the model. It's this:**

> ### Offer design — not model accuracy — decides whether retention makes money.
>
> Same model. Same customers. Same 15% discount.
> **"15% off your next order"** → costs 15% of *one order* → **+£1,841 net**
> **"15% off, standing discount"** → costs 15% of *a year of orders* → **−£858 net**
>
> One marketing decision flips the programme from profitable to value-destroying. No amount of AUC rescues a badly designed offer.

---

## ⚠️ The #1 limitation — read this before believing any number above

**The whole programme rests on an assumed 25% save rate**, and we could not measure it.

| Save rate | Net result |
|---|---|
| 15% | **−£1,066** |
| 20% | **−£1,009** |
| **25%** (our assumption) | **+£1,841** |
| 30% | +£6,652 |

**Below roughly 25%, this campaign destroys value.** We took 25% from typical win-back ranges — a plausible industry figure, not a measured one. This dataset contains no retention campaigns, so it cannot tell us.

**Before a real brand spends a penny here, it must run a holdout test on a few hundred customers and measure its own save rate.** That single number decides whether any of this is worth doing. Everything else in this repo is downstream of it.

---

## The fleet, by value

| Tier | At risk | Worth contacting | Act above |
|---|---|---|---|
| **VIP** (>£10k/yr) | £175,565 | 16 of 102 | **10%** churn risk |
| **High** (£2k–10k) | £533,117 | 291 of 752 | **19%** |
| **Mid** (£500–2k) | £750,857 | 1,062 of 1,716 | **39%** |
| **Low** (<£500) | £294,630 | 931 of 1,736 | **73%** |

**The threshold is per customer, not per book.** A VIP earns a phone call at 10% suspicion, because a year of their margin dwarfs the cost of one discounted order. A low-value customer must clear 73% — there's simply less to win. **That falls out of the economics. It isn't a rule bolted on the side.**

## Chasing every churner loses money

Scored on 1,077 held-out customers against real outcomes:

| Policy | Contacted | Churners caught | Net |
|---|---|---|---|
| p ≥ 30% | 806 | **91%** | £2,454 |
| **p ≥ 50%** | 573 | 73% | **£2,509** |
| p ≥ 70% | 247 | 35% | £29 |
| EV-optimal (value-weighted) | 558 | 56% | £1,841 |

**Catching 91% of churners made less money than catching 73%.** Recall is not the objective. Profit is. This is the entire thesis of the decision layer.

---

## How it works

Six components. The machine learning is one of them, and it's the least interesting.

```
Reality  ->  Data  ->  Understanding  ->  Prediction  ->  Decision  ->  Action
             │         │                  │              │             │
             │         │                  │              │             └─ ranked action list
             │         │                  │              └─ EV = p×value×save − discount − cost
             │         │                  └─ logistic regression (it beat the fancy models)
             │         └─ RFM + cadence + trend + returns (26 features)
             └─ churn DEFINED, not observed: 90 days of silence
```

**Run the whole thing:**

```bash
pip install -r requirements.txt
python run.py          # data -> features -> models -> decisions -> dashboard
open docs/index.html
```

---

## Decision log — why each choice

### Why churn had to be *defined*, not looked up

**This is the hard part, and it's not the modelling.** Online Retail II is a **non-contractual** setting (Fader & Hardie): unlike SaaS or a gym membership, where churn is an *observed event* with a timestamp, retail customers never announce their departure — they **silently attrite.** There is no `churned` column in this data. There cannot be one.

So we built the label. **Every number in this project inherits that choice.**

**We use a fixed-window definition:** churned = no purchase in 90 days, among customers active in the prior 12 months. This is the standard textbook approach for retail churn.

**The window was measured, not assumed.** Across 27,029 real inter-purchase gaps: median **32 days**, 75th percentile **70 days**, 90th percentile **148 days**.

**Its honest cost:** ~19% of real inter-purchase gaps exceed 90 days, so **roughly one in five genuine repeat buyers gets mislabelled as a churner.** A 180-day window would cut that to 7% — but you'd learn a customer churned six months late, long past the point where an offer could work. **We traded label purity for actionability, deliberately.**

**We also excluded 954 long-dormant customers** (last purchase >12 months before cutoff). They're 89% "churned" — but they're not *at risk*, they're **gone**. Leaving them in inflates the churn rate and lets the model win by learning "recency = churn," which is true, trivial, and commercially useless. **A retention budget shouldn't be aimed at corpses.**

**What we deliberately did NOT use** (choice, not ignorance):
- **Dynamic per-customer thresholds** (churn = N× that customer's own average gap). Fixes the slow-buyer problem. We measured the cost and rejected it: **40% of our customers have fewer than 3 orders**, so no stable personal cadence exists — for them a "dynamic" rule collapses back to the population median. Worse, at 2× each customer's own gap the median implied window is **126 days**, but our label window is only **90 days long** — that's all the future the dataset contains. *We cannot observe silence that isn't in the data.* The blocker was never the code. It was the calendar.
- **Probabilistic buy-till-you-die models** — Pareto/NBD and BG/NBD (Fader & Hardie). These infer P(customer is still alive) from recency and frequency with **no arbitrary cutoff**, and are the gold standard for non-contractual CLV. They're the genuinely principled answer. We didn't use them because they don't give a better label for a classifier — **they replace the classifier.** That's a different project, and an honest one to build next.

### Why a baseline first — and why the baseline won

Logistic regression was built first, to set a bar. Then RandomForest and gradient boosting.

| Model | ROC-AUC | Precision | Calib. error |
|---|---|---|---|
| **Logistic (baseline)** ✅ | 0.763 | 0.675 | 0.035 |
| RandomForest | 0.766 | 0.693 | 0.036 |
| GradientBoosting | 0.764 | 0.668 | 0.052 |
| GB + isotonic | 0.763 | 0.671 | 0.029 |

Those gaps looked like noise, so **I bootstrapped the test set 2,000 times.** Every 95% confidence interval straddles zero:

```
RandomForest vs baseline:  AUC +0.0035  CI [-0.0114, +0.0177]  → INDISTINGUISHABLE
```

**Nothing beat the baseline.** So we shipped the baseline — it's naturally calibrated, interpretable, and simplest. **Complexity you cannot justify is complexity you should not deploy.** The bootstrap test runs on every execution; the winner is chosen by evidence, not preference.

### Why calibration mattered here (and didn't in the jet-engine project)

The cost model computes `EV = P(churn) × customer value × save rate`. **It multiplies by the probability.** So the probability has to *mean what it says* — if the model outputs 0.8 for a group, ~80% of them had better actually churn, or every pound of projected margin is fiction.

Ranking well and being calibrated are **different things**. A model can order customers perfectly and still say "0.9" when it means "0.6". We measured it (Brier + reliability curve): the models hug the diagonal, **expected calibration error 3.5%.** The money math is allowed to proceed.

### Why the safety margin lives in the decision layer

Two mistakes in this project cost wildly different amounts, **and it's the opposite of the jet-engine project:**

> **A churner we miss** costs only the contact fee — they never come back, so they never redeem.
> **A stayer we flag** redeems a discount on a purchase they'd have made at full price. **Pure margin burn.**

So the expensive error is the **false alarm**, not the miss. Which is why "catch every churner" is a terrible objective, and why the threshold is set by economics rather than by the model.

### Two bugs this build caught in itself

**1. The top "save this customer" was a ghost.** We first valued customers by annualising their lifetime run-rate. It produced this:

```
Customer 15098:  £39,916 spent | tenure 92 days | ONE single order
                 → extrapolated to a £158,364/year customer → ranked #1 on the action list
```

One bulk order × 4 = a phantom whale at the top of the list. A marketer would have phoned them, and the number would have been fiction. **Fixed: value = revenue actually spent in the last 365 days.** Observed, not extrapolated. **Extrapolation invents revenue. Observation does not.**

**2. The false alarm was priced ~10× too high.** We initially charged a false alarm as 15% of the customer's *annual* revenue. But "15% off your next order" costs 15% of **one order**. Correcting that inverted the conclusion — and turned into the project's headline finding about offer design.

### The idea from the jet-engine project that *failed* here

In predictive maintenance, the strongest feature was `sensor_drift` — a sensor measured against **its own baseline**. I expected the same trick to win here: `recency ÷ that customer's own average gap`.

**It lost.** Raw recency: AUC **0.707**. The ratio: **0.642**.

**Why is the interesting part.** For an engine, the baseline is *neutral* — a sensor starting at 47.3 instead of 47.5 is manufacturing noise, so dividing it out reveals signal. **For a customer, the baseline IS the signal.** A slow purchase rhythm is itself evidence of a weak relationship (`own_mean_gap` alone scores AUC 0.630). Dividing by it didn't normalise away noise — **it divided away information.**

> A customer who orders every 120 days isn't a normal customer on a slower clock. They're a *worse customer*. **Engines don't choose how much they love you.**

---

## Honest limitations

- **The 25% save rate is assumed, not measured.** See the warning at the top. This is the load-bearing assumption and it is unvalidated.
- **Churn is a definition we invented**, and a 19% mislabel rate is baked into every result.
- **ROC-AUC 0.763 is respectable, not spectacular.** Human behaviour is genuinely hard to predict from purchase history alone. The system works *because the decision layer is cost-aware*, not because the model is brilliant.
- **The EV-optimal policy did not beat a flat 50% cutoff** on this test set (bootstrapped difference: −£689, CI [−£3,855, +£2,463] — indistinguishable). We ship it because it's the defensible rule, not because the data proved it wins.
- **Customer values are capped at the 99th percentile.** Without the cap, the entire P&L rode on three customers. A policy whose profitability depends on 0.3% of the book is not a policy, it's a bet.
- **This is a benchmark dataset**, and a *mixed general retailer* at that — see below for why that matters enormously.
- **No feedback loop.** A real system would learn from whether its saves actually worked. A static dataset has nothing to feed back.

---

## Why this transfers

### Same skeleton as my [predictive maintenance project](https://feraselghandour.github.io/predictive-maintenance/) — with the cost asymmetry inverted

| | Jet engines | Customers |
|---|---|---|
| Entity | engine | customer |
| Time | flight cycle | day |
| Signals | 21 sensors | RFM, cadence, returns |
| Target | cycles until failure | will they buy in 90 days |
| **Miss** | engine fails in flight — **catastrophic** | they leave — we lose their margin |
| **False alarm** | service a healthy engine — **cheap** | discount someone who was staying — **expensive** |
| **Therefore** | lean cautious, eat false alarms | **be strict, protect margin** |

Same six boxes. Same code structure. **Opposite optimal policy** — because the economics inverted. The reusable asset is the method, not the numbers.

### 1. Churn is category-dependent. There is no universal window.

**"Churn" is a function of a category's repurchase physics.**

**Replenishment categories** (pet food, supplements, coffee) have **high switching cost and low, predictable churn.** A pet-food customer stays because *the pet is adjusted to the brand* — switching risks an upset stomach, and nobody gambles that to save £3. The purchase cycle is a metronome. **Silence past day 45 is an emergency.**

**Discovery categories** (fashion, curation boxes, homeware — *our dataset*) churn high because **novelty is the product.** No biological clock forces a repurchase. A 90-day gap may be entirely normal. **Silence is weak evidence**, and an offer fired at day 45 mostly discounts people who were coming back anyway.

**Consequence: the window must be derived from the category's actual repurchase cadence, not copied from this repo.** For a monthly-replenishment brand it might be **45 days**. For fashion, **180**. Using our 90 for either would be wrong in opposite directions. A useful rule of thumb: set it around the **75th–90th percentile of that category's real purchase gaps.**

**Our dataset is a mixed general retailer**, so it has a blended, messy cadence — which is exactly why a single fixed window is a reasonable default *here*, and exactly why it must be re-derived per category for a real brand. Multi-category brands may need **more than one window**: segment first, then define churn within each segment.

### 2. Prediction is only one lever — and it's the defensive one

This system answers *"who's about to leave, and who's worth saving?"* That protects value that already exists.

**There's an offensive lever upstream that this build never touches: raising customer value at acquisition.** Bundling to lift first-order AOV, better first-purchase product mix, subscribe-at-checkout — moves that make each customer *worth more before retention economics even apply.*

This matters because **the entire cost model is built on customer value `V`.** Every threshold in this project is downstream of it. **Raise `V` at acquisition and more customers become worth saving — the retention system gets stronger without a single change to the model.**

That's a **different system**, and the natural next build: **customer value optimization** — not *"who's leaving?"* but *"how do we make each customer worth more from the start?"* Naming it here so it's clear this project's scope is deliberate, not a blind spot.

### 3. What to actually do on day one with a real brand

1. **Measure the cadence.** Pull the inter-purchase gap distribution. Don't assume — look.
2. **Set the churn window from that distribution.** Not from a blog post, and not from this repo.
3. **Segment before defining churn** if the brand spans categories with different repurchase physics.
4. **Get the real economics:** actual offer cost, actual gross margin, and above all **the actual save rate from past win-back campaigns.**
5. **Then run this pipeline.** The skeleton doesn't change. The parameters do.

**Every number in this repo is a placeholder for a number a real brand would supply. The reusable asset is the method.**

---

## Repo structure

```
├── data/raw/               Online Retail II (UCI) — 1,067,371 real transactions
├── src/
│   ├── data.py             Box 1 — clean, measure cadence, DEFINE the churn target
│   ├── features.py         Box 2 — RFM + cadence + trend + returns (26 features)
│   ├── understand.py       Box 2 — which features actually separate churners
│   ├── models.py           Box 3 — logistic baseline -> RF -> GBM, + calibration
│   ├── decide.py           Box 4 — the cost model. Expected value, value-weighted.
│   └── dashboard.py        Box 5 — build the static dashboard
├── reports/                separation, calibration, threshold sweep, churn definition
├── docs/index.html         the deployed dashboard (GitHub Pages)
└── run.py                  one command, whole system
```

**Data:** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), UCI Machine Learning Repository — real UK ecommerce transactions, Dec 2009 – Dec 2011.

**References:** Fader, P.S. & Hardie, B.G.S. — contractual vs. non-contractual settings, and buy-till-you-die models (Pareto/NBD, BG/NBD), *Marketing Science*. Schmittlein, Morrison & Colombo (1987), "Counting Your Customers."

---

### About

Built by **Feras Elghandour** — part of a series on building intelligent systems that solve real-world problems.

🔗 [GitHub](https://github.com/FerasElghandour) · [Predictive Maintenance (Project 1)](https://feraselghandour.github.io/predictive-maintenance/) · [Site](https://tenx-gc.com)
