# How churn is defined here — and why it's a decision, not a lookup

*(This section folds into the README. Kept as its own file so it can be edited independently.)*

## Churn cannot be observed in this dataset. It has to be inferred.

There are two kinds of customer relationship, and they demand completely different methods.

**Contractual settings** — SaaS, gym memberships, insurance, mobile plans. Churn is an **observed event**. The customer cancels, and you get a timestamp. The label is handed to you.

**Non-contractual settings** — retail, ecommerce, most DTC. Churn is **never observed**. Customers don't announce their departure; they *silently attrite*. Online Retail II is squarely non-contractual (the distinction is Fader & Hardie's, in *Marketing Science*).

**This is the whole problem.** There is no column in this dataset called `churned`. There cannot be one. Whether a customer has "left" is not a fact we look up — it's a **definition we impose**, and every number downstream inherits it.

## What we use: a fixed-window definition

> **Churned = no purchase in 90 days**, among customers still active within the prior 12 months.

This is the standard textbook approach for retail churn. Fixed thresholds of 3 / 6 / 12 months are the common convention in applied work, and 90 days is the one a retention team can actually act on.

**The window was chosen against the data, not by habit.** Measured purchase cadence across 27,029 real inter-purchase gaps:

| Percentile | Gap |
|---|---|
| 50th (median) | **32 days** |
| 75th | 70 days |
| 90th | 148 days |

## Its known limitation, stated plainly

**A fixed window mislabels slow-but-alive buyers.** If a customer's natural rhythm is 120 days, 90 days of silence is not churn — it's Tuesday.

In this data, **~19% of all real inter-purchase gaps exceed 90 days.** So roughly **one in five genuine repeat buyers gets labelled a churner.** That is the price of the choice, and it is baked into every result in this project.

Widening to 180 days would cut that error to 7% — but you'd learn a customer had churned six months after the fact, long past the point where an offer could bring them back. **We traded label purity for actionability, on purpose.**

## What we deliberately did NOT use

Naming these matters, because it's the difference between a considered choice and not knowing the alternatives existed.

### (a) Dynamic per-customer thresholds

Define churn as a multiple of **each customer's own** average inter-purchase time. This directly fixes the slow-buyer problem — everyone gets judged against their own rhythm.

**We measured what it would cost and rejected it, for two reasons visible in this specific dataset:**

- **Coverage.** 40% of our eligible customers have fewer than 3 orders, so they have no stable personal cadence. 23% have exactly *one* order — no gap exists at all. For those customers a "dynamic" threshold collapses back to the population median, i.e. the fixed rule again. A dynamic label that is fixed for 40% of the population isn't dynamic.
- **Observability.** At 2× each customer's own mean gap, the median implied window is **126 days**; the 90th percentile is **251 days**. But our label window is only **90 days long** — that is all the future the dataset contains after the cutoff. *We cannot observe 126 days of silence that isn't in the data.* Fixing it means moving the cutoff back ~250 days, sacrificing a third of the feature history and shifting the study into a different seasonal regime.

The blocker was never the code. It was the calendar.

### (b) Probabilistic "buy-till-you-die" models — Pareto/NBD and BG/NBD

Fader & Hardie's family of models infer **P(customer is still alive)** directly from recency and frequency, with **no arbitrary cutoff at all**. This is the gold standard for CLV in non-contractual settings, and it is the genuinely principled answer to the problem this section describes.

We didn't use it because it isn't a better label for our classifier — **it replaces the classifier.** That's a different project, and an honest one to build next.

## How we recover most of the benefit anyway

**We moved purchase cadence out of the label and into the features.**

The model gets `recency ÷ this customer's own mean gap` as an input. A customer 80 days silent whose rhythm is 30 days is in serious trouble. A customer 80 days silent whose rhythm is 120 days is **fine**. The model learns that distinction on its own — without us mangling the label or throwing away half the calendar.

This is the same move as the predictive-maintenance project's `sensor_drift` feature: **measure the unit against its own baseline, never against a population constant.** Health is a deviation, not a value. It turns out customers work the same way as jet engines.

---

**References**

- Fader, P.S. & Hardie, B.G.S. — work on contractual vs. non-contractual settings and buy-till-you-die models (Pareto/NBD, BG/NBD), *Marketing Science*.
- Schmittlein, Morrison & Colombo (1987) — "Counting Your Customers: Who Are They and What Will They Do Next?" (the original Pareto/NBD).
