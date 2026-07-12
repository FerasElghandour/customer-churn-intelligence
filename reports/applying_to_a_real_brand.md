# How this applies to a real brand

*(Folds into the README. Written for an operator, not a data scientist.)*

This project is built on a **mixed general retailer** — a UK gift and homeware wholesaler selling everything from candles to cake stands. That matters more than it sounds, and it shapes exactly how you'd port this system to a specific brand.

---

## 1. Churn is category-dependent. There is no universal window.

**"Churn" is not a fixed concept. It's a function of a category's repurchase physics.**

Two brands can have identical revenue and identical customer counts, and a customer who's been silent for 60 days means something completely different in each.

**Replenishment / habitual categories** — pet food, supplements, coffee, skincare refills.
These have **high switching cost and low, predictable churn.** A pet-food customer stays on one brand because *the pet is adjusted to it* — switching risks an upset stomach and a fussy animal, and the owner won't gamble on that to save £3. The purchase cycle is a metronome: a 30-day bag of food gets reordered around day 30. **Silence past day 45 is a genuine emergency.** These customers rarely leave, and when they do, they leave for a reason you can usually find.

**Discovery / novelty categories** — fashion, curation boxes, gifting, homeware.
**Novelty is the product.** Customers churn high because that's what the category is for. There's no biological clock forcing a repurchase. A 90-day gap might be completely normal — they simply didn't need a new cushion. **Silence is weak evidence here**, and a retention offer fired at day 45 would mostly be a discount handed to someone who was coming back anyway.

### The operational consequence

**The churn window must be derived from the category's natural repurchase cycle — not copied from a project like this one.**

For a real brand, the first thing you do is exactly what we did in Box 1: **measure the actual inter-purchase cadence, then set the window against it.** The method transfers. The number 90 does not.

A useful rule of thumb: set the window somewhere around the **75th–90th percentile of that category's real purchase gaps** — far enough out that you're not flagging normal buyers, close enough in that you can still act. For a monthly-replenishment brand that might be **45 days**. For a fashion brand it might be **180**. Using our 90 for either would be wrong in opposite directions.

**Our dataset is a mixed general retailer, so it has a messy blended cadence** — a bit of replenishment, a lot of discovery, all averaged together. That's precisely why a single fixed window is a reasonable default *here*, and precisely why it would need re-deriving per category for a real brand. For a multi-category brand, you may need **more than one window** — segment first, then define churn within each segment.

---

## 2. Prediction is only one lever. This system doesn't touch the bigger one.

This system answers one question: **"Who is about to leave, and which of them is worth saving?"**

That's a *defensive* lever. It protects value that already exists.

There's an **offensive lever upstream that this build never touches: raising customer value at acquisition.** Bundling to lift first-order AOV, better product mix on the first purchase, subscription-at-checkout — moves that make each customer *worth more before retention economics even apply.*

Why this matters: the entire Box 4 cost model is built on customer value `V`. Every threshold, every "who's worth saving" decision, is downstream of how much a customer is worth. **Raise `V` at acquisition and you don't just earn more — you also make more customers worth saving.** The retention system gets more powerful without a single change to the model.

**That's a different system**, and it's the natural next build:

> **Customer value optimization** — not "who's leaving?" but "how do we make each customer worth more from the start?"

Naming it here so it's clear that this project's scope is deliberate, not a blind spot. **Predicting churn is not the same as maximising customer value.** It's one lever of several, and it's the *defensive* one.

---

## 3. What to actually do on day one with a real brand

1. **Measure the cadence.** Pull the inter-purchase gap distribution. Don't assume — look.
2. **Set the churn window from that distribution**, not from a blog post or from this repo.
3. **Segment before defining churn** if the brand spans categories with different repurchase physics.
4. **Get the real numbers for the cost model** — actual retention offer cost, actual gross margin, actual historical save rate from past win-back campaigns. Box 4's economics are only as good as those three inputs.
5. **Then run this exact pipeline.** The skeleton doesn't change. The parameters do.

**The reusable asset is the method. Every number in this repo is a placeholder for a number a real brand would supply.**
