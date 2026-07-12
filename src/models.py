"""
BOX 3 — PREDICTION  (Understanding -> Prediction)

    26 customer features  ->  "this customer is 78% likely to churn"

Note the output. Project 1 produced a NUMBER (cycles left). This produces a
PROBABILITY. That difference drives everything downstream, and it's why this file
does something the jet-engine project never had to: check CALIBRATION.

WHY WE DO NOT REPORT ACCURACY
-----------------------------
Accuracy is the metric that lies. It answers "what fraction did we get right?",
which sounds useful and isn't. We report instead:
    ROC-AUC   — can the model RANK a churner above a stayer? (ordering quality)
    PRECISION — of the customers we flag, what share really churn? (offer waste)
    RECALL    — of the customers who churn, what share did we catch? (missed saves)
    PR-AUC    — precision/recall traded off across all thresholds
Precision is the one an operator feels: it is literally the share of retention
offers that don't get thrown in the bin.

WHY CALIBRATION IS THE POINT OF THIS FILE
-----------------------------------------
Box 4 computes:   expected value = P(churn) x customer_value x save_rate
That equation MULTIPLIES BY THE PROBABILITY. So the probability has to mean what
it says. If the model outputs "0.8" for a group of customers, roughly 80% of them
had better actually churn — otherwise every pound in the cost model is wrong.

Ranking well (good AUC) and being calibrated are DIFFERENT THINGS. A model can
order customers perfectly and still say "0.9" when it means "0.6". Tree ensembles
are notorious for exactly this. So we measure it (Brier score + a reliability
curve) and, if needed, fix it with isotonic calibration.

This is the step that would have been optional in Project 1 and is load-bearing here.

SPLIT: by customer, stratified by churn. Every customer appears in exactly one
side. Stratified so both sides carry the same ~49% churn rate.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score, f1_score)

from features import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
ARTIFACTS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

SEED = 42
TEST_FRAC = 0.25


def evaluate(y, p, name, threshold=0.5):
    """Threshold 0.5 is a PLACEHOLDER here. Box 4 replaces it with an economic one."""
    pred = (p >= threshold).astype(int)
    return dict(
        model=name,
        ROC_AUC=round(roc_auc_score(y, p), 3),
        PR_AUC=round(average_precision_score(y, p), 3),
        precision=round(precision_score(y, pred, zero_division=0), 3),
        recall=round(recall_score(y, pred, zero_division=0), 3),
        F1=round(f1_score(y, pred, zero_division=0), 3),
        Brier=round(brier_score_loss(y, p), 4),   # calibration: lower = better
    )


def bootstrap_auc_gap(y, p_a, p_b, n=2000, seed=0):
    """
    Is model B actually better than model A, or did it just get lucky on 1,077 people?

    Resample the test set with replacement, recompute the AUC gap each time, and
    look at the 95% interval. If it straddles zero, the "improvement" is noise and
    we should not pay complexity for it. This is the check that decides which model
    we ship — see the WINNER selection in main().
    """
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(mean_gap=round(float(d.mean()), 4),
                ci_low=round(float(lo), 4), ci_high=round(float(hi), 4),
                significant=bool(lo > 0))


def calibration_error(y, p, bins=10):
    """Expected Calibration Error: mean |predicted prob - actual rate| across bins."""
    frac, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    counts = np.histogram(p, bins=edges)[0]
    w = counts[: len(frac)] / counts[: len(frac)].sum()
    return float(np.sum(w * np.abs(frac - mean_pred)))


def plot_calibration(y, preds: dict, fname="calibration.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, p in preds.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", lw=1.6, ms=5, label=name)
    ax.set_xlabel("what the model SAID (predicted probability)")
    ax.set_ylabel("what ACTUALLY happened (observed churn rate)")
    ax.set_title("Reliability curve — do the probabilities mean what they say?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for name, p in preds.items():
        ax.hist(p, bins=25, alpha=0.45, label=name)
    ax.set_xlabel("predicted P(churn)")
    ax.set_ylabel("customers")
    ax.set_title("Where the model puts its bets")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("Box 4 MULTIPLIES by these probabilities. If the line bends away from the "
                 "diagonal,\nevery pound in the cost model is wrong.", fontsize=11)
    fig.tight_layout()
    out = REPORTS / fname
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    df = pd.read_parquet(ARTIFACTS / "customer_features.parquet")
    X = df[FEATURE_COLS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["churned"].to_numpy()

    # split by CUSTOMER, stratified by churn — same rate on both sides
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=TEST_FRAC, random_state=SEED, stratify=y
    )
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]
    print(f"train {len(ytr):,} customers ({ytr.mean()*100:.1f}% churn)   "
          f"test {len(yte):,} customers ({yte.mean()*100:.1f}% churn)\n")

    results, preds, models = [], {}, {}

    # ---- 1. BASELINE: logistic regression --------------------------------
    # Naturally well-calibrated (it optimises log-loss directly). Sets the bar.
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    logit.fit(Xtr, ytr)
    p = logit.predict_proba(Xte)[:, 1]
    results.append(evaluate(yte, p, "1. Logistic (baseline)"))
    preds["Logistic (baseline)"] = p
    models["logistic"] = logit

    # ---- 2. RandomForest --------------------------------------------------
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                max_features="sqrt", n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    results.append(evaluate(yte, p, "2. RandomForest"))
    preds["RandomForest"] = p
    models["rf"] = rf

    # ---- 3. Gradient boosting ---------------------------------------------
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                        max_leaf_nodes=31, l2_regularization=1.0,
                                        early_stopping=True, validation_fraction=0.15,
                                        random_state=SEED)
    gb.fit(Xtr, ytr)
    p = gb.predict_proba(Xte)[:, 1]
    results.append(evaluate(yte, p, "3. GradientBoosting"))
    preds["GradientBoosting"] = p
    models["gb"] = gb

    # ---- 4. Gradient boosting + isotonic calibration -----------------------
    # Trees rank well but their probabilities are often over-confident. Isotonic
    # regression re-maps them onto observed frequencies, using cross-validation
    # so we never calibrate on data the model trained on.
    gb_cal = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                       max_leaf_nodes=31, l2_regularization=1.0,
                                       early_stopping=True, validation_fraction=0.15,
                                       random_state=SEED),
        method="isotonic", cv=5,
    )
    gb_cal.fit(Xtr, ytr)
    p = gb_cal.predict_proba(Xte)[:, 1]
    results.append(evaluate(yte, p, "4. GB + isotonic calibration"))
    preds["GB + calibrated"] = p
    models["gb_calibrated"] = gb_cal

    table = pd.DataFrame(results)
    table["calib_error"] = [round(calibration_error(yte, preds[k]), 4) for k in preds]
    table.to_csv(REPORTS / "model_comparison.csv", index=False)

    print("SCORED ON 1,077 HELD-OUT CUSTOMERS (never seen in training)")
    print("=" * 100)
    print(table.to_string(index=False))
    print("\nBrier / calib_error: LOWER = probabilities more trustworthy.")
    print("ROC-AUC: ranking quality.  Precision: share of retention offers not wasted.")

    # ---- IS THE COMPLEXITY EARNING ITS KEEP? ------------------------------
    print("\n" + "=" * 100)
    print("BOOTSTRAP TEST: are the fancy models actually better, or just luckier?")
    print("=" * 100)
    gaps = {}
    for name in ["RandomForest", "GradientBoosting", "GB + calibrated"]:
        g = bootstrap_auc_gap(yte, preds["Logistic (baseline)"], preds[name])
        gaps[name] = g
        verdict = "REAL gain" if g["significant"] else "INDISTINGUISHABLE from baseline"
        print(f"  {name:<20} vs baseline:  AUC {g['mean_gap']:+.4f}  "
              f"95% CI [{g['ci_low']:+.4f}, {g['ci_high']:+.4f}]  -> {verdict}")

    # WINNER SELECTION, ON EVIDENCE:
    # No model beats the logistic baseline by a statistically real margin. When
    # models tie, the tiebreak is not taste — it's what the next box needs and what
    # a human can defend. Logistic regression is:
    #    - naturally calibrated (it optimises log-loss, so its probabilities mean
    #      what they say — exactly what Box 4's cost model multiplies by)
    #    - interpretable (a marketer can read the coefficients)
    #    - the simplest thing that works
    # So we ship the baseline. Complexity you cannot justify is complexity you
    # should not deploy.
    any_real = any(g["significant"] for g in gaps.values())
    if any_real:
        best_key, winner = "gb_calibrated", "4. GB + isotonic calibration"
    else:
        best_key, winner = "logistic", "1. Logistic (baseline)"
        print("\n  -> Nothing beat the baseline. SHIPPING THE BASELINE.")

    from sklearn.inspection import permutation_importance
    imp = permutation_importance(models[best_key], Xte, yte, n_repeats=5,
                                 random_state=SEED, scoring="roc_auc", n_jobs=-1)
    fi = (pd.DataFrame({"feature": FEATURE_COLS, "importance": imp.importances_mean})
          .sort_values("importance", ascending=False).reset_index(drop=True))
    fi.to_csv(REPORTS / "feature_importance.csv", index=False)
    print("\nTop 10 features the winning model actually leaned on:")
    print(fi.head(10).to_string(index=False))

    plot_calibration(yte, preds)

    # ---- persist everything Box 4 needs ------------------------------------
    # Score the FULL population (train + test) so the dashboard shows every customer.
    # Flag which rows were held out, so the honest numbers can be quoted from test only.
    p_all = models[best_key].predict_proba(X)[:, 1]
    out = df[["Customer ID", "churned"]].copy()
    out["churn_prob"] = np.round(p_all, 4)
    out["in_test_set"] = False
    out.loc[df.index[idx_te], "in_test_set"] = True
    for c in ["recency", "frequency", "monetary_total", "monetary_avg", "tenure",
              "own_mean_gap", "orders_last_90d", "revenue_last_90d",
              "revenue_last_365d", "orders_last_365d"]:
        out[c] = df[c].values
    out.to_parquet(ARTIFACTS / "customer_scores.parquet", index=False)

    joblib.dump({"model": models[best_key], "features": FEATURE_COLS},
                ARTIFACTS / "churn_model.joblib")
    json.dump({"winner": winner, "results": results, "bootstrap_vs_baseline": gaps,
               "test_size": int(len(yte)), "churn_rate": float(y.mean())},
              open(ARTIFACTS / "metrics.json", "w"), indent=2)
    print(f"\nWINNER (for Box 4): {winner}")
    print("saved -> artifacts/customer_scores.parquet, artifacts/churn_model.joblib")


if __name__ == "__main__":
    main()
