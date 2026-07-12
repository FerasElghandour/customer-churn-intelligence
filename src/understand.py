"""
BOX 2 (part 2) — WHICH FEATURES ACTUALLY SEPARATE CHURNERS?

Same triage as Project 1's 21 sensors, different nouns. There the question was
"does this sensor drift as the engine dies?" Here it's "does this behaviour differ
between customers who left and customers who stayed?"

We score separation two ways:
  AUC  — if I pick one churner and one survivor at random, how often does this
         feature rank them correctly? 0.5 = coin flip (useless). 1.0 = perfect.
         Robust to scale and to outliers, which matter a lot with money columns.
  LIFT — churn rate in the top decile of the feature vs the bottom decile.
         Speaks in the language of a marketer, not a statistician.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from data import load_raw, clean, build_target
from features import build_features, FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
REPORTS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)


def separation_table(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in FEATURE_COLS:
        v = X[c].astype(float)
        if v.nunique() < 2:
            continue
        auc = roc_auc_score(y, v)
        # decile lift: churn rate in the top 10% of this feature vs the bottom 10%
        try:
            q_hi, q_lo = v.quantile(0.9), v.quantile(0.1)
            hi = y[(v >= q_hi).values].mean() if (v >= q_hi).sum() else np.nan
            lo = y[(v <= q_lo).values].mean() if (v <= q_lo).sum() else np.nan
        except Exception:
            hi = lo = np.nan
        rows.append(dict(
            feature=c,
            auc=round(auc, 3),
            separation=round(abs(auc - 0.5) * 2, 3),   # 0 = useless, 1 = perfect
            churn_top_decile=round(hi * 100, 1) if pd.notna(hi) else np.nan,
            churn_bottom_decile=round(lo * 100, 1) if pd.notna(lo) else np.nan,
            mean_churners=round(v[y == 1].mean(), 1),
            mean_stayers=round(v[y == 0].mean(), 1),
        ))
    t = pd.DataFrame(rows).sort_values("separation", ascending=False).reset_index(drop=True)
    t["verdict"] = np.where(t.separation >= 0.30, "STRONG",
                     np.where(t.separation >= 0.10, "useful", "weak / noise"))
    return t


def plot_top_features(X, y, table, fname="feature_separation.png", n=6):
    top = table.head(n)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, c in zip(axes.ravel(), top):
        v = X[c].astype(float)
        # clip the long money tails so the plot is readable
        hi = v.quantile(0.98)
        bins = np.linspace(v.min(), hi if hi > v.min() else v.max() + 1, 30)
        ax.hist(v[y == 0].clip(upper=hi), bins=bins, alpha=0.65, label="stayed", density=True)
        ax.hist(v[y == 1].clip(upper=hi), bins=bins, alpha=0.65, label="churned", density=True)
        auc = table.loc[table.feature == c, "auc"].iloc[0]
        ax.set_title(f"{c}   (AUC {auc})", fontsize=10)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Do churners and stayers actually look different?\n"
                 "Where the two curves separate, there is signal.", fontsize=12)
    fig.tight_layout()
    out = REPORTS / fname
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_recency_ratio_story(X, y, fname="recency_vs_ratio.png"):
    """
    THE PLOT THAT PROVED ME WRONG.

    I expected recency_ratio (recency measured against the customer's own rhythm)
    to beat raw recency, because that trick worked in the jet-engine project.
    It did not. Raw recency: AUC 0.707. The ratio: 0.642.

    The reason is the interesting bit, and it's a real difference between machines
    and people. For an engine, the baseline is neutral — a sensor's starting value
    is just manufacturing variation, so dividing it out removes noise. For a
    customer, the baseline IS the signal: a slow rhythm is itself evidence of a
    weak relationship (own_mean_gap alone scores AUC 0.630). Dividing recency by
    it throws away information rather than normalising it.

    We keep all three columns (recency, own_mean_gap, recency_ratio) and let the
    model decide how to combine them. Which is the point of showing it the truth.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, col, title in [
        (axes[0], "recency", "Raw recency — 'days since last order'\nAUC 0.707  ← WINNER"),
        (axes[1], "recency_ratio", "Recency ÷ their OWN normal gap\nAUC 0.642  ← my clever idea, beaten"),
    ]:
        v = X[col].astype(float)
        bins = pd.qcut(v, 10, duplicates="drop", labels=False)
        rate = pd.DataFrame({"b": bins, "y": y}).groupby("b")["y"].mean() * 100
        ax.bar(rate.index, rate.values, color="#c0504d")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("decile (low → high)")
        ax.set_ylabel("churn rate %")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("The trick that won on jet engines LOST on customers.\n"
                 "For an engine, the baseline is neutral noise. For a customer, a slow rhythm is "
                 "itself a warning — so dividing it out destroys signal.", fontsize=11)
    fig.tight_layout()
    out = REPORTS / fname
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    purchases, returns = clean(load_raw())
    target, feat, lab, meta = build_target(purchases)
    X = build_features(feat, returns, meta["cutoff"], set(target["Customer ID"]))
    df = target.merge(X, on="Customer ID", how="left")
    df.to_parquet(ARTIFACTS / "customer_features.parquet", index=False)

    y = df["churned"].to_numpy()
    table = separation_table(df, y)
    table.to_csv(REPORTS / "feature_separation.csv", index=False)

    print(f"customers {len(df):,}   churn rate {y.mean()*100:.1f}%   features {len(FEATURE_COLS)}\n")
    print(table[["feature", "auc", "separation", "churn_top_decile",
                 "churn_bottom_decile", "verdict"]].to_string(index=False))

    plot_top_features(df, y, table)
    plot_recency_ratio_story(df, y)
    print("\nplots -> reports/feature_separation.png, reports/recency_vs_ratio.png")


if __name__ == "__main__":
    main()
