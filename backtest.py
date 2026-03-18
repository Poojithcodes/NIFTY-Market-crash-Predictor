"""
backtest.py — Strategy backtesting for the Nifty 50 Crash Predictor.

Logic
-----
- When the model predicts crash probability > threshold, move to cash
- Otherwise stay invested in Nifty
- Compare cumulative returns vs simple buy-and-hold

Run:
    python backtest.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from model import download_data, engineer_features, train_model, FEATURES


THRESHOLD = 0.40


def run_backtest(df, model, threshold=THRESHOLD):
    """
    Simulate the strategy on the test set (last 20% of data).
    Returns a DataFrame with daily returns for both strategies.
    """
    X     = df[FEATURES]
    split = int(len(X) * 0.8)

    test_df        = df.iloc[split:].copy()
    X_test         = X.iloc[split:]

    test_df["pred_prob"]    = model.predict_proba(X_test)[:, 1]
    test_df["signal"]       = (test_df["pred_prob"] < threshold).astype(int)

    test_df["daily_ret"]    = test_df["Close"].pct_change()
    test_df["strategy_ret"] = test_df["signal"].shift(1) * test_df["daily_ret"]

    test_df["bah_growth"]      = 100 * (1 + test_df["daily_ret"]).cumprod()
    test_df["strategy_growth"] = 100 * (1 + test_df["strategy_ret"]).cumprod()

    return test_df


def compute_metrics(test_df):
    """Print key performance metrics for both strategies."""
    trading_days = 252
    for name, col in [("Buy & Hold", "daily_ret"), ("Model Strategy", "strategy_ret")]:
        rets   = test_df[col].dropna()
        total  = (1 + rets).prod() - 1
        ann    = (1 + total) ** (trading_days / len(rets)) - 1
        vol    = rets.std() * np.sqrt(trading_days)
        sharpe = ann / vol if vol > 0 else 0
        dd     = ((1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1).min()

        print(f"\n  {name}")
        print(f"    Total return      : {total*100:+.1f}%")
        print(f"    Annualised return  : {ann*100:+.1f}%")
        print(f"    Annualised vol     : {vol*100:.1f}%")
        print(f"    Sharpe ratio       : {sharpe:.2f}")
        print(f"    Max drawdown       : {dd*100:.1f}%")


def plot_results(test_df, save_path="backtest_results.png"):
    """Save the main backtest chart."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle("Nifty 50 Crash Predictor — Backtest Results", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(test_df.index, test_df["bah_growth"],      label="Buy & Hold",     color="#888")
    ax1.plot(test_df.index, test_df["strategy_growth"], label="Model Strategy", color="#1a73e8", linewidth=1.8)
    ax1.set_ylabel("Portfolio value (Rs)")
    ax1.set_title("Cumulative Returns — Rs 100 invested")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(test_df.index, test_df["pred_prob"], color="#e8491a", linewidth=1)
    ax2.axhline(THRESHOLD, linestyle="--", color="#333", linewidth=0.8, label=f"Threshold ({THRESHOLD})")
    ax2.fill_between(test_df.index, test_df["pred_prob"], THRESHOLD,
                     where=(test_df["pred_prob"] > THRESHOLD),
                     alpha=0.25, color="#e8491a", label="In cash (crash risk)")
    ax2.set_ylabel("Crash probability")
    ax2.set_title("Model Crash Probability Over Time")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    ax3.fill_between(test_df.index, test_df["crash"],
                     alpha=0.7, color="#c0392b", label="Actual crash days")
    ax3.set_ylabel("Crash (1/0)")
    ax3.set_title("Actual Crash Events")
    ax3.set_ylim(0, 1.2)
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved to {save_path}")


def plot_covid_crash_demo(df, model, save_path="covid_crash_demo.png"):
    """Zoom into 2020 COVID crash and show model predictions."""
    crash_window = df.loc["2020-01-01":"2020-07-31"].copy()

    if len(crash_window) == 0:
        print("  COVID crash data not in dataset — skipping.")
        return

    X_window = crash_window[FEATURES]
    crash_window["pred_prob"] = model.predict_proba(X_window)[:, 1]

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    fig.suptitle("2020 COVID Crash — Did the Model See It Coming?",
                 fontsize=15, fontweight="bold", color="white", y=0.98)

    bottom_date = crash_window["Close"].idxmin()
    bottom_val  = crash_window["Close"].min()

    ax1 = axes[0]
    ax1.plot(crash_window.index, crash_window["Close"], color="#4A90FF", linewidth=1.8)
    ax1.axvline(bottom_date, color="#FF4444", linestyle="--", linewidth=1, alpha=0.7)
    ax1.annotate(f"Bottom\n{bottom_date.strftime('%d %b %Y')}\n{bottom_val:,.0f}",
                 xy=(bottom_date, bottom_val), xytext=(20, 30),
                 textcoords="offset points", color="#FF4444", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#FF4444"))
    ax1.set_ylabel("Nifty Close", color="white")
    ax1.set_title("Nifty 50 Price", color="white")
    ax1.grid(alpha=0.15, color="white")

    ax2 = axes[1]
    ax2.plot(crash_window.index, crash_window["pred_prob"], color="#FF6B6B", linewidth=1.8)
    ax2.axhline(THRESHOLD, linestyle="--", color="#FFA500", linewidth=1, label=f"Threshold ({THRESHOLD})")
    ax2.fill_between(crash_window.index, crash_window["pred_prob"], THRESHOLD,
                     where=(crash_window["pred_prob"] > THRESHOLD),
                     alpha=0.3, color="#FF4444", label="Model says: EXIT")
    ax2.axvline(bottom_date, color="#FF4444", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_ylabel("Crash Probability", color="white")
    ax2.set_title("Model Prediction — Did it warn us?", color="white")
    ax2.set_ylim(0, 1)
    ax2.legend(facecolor="#1a1a2e", labelcolor="white")
    ax2.grid(alpha=0.15, color="white")

    ax3 = axes[2]
    ax3.plot(crash_window.index, crash_window["vix"], color="#00D4AA", linewidth=1.5)
    ax3.axhline(20, linestyle=":", color="#FFA500", linewidth=1, label="VIX = 20 (danger zone)")
    ax3.axvline(bottom_date, color="#FF4444", linestyle="--", linewidth=1, alpha=0.7)
    vix_peak_date = crash_window["vix"].idxmax()
    vix_peak_val  = crash_window["vix"].max()
    ax3.annotate(f"VIX peak\n{vix_peak_val:.1f}",
                 xy=(vix_peak_date, vix_peak_val), xytext=(-60, -30),
                 textcoords="offset points", color="#00D4AA", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#00D4AA"))
    ax3.set_ylabel("India VIX", color="white")
    ax3.set_title("India VIX — Fear Index", color="white")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white")
    ax3.grid(alpha=0.15, color="white")

    ax4 = axes[3]
    ax4.fill_between(crash_window.index, crash_window["crash"],
                     alpha=0.8, color="#c0392b", label="Actual crash days (>5% drop in 5d)")
    ax4.axvline(bottom_date, color="#FF4444", linestyle="--", linewidth=1, alpha=0.7)
    ax4.set_ylabel("Crash label", color="white")
    ax4.set_title("Ground Truth — Days with >5% drop in next 5 trading days", color="white")
    ax4.set_ylim(0, 1.3)
    ax4.legend(facecolor="#1a1a2e", labelcolor="white")
    ax4.grid(alpha=0.15, color="white")

    price_drop = (crash_window["Close"].min() / crash_window["Close"].iloc[0] - 1) * 100
    max_prob   = crash_window["pred_prob"].max()
    crash_days = crash_window["crash"].sum()
    fig.text(0.5, 0.01,
             f"Peak crash probability: {max_prob*100:.1f}%   |   "
             f"Nifty peak-to-trough drop: {price_drop:.1f}%   |   "
             f"Crash days flagged: {crash_days}",
             ha="center", color="#aaaaaa", fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print(f"  COVID crash demo saved to {save_path}")


if __name__ == "__main__":
    print("\n[1/4] Loading data and model...")
    nifty = download_data(start="2015-01-01")
    nifty = engineer_features(nifty)

    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        print("  Loaded model.pkl")
    else:
        print("  model.pkl not found — training now...")
        model = train_model(nifty)

    print("\n[2/4] Running backtest...")
    results = run_backtest(nifty, model, threshold=THRESHOLD)

    print("\n[3/4] Performance metrics:")
    compute_metrics(results)

    print("\n[4/4] Generating charts...")
    plot_results(results)
    plot_covid_crash_demo(nifty, model)

    print("\nDone. Open backtest_results.png and covid_crash_demo.png.")