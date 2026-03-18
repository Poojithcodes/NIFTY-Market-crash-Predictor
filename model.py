"""
model.py — Data pipeline, feature engineering, and model training
for the Nifty 50 Crash Predictor.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import warnings
warnings.filterwarnings("ignore")


# ── 1. Data Download ──────────────────────────────────────────────────────────

def download_data(start="2015-01-01", end=None):
    """Download Nifty 50 + India VIX from Yahoo Finance."""
    print("  Downloading Nifty 50...")
    nifty = yf.download("^NSEI", start=start, end=end, progress=False)
    print("  Downloading India VIX...")
    vix   = yf.download("^INDIAVIX", start=start, end=end, progress=False)

    for df in [nifty, vix]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    nifty = nifty[["Close", "Volume"]].copy()
    vix   = vix[["Close"]].rename(columns={"Close": "vix"})

    nifty = nifty.join(vix, how="left")
    nifty["vix"] = nifty["vix"].ffill()
    return nifty


# ── 2. Feature Engineering ───────────────────────────────────────────────────

def engineer_features(df):
    """
    Build predictive features from raw OHLCV + VIX data.

    Feature groups
    --------------
    - Returns       : momentum over 1 / 5 / 10 / 20 days
    - Volatility    : rolling std of returns (5 / 10 / 20 days)
    - Drawdown      : distance from recent rolling highs (5 / 20 days)
    - Trend         : price vs 50-day and 200-day moving averages
    - Volume        : abnormal volume ratio (today vs 20-day avg)
    - VIX           : raw level + 5-day change + spike flag (>20)
    - Composite     : panic signal (VIX x volatility)
    - Sentiment     : NLP news scores (default 0, injected live in app.py)
    """
    d = df.copy()
    close  = d["Close"].squeeze()
    volume = d["Volume"].squeeze()

    # Returns
    for n in [1, 5, 10, 20]:
        d[f"ret_{n}d"] = close.pct_change(n)

    # Volatility
    for n in [5, 10, 20]:
        d[f"volatility_{n}d"] = d["ret_1d"].rolling(n).std()

    # Drawdown from rolling high
    for n in [5, 20]:
        rolling_max = close.rolling(n).max()
        d[f"drawdown_{n}d"] = (close - rolling_max) / rolling_max

    # Trend features
    d["ma_50"]  = close.rolling(50).mean()
    d["ma_200"] = close.rolling(200).mean()
    d["price_vs_ma50"]  = close / d["ma_50"]  - 1
    d["price_vs_ma200"] = close / d["ma_200"] - 1
    d["ma50_vs_ma200"]  = d["ma_50"] / d["ma_200"] - 1

    # Volume
    d["volume_ratio"] = volume / volume.rolling(20).mean()

    # VIX features
    d["vix_change_5d"] = d["vix"].pct_change(5)
    d["vix_spike"]     = (d["vix"] > 20).astype(int)

    # Composite
    d["panic_signal"] = d["vix"] * d["volatility_5d"]

    # Sentiment placeholders — neutral during training
    # Live values are injected by app.py at prediction time
    d["news_compound"]  = 0.0
    d["news_negative"]  = 0.0
    d["news_fear_flag"] = 0

    # Target: Nifty drops >5% in next 5 trading days
    future_ret = close.shift(-5) / close - 1
    d["crash"] = (future_ret < -0.05).astype(int)

    d.dropna(inplace=True)
    return d


FEATURES = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "drawdown_5d", "drawdown_20d",
    "price_vs_ma50", "price_vs_ma200", "ma50_vs_ma200",
    "volume_ratio",
    "vix", "vix_change_5d", "vix_spike",
    "panic_signal",
    "news_compound", "news_negative", "news_fear_flag",
]


# ── 3. Walk-Forward Cross-Validation ─────────────────────────────────────────

def walk_forward_cv(df, n_splits=5):
    """
    TimeSeriesSplit CV — no future data ever leaks into training.
    Returns list of per-fold AUC scores.
    """
    X = df[FEATURES]
    y = df["crash"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []

    print(f"  Walk-forward CV ({n_splits} folds):")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        m = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            scale_pos_weight=spw, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        aucs.append(auc)
        print(f"    Fold {fold}: AUC = {auc:.4f}")

    print(f"  Mean AUC = {np.mean(aucs):.4f}  +/-  {np.std(aucs):.4f}\n")
    return aucs


# ── 4. Final Model Training ───────────────────────────────────────────────────

def train_model(df):
    """
    Train on first 80% of data, evaluate on last 20%.
    Saves model.pkl for Streamlit app.
    """
    X = df[FEATURES]
    y = df["crash"]

    split   = int(len(X) * 0.8)
    X_train = X.iloc[:split];  X_test = X.iloc[split:]
    y_train = y.iloc[:split];  y_test = y.iloc[split:]

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("  Held-out test set (last 20% of data):")
    print(classification_report(y_test, y_pred, target_names=["No Crash", "Crash"]))
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    joblib.dump(model, "model.pkl")
    print("  Model saved to model.pkl\n")
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[1/4] Downloading data...")
    nifty = download_data()

    print("\n[2/4] Engineering features...")
    nifty = engineer_features(nifty)
    print(f"  Dataset: {len(nifty)} rows | Crash rate: {nifty['crash'].mean()*100:.1f}%")

    print("\n[3/4] Walk-forward cross-validation...")
    walk_forward_cv(nifty)

    print("[4/4] Training final model on 80% of data...")
    train_model(nifty)
    print("Done.")