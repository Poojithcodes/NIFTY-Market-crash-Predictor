"""
app.py — Streamlit dashboard for the Nifty 50 Crash Predictor.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from model import download_data, engineer_features, train_model, FEATURES, walk_forward_cv
from backtest import run_backtest, THRESHOLD

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nifty 50 Crash Predictor",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Nifty 50 — Market Crash Predictor")
st.caption(
    "XGBoost model trained on 2015 to today. "
    "Predicts the probability of a >5% Nifty 50 drop within the next 5 trading days."
)

# ── Data & Model loading ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading market data...", ttl=3600)
def load_data():
    nifty = download_data()
    nifty = engineer_features(nifty)
    return nifty

@st.cache_resource(show_spinner="Loading model...")
def load_model(nifty):
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return train_model(nifty)

nifty = load_data()
model = load_model(nifty)

X = nifty[FEATURES]
nifty = nifty.copy()
nifty["pred_prob"] = model.predict_proba(X)[:, 1]

# ── Live sentiment injection ──────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching news sentiment...", ttl=3600)
def load_sentiment():
    try:
        from sentiment import get_sentiment_score, sentiment_to_features
        scores = get_sentiment_score()
        return sentiment_to_features(scores)
    except Exception:
        return {"news_compound": 0.0, "news_negative": 0.0, "news_fear_flag": 0}

sent = load_sentiment()

# ── Live signal ───────────────────────────────────────────────────────────────

latest = X.iloc[[-1]].copy()
latest["news_compound"]  = sent["news_compound"]
latest["news_negative"]  = sent["news_negative"]
latest["news_fear_flag"] = sent["news_fear_flag"]

current_prob = float(model.predict_proba(latest)[0][1])
crash_flag   = current_prob >= THRESHOLD

# ── Top metrics ───────────────────────────────────────────────────────────────

st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Crash Probability", f"{current_prob*100:.1f}%")
col2.metric("Signal", "CRASH RISK" if crash_flag else "STABLE",
            delta="EXIT" if crash_flag else "INVESTED",
            delta_color="inverse")
col3.metric("Latest Nifty Close", f"{float(nifty['Close'].iloc[-1]):,.0f}")
col4.metric("India VIX", f"{float(nifty['vix'].iloc[-1]):.1f}")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Dashboard",
    "Backtest",
    "Model Info",
    "🔴 Crash Demo",
    "📰 News Warning",
])

# ── Tab 1: Live Dashboard ─────────────────────────────────────────────────────
with tab1:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Nifty 50 — Close Price")
        st.line_chart(nifty["Close"])

    with col_r:
        st.subheader("Crash Probability Over Time")
        st.line_chart(nifty[["pred_prob"]].rename(columns={"pred_prob": "Crash Probability"}))

    st.subheader("Top Risk Drivers (Feature Importance)")
    importance = (
        pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    st.bar_chart(importance.set_index("feature"))

    st.divider()
    st.subheader("What-if Simulation — Adjust VIX")
    st.caption("Drag the slider to see how a VIX spike changes crash probability.")

    vix_change = st.slider("Increase VIX by", 0, 25, 0, step=1)
    sim = latest.copy()
    sim["vix"]          = sim["vix"] + vix_change
    sim["vix_spike"]    = int(float(sim["vix"].iloc[0]) > 20)
    sim["panic_signal"] = sim["vix"] * sim["volatility_5d"]
    new_prob = float(model.predict_proba(sim)[0][1])

    c1, c2 = st.columns(2)
    c1.metric("Baseline probability", f"{current_prob*100:.1f}%")
    c2.metric("Adjusted probability", f"{new_prob*100:.1f}%",
              delta=f"{(new_prob - current_prob)*100:+.1f}%",
              delta_color="inverse")

# ── Tab 2: Backtest ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("Strategy vs Buy & Hold — Last 20% of Data (Test Period)")
    st.caption(
        f"The model exits to cash whenever crash probability > {THRESHOLD:.0%}. "
        "All results are on out-of-sample data only."
    )
    st.info("🚧 Backtest results ")

    results = run_backtest(nifty, model, threshold=THRESHOLD)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Cumulative Returns")
        st.line_chart(results[["bah_growth", "strategy_growth"]].rename(
            columns={"bah_growth": "Buy & Hold", "strategy_growth": "Model Strategy"}
        ))
    with col_r:
        st.subheader("Crash Probability + Signal")
        st.line_chart(results[["pred_prob"]].rename(columns={"pred_prob": "Crash Probability"}))

    st.subheader("Performance Metrics")
    trading_days = 252
    rows = []
    for name, col in [("Buy & Hold", "daily_ret"), ("Model Strategy", "strategy_ret")]:
        rets   = results[col].dropna()
        total  = (1 + rets).prod() - 1
        ann    = (1 + total) ** (trading_days / len(rets)) - 1
        vol    = rets.std() * np.sqrt(trading_days)
        sharpe = ann / vol if vol > 0 else 0
        dd     = ((1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1).min()
        rows.append({
            "Strategy":       name,
            "Total Return":   f"{total*100:+.1f}%",
            "Ann. Return":    f"{ann*100:+.1f}%",
            "Ann. Volatility":f"{vol*100:.1f}%",
            "Sharpe Ratio":   f"{sharpe:.2f}",
            "Max Drawdown":   f"{dd*100:.1f}%",
        })
    st.table(pd.DataFrame(rows).set_index("Strategy"))

# ── Tab 3: Model Info ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Details")
    st.markdown("""
| Item | Detail |
|---|---|
| Algorithm | XGBoost Classifier |
| Data | Nifty 50 (`^NSEI`) + India VIX (`^INDIAVIX`), 2015 to today |
| Target | Binary — will Nifty drop >5% in next 5 trading days? |
| Train / Test split | 80% / 20% (time-based, no shuffle) |
| Imbalance handling | `scale_pos_weight` |
| Validation | 5-fold walk-forward TimeSeriesSplit |
| Features | 20 (returns, volatility, drawdown, trend, volume, VIX, sentiment) |
    """)

    st.subheader("Walk-Forward CV Results")
    st.caption("Run `python model.py` to regenerate these in your terminal.")

    st.subheader("Feature List")
    st.code("\n".join(FEATURES))

# ── Tab 4: Crash Demo ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("🔴 Market Crash Simulation")
    st.caption("Synthetic 30-day crash scenario fed into the real trained model. No internet needed.")

    scenario_data = {
        "Day":   list(range(1, 31)),
        "Phase": ["Normal"]*10 + ["CRASH"]*10 + ["Recovery"]*10,
        "Nifty_Close": [
            22100, 22250, 22180, 22300, 22420, 22380, 22500, 22450, 22600, 22550,
            22200, 21600, 20800, 19900, 19100, 18400, 17800, 17200, 16800, 16500,
            16900, 17300, 17600, 17900, 18200, 18600, 19000, 19400, 19700, 20000,
        ],
        "VIX": [
            13.2, 13.5, 13.1, 13.8, 12.9, 13.4, 12.8, 13.6, 13.0, 13.3,
            18.5, 26.4, 35.2, 44.8, 52.3, 58.7, 62.1, 65.4, 61.2, 57.8,
            50.2, 44.1, 38.6, 33.2, 28.7, 25.1, 22.4, 19.8, 17.5, 15.9,
        ],
        "ret_1d": [
             0.007,  0.007, -0.003,  0.005,  0.005, -0.002,  0.005, -0.002,  0.007, -0.002,
            -0.016, -0.027, -0.037, -0.043, -0.040, -0.037, -0.033, -0.034, -0.023, -0.018,
             0.024,  0.024,  0.017,  0.017,  0.017,  0.022,  0.022,  0.021,  0.015,  0.015,
        ],
        "volatility_5d": [
            0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
            0.010, 0.018, 0.028, 0.038, 0.042, 0.044, 0.043, 0.042, 0.039, 0.036,
            0.030, 0.026, 0.022, 0.019, 0.017, 0.015, 0.014, 0.013, 0.012, 0.011,
        ],
        "drawdown_5d": [
             0.000,  0.000, -0.003,  0.000,  0.000, -0.002,  0.000, -0.002,  0.000, -0.002,
            -0.016, -0.038, -0.072, -0.108, -0.140, -0.168, -0.191, -0.211, -0.224, -0.233,
            -0.220, -0.205, -0.190, -0.175, -0.160, -0.143, -0.126, -0.110, -0.096, -0.082,
        ],
    }

    demo_df = pd.DataFrame(scenario_data)

    demo_rows = []
    for _, row in demo_df.iterrows():
        f = {feat: 0.0 for feat in FEATURES}
        f["vix"]             = row["VIX"]
        f["vix_spike"]       = 1 if row["VIX"] > 20 else 0
        f["vix_change_5d"]   = (row["VIX"] - 13.0) / 13.0
        f["ret_1d"]          = row["ret_1d"]
        f["ret_5d"]          = row["ret_1d"] * 4
        f["ret_10d"]         = row["ret_1d"] * 7
        f["ret_20d"]         = row["ret_1d"] * 12
        f["volatility_5d"]   = row["volatility_5d"]
        f["volatility_10d"]  = row["volatility_5d"] * 1.1
        f["volatility_20d"]  = row["volatility_5d"] * 1.2
        f["drawdown_5d"]     = row["drawdown_5d"]
        f["drawdown_20d"]    = row["drawdown_5d"] * 1.5
        f["panic_signal"]    = row["VIX"] * row["volatility_5d"]
        f["price_vs_ma50"]   = row["drawdown_5d"] * 0.5
        f["price_vs_ma200"]  = row["drawdown_5d"] * 0.3
        f["ma50_vs_ma200"]   = row["drawdown_5d"] * 0.1
        f["volume_ratio"]    = 1.0 + abs(row["ret_1d"]) * 20
        f["news_compound"]   = 0.0
        f["news_negative"]   = 0.0
        f["news_fear_flag"]  = 0
        demo_rows.append(f)

    X_demo = pd.DataFrame(demo_rows)
    demo_df["crash_prob"] = model.predict_proba(X_demo)[:, 1]

    st.markdown("### Drag the slider to replay the crash day by day")
    day     = st.slider("Day", min_value=1, max_value=30, value=1, step=1)
    current = demo_df[demo_df["Day"] == day].iloc[0]
    phase   = current["Phase"]

    if phase == "Normal":
        st.success(f"📅 Day {day} — Market is NORMAL. No signs of stress.")
    elif phase == "CRASH":
        st.error(f"💥 Day {day} — CRASH IN PROGRESS. Model is on HIGH ALERT.")
    else:
        st.warning(f"📈 Day {day} — Recovery phase. Risk slowly decreasing.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nifty Close",       f"{current['Nifty_Close']:,.0f}",
              delta=f"{current['ret_1d']*100:+.1f}%")
    c2.metric("India VIX",         f"{current['VIX']:.1f}")
    c3.metric("5d Volatility",     f"{current['volatility_5d']*100:.2f}%")
    c4.metric("Crash Probability", f"{current['crash_prob']*100:.1f}%",
              delta="EXIT" if current["crash_prob"] >= 0.40 else "HOLD",
              delta_color="inverse" if current["crash_prob"] >= 0.40 else "off")

    so_far = demo_df[demo_df["Day"] <= day]
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Nifty 50 Price**")
        st.line_chart(so_far.set_index("Day")["Nifty_Close"])
    with col_r:
        st.markdown("**Model Crash Probability**")
        st.line_chart(so_far.set_index("Day")["crash_prob"])

    prob = current["crash_prob"]
    if prob >= 0.40:
        st.error(f"🚨 CRASH PROBABILITY: {prob*100:.1f}% — Model says EXIT TO CASH immediately.")
    elif prob >= 0.25:
        st.warning(f"⚠️ ELEVATED RISK: {prob*100:.1f}% — Monitor closely.")
    else:
        st.success(f"✅ LOW RISK: {prob*100:.1f}% — Market looks stable.")

    with st.expander("🔍 See exact feature values being fed into the model"):
        st.dataframe(
            X_demo.iloc[day-1:day].T.rename(columns={day-1: "Value"}).style.format("{:.4f}")
        )

    st.divider()
    st.markdown("### Full Scenario Summary")
    peak      = demo_df["Nifty_Close"].max()
    trough    = demo_df["Nifty_Close"].min()
    drop      = (trough / peak - 1) * 100
    max_p     = demo_df["crash_prob"].max()
    alert_day = demo_df[demo_df["crash_prob"] >= 0.40]["Day"].min()

    s1, s2, s3 = st.columns(3)
    s1.metric("Peak to Trough Drop",      f"{drop:.1f}%")
    s2.metric("Max Crash Probability",    f"{max_p*100:.1f}%")
    s3.metric("Model First Alert on Day", f"Day {int(alert_day)}" if not pd.isna(alert_day) else "No alert")

    st.dataframe(
        demo_df[["Day", "Phase", "Nifty_Close", "VIX", "ret_1d", "volatility_5d", "crash_prob"]]
        .rename(columns={"ret_1d": "1d Return", "volatility_5d": "5d Vol", "crash_prob": "Crash Prob"})
        .set_index("Day")
        .style.format({
            "Nifty_Close": "{:,.0f}",
            "VIX":         "{:.1f}",
            "1d Return":   "{:.2%}",
            "5d Vol":      "{:.3f}",
            "Crash Prob":  "{:.1%}",
        }),
        use_container_width=True
    )

# ── Tab 5: News Warning ───────────────────────────────────────────────────────
with tab5:
    from news_warning import run_warning_system, get_warning_level

    st.subheader("📰 Global & Domestic News Warning Indicator")
    st.caption(
        "Scans real headlines across 8 risk categories using NLP sentiment analysis. "
        "Leading indicators — news that causes crashes, not news about crashes."
    )

    @st.cache_data(show_spinner="Scanning global headlines...", ttl=3600)
    def load_warning():
        return run_warning_system()

    if st.button("🔄 Refresh News Now"):
        st.cache_data.clear()

    report = load_warning()
    level  = report["level"]
    label  = report["label"]
    emoji  = report["emoji"]
    score  = report["overall_compound"]

    if level == "RED":
        st.error(f"{emoji}  Overall Market Warning: **{label}**  |  Sentiment Score: {score:+.3f}  |  {report['fetched_at']}")
    elif level == "YELLOW":
        st.warning(f"{emoji}  Overall Market Warning: **{label}**  |  Sentiment Score: {score:+.3f}  |  {report['fetched_at']}")
    else:
        st.success(f"{emoji}  Overall Market Warning: **{label}**  |  Sentiment Score: {score:+.3f}  |  {report['fetched_at']}")

    st.divider()
    st.subheader("Risk Category Breakdown")
    st.caption("Each category scored independently. Red = bearish news detected.")

    cats      = list(report["category_scores"].keys())
    col_pairs = [st.columns(2) for _ in range(4)]
    cols      = [c for pair in col_pairs for c in pair]

    for i, cat in enumerate(cats):
        cat_score = report["category_scores"][cat]
        _, cat_label, cat_emoji = get_warning_level(cat_score)
        if cat_score <= -0.15:
            cols[i].error(f"{cat_emoji} **{cat}**\n\nScore: `{cat_score:+.3f}` — {cat_label}")
        elif cat_score <= -0.05:
            cols[i].warning(f"{cat_emoji} **{cat}**\n\nScore: `{cat_score:+.3f}` — {cat_label}")
        else:
            cols[i].success(f"{cat_emoji} **{cat}**\n\nScore: `{cat_score:+.3f}` — {cat_label}")

    st.divider()
    st.subheader("Latest Headlines Driving the Warning")
    st.caption("Actual headlines being scored. Negative headlines push warning toward red.")

    for cat, headlines in report["category_headlines"].items():
        cat_score = report["category_scores"][cat]
        _, _, cat_emoji = get_warning_level(cat_score)
        with st.expander(f"{cat_emoji}  {cat}  —  score: {cat_score:+.3f}"):
            if headlines:
                for h in headlines[:5]:
                    st.markdown(f"- {h[:120]}")
            else:
                st.caption("No headlines found for this category.")

    st.divider()
    st.subheader("What This Means for You")
    if level == "RED":
        st.markdown("""
**🔴 High Risk Environment Detected**
- Global news environment is significantly negative for Indian markets
- Consider reducing equity exposure or moving to defensive positions
- Watch for FII outflows and VIX spikes in the coming days
- This is a *warning signal*, not a guarantee — always do your own research
        """)
    elif level == "YELLOW":
        st.markdown("""
**🟡 Elevated Risk — Watch Closely**
- Some concerning signals in the news environment
- Monitor the red categories closely over the next 1-2 days
- Avoid adding new positions until the picture clears
- This is a *caution signal*, not a sell signal
        """)
    else:
        st.markdown("""
**🟢 Low Risk Environment**
- Global news environment looks broadly neutral to positive
- No major geopolitical, trade, or macro stress signals detected
- Normal market conditions — standard risk management applies
- Refresh in a few hours as news can change quickly
        """)