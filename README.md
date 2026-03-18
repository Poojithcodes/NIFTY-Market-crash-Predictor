# Nifty 50 Market Crash Predictor

An end-to-end machine learning system that predicts the probability of a **>5% Nifty 50 crash within the next 5 trading days**, with a live Streamlit dashboard, strategy backtest, crash simulation, and real-time NLP news warning system.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Demo

![Backtest Results](backtest_results.png)

---

## Features

- **Live crash probability** — pulls real Nifty 50 + India VIX data from Yahoo Finance, refreshed hourly
- **VIX stress test** — slider to simulate fear spikes and see crash probability respond in real time
- **Strategy backtest** — compares model strategy vs buy-and-hold with Sharpe ratio and max drawdown
- **Crash simulation** — 30-day synthetic crash scenario fed into the real trained model day by day
- **NLP news warning** — VADER sentiment on live headlines across 8 global and domestic risk categories

---

## Quickstart
```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/nifty-crash-predictor.git
cd nifty-crash-predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python model.py

# 5. Run the backtest
python backtest.py

# 6. Launch the dashboard
streamlit run app.py
```

---

## News Warning Setup (optional)

The news warning tab requires a free NewsAPI key.

1. Sign up at [newsapi.org](https://newsapi.org)
2. Set your key as an environment variable:
```bash
set NEWSAPI_KEY=your_key_here       # Windows
export NEWSAPI_KEY=your_key_here    # Mac/Linux
```

If no key is set, the tab falls back to demo mode automatically.

---

## Project Structure
```
nifty-crash-predictor/
├── model.py          # Data download, feature engineering, XGBoost training
├── backtest.py       # Strategy backtest vs buy-and-hold
├── app.py            # Streamlit dashboard (5 tabs)
├── sentiment.py      # VADER sentiment scoring
├── news_warning.py   # NLP news warning across 8 risk categories
├── requirements.txt
└── README.md
```

---

## Model

| Item | Detail |
|---|---|
| Algorithm | XGBoost Classifier |
| Data | Nifty 50 (`^NSEI`) + India VIX (`^INDIAVIX`), 2015–today |
| Target | Binary — Nifty drops >5% in next 5 trading days |
| Features | 20 (returns, volatility, drawdown, trend, volume, VIX, sentiment) |
| Train/Test split | 80/20 time-based, no shuffle |
| Validation | 5-fold walk-forward TimeSeriesSplit |
| Class imbalance | `scale_pos_weight` |

### Feature Groups

| Group | Features |
|---|---|
| Returns | ret_1d, ret_5d, ret_10d, ret_20d |
| Volatility | volatility_5d, volatility_10d, volatility_20d |
| Drawdown | drawdown_5d, drawdown_20d |
| Trend | price_vs_ma50, price_vs_ma200, ma50_vs_ma200 |
| Volume | volume_ratio |
| VIX | vix, vix_change_5d, vix_spike |
| Composite | panic_signal |
| Sentiment | news_compound, news_negative, news_fear_flag |

---

## Results

> Run `python model.py` and `python backtest.py` to generate your own results.

| Metric | Value |
|---|---|
| ROC-AUC | *(run model.py)* |
| Walk-forward mean AUC | *(run model.py)* |
| Strategy total return | *(run backtest.py)* |
| Strategy Sharpe ratio | *(run backtest.py)* |
| Strategy max drawdown | *(run backtest.py)* |
| Buy & Hold max drawdown | *(run backtest.py)* |

---

## Team

Built for hackathon by:

- **Poojith Khanappur** — 1RV24CS189
- **Mohak Natarajan** — 1RV24CS155

RV College of Engineering, Computer Science

---

## Disclaimer

This project is for educational purposes only. It is not financial advice and should not be used to make real investment decisions.
