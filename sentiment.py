"""
sentiment.py — Real-time NLP news sentiment for the Nifty 50 Crash Predictor.

Fetches latest Indian market headlines from NewsAPI and scores them
using VADER sentiment analysis. Returns a daily sentiment score
that gets fed into the crash prediction model.

Setup:
    1. Get a free API key from https://newsapi.org
    2. Set it as environment variable: set NEWSAPI_KEY=your_key_here
    3. Or pass it directly to get_sentiment_score()

Run standalone:
    python sentiment.py
"""

import os
import datetime
import numpy as np

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


# ── Keywords to search for ────────────────────────────────────────────────────
SEARCH_QUERIES = [
    "Nifty 50 stock market India",
    "NSE Sensex crash India",
    "Indian stock market fall",
    "RBI interest rate India",
    "FII selling India market",
]


def fetch_headlines(api_key, days_back=3):
    """
    Fetch recent Indian market headlines from NewsAPI.
    Returns list of headline strings.
    """
    if not NEWSAPI_AVAILABLE:
        print("  newsapi-python not installed. Run: pip install newsapi-python")
        return []

    try:
        client = NewsApiClient(api_key=api_key)
        from_date = (datetime.date.today() - datetime.timedelta(days=days_back)).isoformat()

        headlines = []
        for query in SEARCH_QUERIES:
            response = client.get_everything(
                q=query,
                from_param=from_date,
                language="en",
                sort_by="relevancy",
                page_size=10
            )
            if response["status"] == "ok":
                for article in response["articles"]:
                    title = article.get("title", "")
                    desc  = article.get("description", "")
                    if title:
                        headlines.append(title)
                    if desc:
                        headlines.append(desc)

        # Deduplicate
        headlines = list(set(headlines))
        print(f"  Fetched {len(headlines)} headlines from NewsAPI")
        return headlines

    except Exception as e:
        print(f"  NewsAPI error: {e}")
        return []


def score_headlines(headlines):
    """
    Score a list of headlines using VADER sentiment.

    Returns dict with:
        compound   : overall sentiment (-1 = very negative, +1 = very positive)
        negative   : fraction of negative sentiment
        positive   : fraction of positive sentiment
        n_headlines: number of headlines scored
    """
    if not VADER_AVAILABLE:
        print("  vaderSentiment not installed. Run: pip install vaderSentiment")
        return _neutral_scores()

    if not headlines:
        print("  No headlines to score — returning neutral")
        return _neutral_scores()

    analyzer  = SentimentIntensityAnalyzer()
    scores    = [analyzer.polarity_scores(h) for h in headlines]

    compound  = np.mean([s["compound"] for s in scores])
    negative  = np.mean([s["neg"]      for s in scores])
    positive  = np.mean([s["pos"]      for s in scores])

    return {
        "compound":    round(compound, 4),
        "negative":    round(negative, 4),
        "positive":    round(positive, 4),
        "n_headlines": len(headlines),
    }


def _neutral_scores():
    return {"compound": 0.0, "negative": 0.0, "positive": 0.0, "n_headlines": 0}


def get_sentiment_score(api_key=None, days_back=3):
    """
    Main function — fetch headlines and return sentiment scores.

    If api_key is None, tries to read from NEWSAPI_KEY environment variable.
    If that also fails, returns neutral scores (graceful fallback).
    """
    if api_key is None:
        api_key = os.environ.get("NEWSAPI_KEY", None)

    if api_key is None:
        print("  No NewsAPI key found — sentiment defaulting to neutral")
        print("  Set your key: set NEWSAPI_KEY=your_key_here  (Windows)")
        print("                export NEWSAPI_KEY=your_key_here  (Mac/Linux)")
        return _neutral_scores()

    headlines = fetch_headlines(api_key, days_back=days_back)
    scores    = score_headlines(headlines)
    return scores


def sentiment_to_features(scores):
    """
    Convert raw sentiment scores into model-ready features.

    Features added:
        news_compound   : overall tone (-1 to +1), negative = bearish
        news_negative   : fraction of negative language (fear, crash, fall)
        news_fear_flag  : 1 if compound < -0.2 (market fear detected in news)
    """
    return {
        "news_compound":  scores["compound"],
        "news_negative":  scores["negative"],
        "news_fear_flag": 1 if scores["compound"] < -0.2 else 0,
    }


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Nifty 50 News Sentiment Analysis ──\n")

    scores = get_sentiment_score()

    print(f"\n  Headlines scored  : {scores['n_headlines']}")
    print(f"  Compound score    : {scores['compound']:+.4f}  (-1=very bearish, +1=very bullish)")
    print(f"  Negative fraction : {scores['negative']:.4f}")
    print(f"  Positive fraction : {scores['positive']:.4f}")

    features = sentiment_to_features(scores)
    print(f"\n  Model features:")
    for k, v in features.items():
        print(f"    {k}: {v}")

    if scores["compound"] < -0.2:
        print("\n  ⚠️  News sentiment is BEARISH — market fear detected in headlines")
    elif scores["compound"] > 0.2:
        print("\n  ✅  News sentiment is BULLISH — positive market narrative")
    else:
        print("\n  ➖  News sentiment is NEUTRAL")