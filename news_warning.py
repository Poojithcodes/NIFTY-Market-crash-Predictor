"""
news_warning.py — News-based Market Warning Indicator for Indian Investors.

Fetches real global and domestic headlines relevant to Indian markets,
scores them using VADER sentiment, and shows a traffic light warning.

No probability calculation. Just: is the news environment dangerous right now?

Setup:
    pip install newsapi-python vaderSentiment
    Set NEWSAPI_KEY environment variable or paste key directly below.

Run standalone:
    python news_warning.py
"""

import os
import datetime
import numpy as np

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    print("  Install newsapi: pip install newsapi-python")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("  Install vader: pip install vaderSentiment")


# ── Paste your NewsAPI key here if not using environment variable ─────────────
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")


# ── Search queries — all LEADING indicators, not lagging ─────────────────────
# These are things that cause crashes, not things reported after crashes.

SEARCH_QUERIES = {

    # ── GLOBAL LEADING INDICATORS ─────────────────────────────────────────────

    "Global Wars & Conflicts": [
        "Russia Ukraine war escalation",
        "Middle East conflict Israel Iran war",
        "Taiwan China military tension invasion",
        "NATO war troops deployment",
        "global conflict ceasefire breakdown",
    ],

    "Global Trade & Tariffs": [
        "US tariffs trade war China Europe",
        "WTO trade restrictions sanctions",
        "OPEC oil supply cut production",
        "global supply chain disruption",
        "semiconductor chip export ban",
    ],

    "Global Central Banks": [
        "US Federal Reserve rate hike pause",
        "ECB European Central Bank rate decision",
        "Bank of England rate hike recession",
        "global liquidity tightening quantitative",
        "dollar index DXY surge emerging markets",
    ],

    "Global Economic Stress": [
        "US recession GDP contraction unemployment",
        "China property crisis Evergrande default",
        "Europe energy crisis recession inflation",
        "emerging markets debt crisis default",
        "global banking crisis SVB collapse",
    ],

    # ── DOMESTIC LEADING INDICATORS ───────────────────────────────────────────

    "India Political & Social": [
        "India protests bandh strike shutdown",
        "India election results political uncertainty",
        "India government policy reform opposition",
        "India farmer protest agitation highway",
        "India communal tension unrest state",
    ],

    "India Macro & RBI": [
        "RBI repo rate hike cut inflation India",
        "India rupee depreciation dollar fall",
        "India GDP slowdown IIP data weak",
        "India current account deficit trade",
        "India inflation WPI CPI surge",
    ],

    "India Trade & Sanctions": [
        "India US tariffs export restrictions",
        "India China border trade ban",
        "India import duty customs shock",
        "India sanctions FATF compliance",
        "India export ban commodity wheat rice",
    ],

    "India Banking & FII": [
        "FII FPI selling India equity outflow",
        "India bank NPA fraud default RBI action",
        "India corporate debt restructuring crisis",
        "India stock market SEBI action fraud",
        "India mutual fund redemption pressure",
    ],
}

# ── Warning thresholds ────────────────────────────────────────────────────────
# VADER compound score: -1 (very negative) to +1 (very positive)

THRESHOLDS = {
    "RED":    -0.15,   # compound below this = HIGH RISK
    "YELLOW": -0.05,   # compound below this = ELEVATED RISK
    # above -0.05 = GREEN (safe)
}


def fetch_headlines(api_key, days_back=2):
    """
    Fetch headlines for all query categories.
    Returns dict of {category: [headlines]}
    """
    if not NEWSAPI_AVAILABLE:
        return {}

    client   = NewsApiClient(api_key=api_key)
    from_date = (datetime.date.today() - datetime.timedelta(days=days_back)).isoformat()
    results   = {}

    for category, queries in SEARCH_QUERIES.items():
        cat_headlines = []
        for query in queries:
            try:
                resp = client.get_everything(
                    q=query,
                    from_param=from_date,
                    language="en",
                    sort_by="relevancy",
                    page_size=5
                )
                if resp["status"] == "ok":
                    for article in resp["articles"]:
                        title = article.get("title", "")
                        desc  = article.get("description", "")
                        if title:
                            cat_headlines.append(title)
                        if desc:
                            cat_headlines.append(desc)
            except Exception as e:
                print(f"  Error fetching '{query}': {e}")

        results[category] = list(set(cat_headlines))

    return results


def score_category(headlines):
    """Score a list of headlines with VADER. Returns compound score."""
    if not VADER_AVAILABLE or not headlines:
        return 0.0

    analyzer = SentimentIntensityAnalyzer()
    scores   = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    return round(float(np.mean(scores)), 4)


def get_warning_level(compound):
    """Convert compound score to traffic light level."""
    if compound <= THRESHOLDS["RED"]:
        return "RED",    "HIGH RISK",      "🔴"
    elif compound <= THRESHOLDS["YELLOW"]:
        return "YELLOW", "ELEVATED RISK",  "🟡"
    else:
        return "GREEN",  "LOW RISK",       "🟢"


def run_warning_system(api_key=None):
    """
    Main function. Fetches news, scores each category,
    returns full warning report as a dict.
    """
    if api_key is None:
        api_key = NEWSAPI_KEY

    if not api_key:
        print("  No API key — returning demo data")
        return _demo_data()

    print("  Fetching headlines...")
    all_headlines = fetch_headlines(api_key)

    if not all_headlines:
        return _demo_data()

    # Score each category
    category_scores  = {}
    category_headlines = {}
    all_scores = []

    for category, headlines in all_headlines.items():
        score = score_category(headlines)
        category_scores[category]    = score
        category_headlines[category] = headlines[:5]  # keep top 5 per category
        all_scores.append(score)

    overall_compound = round(float(np.mean(all_scores)), 4) if all_scores else 0.0
    level, label, emoji = get_warning_level(overall_compound)

    return {
        "overall_compound":   overall_compound,
        "level":              level,
        "label":              label,
        "emoji":              emoji,
        "category_scores":    category_scores,
        "category_headlines": category_headlines,
        "fetched_at":         datetime.datetime.now().strftime("%d %b %Y %H:%M IST"),
    }


def _demo_data():
    """Fallback demo data when no API key is available."""
    return {
        "overall_compound":   -0.21,
        "level":              "RED",
        "label":              "HIGH RISK",
        "emoji":              "🔴",
        "category_scores": {
            "Global Wars & Conflicts":  -0.38,
            "Global Trade & Tariffs":   -0.29,
            "Global Central Banks":     -0.14,
            "Global Economic Stress":   -0.22,
            "India Political & Social": -0.11,
            "India Macro & RBI":        -0.18,
            "India Trade & Sanctions":  -0.24,
            "India Banking & FII":      -0.16,
        },
        "category_headlines": {
            "Global Wars & Conflicts":  ["Tensions escalate as Russia launches new offensive in Ukraine"],
            "Global Trade & Tariffs":   ["US announces sweeping new tariffs on Asian imports"],
            "Global Central Banks":     ["Fed signals rates to stay higher for longer amid sticky inflation"],
            "Global Economic Stress":   ["China property sector continues to drag on global growth outlook"],
            "India Political & Social": ["Nationwide strike called over fuel price hike affects transport"],
            "India Macro & RBI":        ["RBI warns of rupee pressure as dollar strengthens globally"],
            "India Trade & Sanctions":  ["India faces export restrictions on key pharmaceutical ingredients"],
            "India Banking & FII":      ["FII outflows from Indian equities hit three-month high"],
        },
        "fetched_at": "Demo Mode — set NEWSAPI_KEY to fetch real headlines",
    }


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Nifty 50 News Warning System ──\n")
    report = run_warning_system()

    print(f"  Overall Warning  : {report['emoji']}  {report['label']}")
    print(f"  Sentiment Score  : {report['overall_compound']:+.4f}")
    print(f"  Fetched at       : {report['fetched_at']}")
    print()

    print("  Category Breakdown:")
    for cat, score in report["category_scores"].items():
        _, lbl, em = get_warning_level(score)
        print(f"    {em}  {cat:<25} score: {score:+.4f}  ({lbl})")

    print()
    print("  Sample Headlines:")
    for cat, headlines in report["category_headlines"].items():
        if headlines:
            print(f"\n  [{cat}]")
            for h in headlines[:2]:
                print(f"    • {h[:90]}")