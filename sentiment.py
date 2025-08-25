# app_v2.py - Stock Sentiment Analysis MVP V2 using Streamlit + FinBERT + RSS (no paid APIs)

import streamlit as st
import requests
import feedparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from urllib.parse import quote_plus

st.set_page_config(page_title="Stock Sentiment Analysis MVP V2", layout="wide")

# ---------------------
# FinBERT (cached)
# ---------------------
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_finbert()

# ---------------------
# Helpers
# ---------------------
def _fetch_rss(url: str, timeout: int = 12):
    # Google News RSS is XML; use feedparser for robust parsing
    try:
        # small HEAD/GET to ensure outbound OK; feedparser will fetch itself too, but we short-circuit obvious failures
        _ = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
    except Exception:
        return []
    feed = feedparser.parse(url)
    titles = []
    for e in feed.entries[:50]:
        title = (e.title or "").strip()
        if title:
            titles.append(title)
    return titles

# ---------------------
# Scrapers (re-implemented as RSS queries)
# ---------------------
def scrape_yahoo(ticker):
    """
    Use Google News RSS restricted to Yahoo Finance headlines for this ticker.
    Works on Streamlit Cloud, no API key needed.
    """
    q = quote_plus(f"site:finance.yahoo.com {ticker}")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    titles = _fetch_rss(url)
    return titles

def scrape_reddit(ticker):
    """
    Use Google News RSS restricted to r/wallstreetbets mentions of the ticker.
    Avoids Reddit 403 without API keys.
    """
    q = quote_plus(f"site:reddit.com/r/wallstreetbets {ticker}")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    titles = _fetch_rss(url)
    return titles

# ---------------------
# Sentiment (FinBERT)
# ---------------------
def analyze_sentiment(texts):
    sentiments = []
    if not texts:
        return sentiments
    # Batch for speed
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        labels = ['positive', 'negative', 'neutral']  # FinBERT label order
        for text, p in zip(batch, probs):
            idx = p.argmax()
            sentiments.append((text, labels[idx], float(p[idx])))
    return sentiments

# ---------------------
# Storage
# ---------------------
def store_raw_data(ticker, source, texts):
    if not texts:
        return
    conn = sqlite3.connect('sentiment.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS raw_data
                 (ticker TEXT, source TEXT, text TEXT, timestamp TEXT)''')
    timestamp = datetime.now().isoformat(timespec="seconds")
    c.executemany("INSERT INTO raw_data VALUES (?, ?, ?, ?)", [(ticker, source, t, timestamp) for t in texts])
    conn.commit()
    conn.close()

def fetch_historical_data(ticker):
    conn = sqlite3.connect('sentiment.db')
    c = conn.cursor()
    one_day_ago = (datetime.now() - timedelta(hours=24)).isoformat(timespec="seconds")
    c.execute("SELECT source, text, timestamp FROM raw_data WHERE ticker = ? AND timestamp > ?", (ticker, one_day_ago))
    data = c.fetchall()
    conn.close()
    return data

# ---------------------
# Aggregations
# ---------------------
def aggregate_sentiments(sentiments):
    if not sentiments:
        return {'positive': 0, 'negative': 0, 'neutral': 0}, 0.0
    pos = sum(1 for _, s, _ in sentiments if s == 'positive')
    neg = sum(1 for _, s, _ in sentiments if s == 'negative')
    neu = sum(1 for _, s, _ in sentiments if s == 'neutral')
    total = len(sentiments)
    pos_pct = (pos / total) * 100
    neg_pct = (neg / total) * 100
    neu_pct = (neu / total) * 100
    avg_score = sum(sc for _, _, sc in sentiments) / total
    return {'positive': pos_pct, 'negative': neg_pct, 'neutral': neu_pct}, avg_score

def aggregate_by_source(sentiments, source_texts):
    # source_texts: list[(text, 'Yahoo'|'Reddit')]
    source_agg = {}
    for source in ['Yahoo', 'Reddit']:
        # filter sentiments by matching original source_texts
        source_indices = [i for i, (_, src) in enumerate(source_texts) if src == source]
        source_sents = [sentiments[i] for i in source_indices] if source_indices else []
        if source_sents:
            pos = sum(1 for _, s, _ in source_sents if s == 'positive')
            neg = sum(1 for _, s, _ in source_sents if s == 'negative')
            neu = sum(1 for _, s, _ in source_sents if s == 'neutral')
            total = len(source_sents)
            source_agg[source] = {
                'positive': (pos/total)*100,
                'negative': (neg/total)*100,
                'neutral':  (neu/total)*100
            }
        else:
            source_agg[source] = {'positive': 0, 'negative': 0, 'neutral': 0}
    return source_agg

def count_mentions(texts):
    return len(texts)

def get_trading_signal(pos_pct, neg_pct):
    if pos_pct > 70:
        return 'Bullish'
    elif neg_pct > 70:
        return 'Bearish'
    else:
        return 'Neutral'

# ---------------------
# UI
# ---------------------
st.title("Stock Sentiment Analysis MVP V2")

ticker = st.text_input("Enter stock ticker (e.g., TSLA):").upper().strip()

if st.button("Analyze Sentiment"):
    if not ticker:
        st.error("Please enter a ticker symbol.")
        st.stop()

    with st.spinner("Fetching news via RSS..."):
        yahoo_headlines = scrape_yahoo(ticker)
        reddit_posts = scrape_reddit(ticker)

    all_texts = [(t, 'Yahoo') for t in yahoo_headlines] + [(t, 'Reddit') for t in reddit_posts]

    if not all_texts:
        st.error("No data fetched from RSS sources. Try another ticker (e.g., AAPL, MSFT, NVDA).")
        st.stop()

    # Store raw
    store_raw_data(ticker, 'Yahoo', yahoo_headlines)
    store_raw_data(ticker, 'Reddit', reddit_posts)

    with st.spinner("Analyzing sentiment with FinBERT..."):
        sentiments = analyze_sentiment([t for t, _ in all_texts])

    agg, avg_score = aggregate_sentiments(sentiments)
    signal = get_trading_signal(agg['positive'], agg['negative'])
    source_agg = aggregate_by_source(sentiments, all_texts)
    mention_count = count_mentions(all_texts)

    # Layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Trading Signal")
        st.metric("Signal", signal)
        st.metric("Mentions", mention_count)
    with col2:
        st.subheader("Overall Sentiment")
        df_agg = pd.DataFrame(list(agg.items()), columns=['Sentiment', 'Percentage'])
        st.table(df_agg)
    with col3:
        st.subheader("Average Confidence")
        st.metric("Avg Score", f"{avg_score:.2f}")

    # Pie chart
    fig, ax = plt.subplots()
    ax.pie(agg.values(), labels=agg.keys(), autopct='%1.1f%%')
    ax.set_title('Overall Sentiment Distribution')
    st.pyplot(fig)

    # Bar by source
    st.subheader("Sentiment by Source")
    sources = ['Yahoo', 'Reddit']
    pos_values = [source_agg[s]['positive'] for s in sources]
    neg_values = [source_agg[s]['negative'] for s in sources]
    neu_values = [source_agg[s]['neutral']  for s in sources]

    fig2, ax2 = plt.subplots()
    x = range(len(sources))
    ax2.bar([i - 0.3 for i in x], pos_values, width=0.3, label='Positive')
    ax2.bar(x,                     neg_values, width=0.3, label='Negative')
    ax2.bar([i + 0.3 for i in x], neu_values, width=0.3, label='Neutral')
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(sources)
    ax2.set_ylabel('Percentage')
    ax2.set_title('By Source')
    ax2.legend()
    st.pyplot(fig2)

    # Trend (last 24h vs now)
    st.subheader("Sentiment Trend (Last 24h)")
    historical_data = fetch_historical_data(ticker)
    if historical_data:
        hist_df = pd.DataFrame(historical_data, columns=['source', 'text', 'timestamp'])
        hist_texts = hist_df['text'].tolist()
        hist_sentiments = analyze_sentiment(hist_texts)
        hist_agg, hist_avg_score = aggregate_sentiments(hist_sentiments)

        st.write("Last 24h Sentiment:")
        df_hist_agg = pd.DataFrame(list(hist_agg.items()), columns=['Sentiment', 'Percentage'])
        st.table(df_hist_agg)
        st.write(f"Last 24h Average Sentiment Score: {hist_avg_score:.2f}")

        fig3, ax3 = plt.subplots()
        times = ['Last 24h', 'Now']
        ax3.plot(times, [hist_agg['positive'], agg['positive']], marker='o', label='Positive')
        ax3.plot(times, [hist_agg['negative'], agg['negative']], marker='o', label='Negative')
        ax3.set_ylabel('Percentage')
        ax3.set_title('Sentiment Trend')
        ax3.legend()
        st.pyplot(fig3)
    else:
        st.write("No historical data available for trend analysis yet.")

    # Recent texts
    st.subheader("Recent Texts with Sentiment")
    for (text, source), (_, sentiment, score) in zip(all_texts[:30], sentiments[:30]):
        st.write(f"[{source}] [{sentiment.upper()}] (Score: {score:.2f}): {text}")
