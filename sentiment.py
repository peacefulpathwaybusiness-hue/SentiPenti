#hello
# app_v2.py - Stock Sentiment Analysis MVP V2 using Streamlit and FinBERT
# To run: pip install streamlit requests beautifulsoup4 transformers torch pandas matplotlib
# Then: streamlit run app_v2.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Initialize FinBERT model and tokenizer
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_finbert()

# Function to scrape Yahoo Finance news headlines
def scrape_yahoo(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []
        for item in soup.find_all('h3', class_='Mb(5px)'):
            headline = item.text.strip()
            if headline:
                headlines.append(headline)
        return headlines
    except Exception as e:
        st.error(f"Error scraping Yahoo Finance: {e}")
        return []

# Function to scrape Reddit r/wallstreetbets posts mentioning the ticker
def scrape_reddit(ticker):
    url = f"https://old.reddit.com/r/wallstreetbets/search?q={ticker}&restrict_sr=1&sort=new"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        posts = []
        for post in soup.find_all('div', class_='thing'):
            title_elem = post.find('p', class_='title')
            if title_elem:
                title = title_elem.text.strip()
                if title:
                    posts.append(title)
        return posts
    except Exception as e:
        st.error(f"Error scraping Reddit: {e}")
        return []

# Function to analyze sentiment using FinBERT
def analyze_sentiment(texts):
    sentiments = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
        labels = ['positive', 'negative', 'neutral']
        sentiment = labels[probs.argmax()]
        score = probs[probs.argmax()]
        sentiments.append((text, sentiment, score))
    return sentiments

# Function to store raw text in SQLite
def store_raw_data(ticker, source, texts):
    conn = sqlite3.connect('sentiment.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS raw_data
                 (ticker TEXT, source TEXT, text TEXT, timestamp TEXT)''')
    timestamp = datetime.now().isoformat()
    for text in texts:
        c.execute("INSERT INTO raw_data VALUES (?, ?, ?, ?)", (ticker, source, text, timestamp))
    conn.commit()
    conn.close()

# Function to fetch historical data for trend analysis (last 24h)
def fetch_historical_data(ticker):
    conn = sqlite3.connect('sentiment.db')
    c = conn.cursor()
    one_day_ago = (datetime.now() - timedelta(hours=24)).isoformat()
    c.execute("SELECT source, text, timestamp FROM raw_data WHERE ticker = ? AND timestamp > ?", (ticker, one_day_ago))
    data = c.fetchall()
    conn.close()
    return data

# Function to aggregate sentiments
def aggregate_sentiments(sentiments):
    if not sentiments:
        return {'positive': 0, 'negative': 0, 'neutral': 0}, 0.0
    pos = sum(1 for _, s, _ in sentiments if s == 'positive')
    neg = sum(1 for _, s, _ in sentiments if s == 'negative')
    neu = sum(1 for _, s, _ in sentiments if s == 'neutral')
    total = len(sentiments)
    pos_pct = (pos / total) * 100 if total > 0 else 0
    neg_pct = (neg / total) * 100 if total > 0 else 0
    neu_pct = (neu / total) * 100 if total > 0 else 0
    avg_score = sum(s for _, _, s in sentiments) / total if total > 0 else 0
    return {'positive': pos_pct, 'negative': neg_pct, 'neutral': neu_pct}, avg_score

# Function to aggregate sentiments by source
def aggregate_by_source(sentiments, source_texts):
    source_agg = {}
    for source in ['Yahoo', 'Reddit']:
        source_sents = [s for s, t in zip(sentiments, source_texts) if t[1] == source]
        if source_sents:
            pos = sum(1 for _, s, _ in source_sents if s == 'positive')
            neg = sum(1 for _, s, _ in source_sents if s == 'negative')
            neu = sum(1 for _, s, _ in source_sents if s == 'neutral')
            total = len(source_sents)
            pos_pct = (pos / total) * 100 if total > 0 else 0
            neg_pct = (neg / total) * 100 if total > 0 else 0
            neu_pct = (neu / total) * 100 if total > 0 else 0
            source_agg[source] = {'positive': pos_pct, 'negative': neg_pct, 'neutral': neu_pct}
        else:
            source_agg[source] = {'positive': 0, 'negative': 0, 'neutral': 0}
    return source_agg

# Function to count ticker mentions
def count_mentions(ticker, texts):
    return len(texts)

# Function to generate trading signal
def get_trading_signal(pos_pct, neg_pct):
    if pos_pct > 70:
        return 'Bullish'
    elif neg_pct > 70:
        return 'Bearish'
    else:
        return 'Neutral'

# Streamlit app
st.title("Stock Sentiment Analysis MVP V2")

ticker = st.text_input("Enter stock ticker (e.g., TSLA):").upper()

if st.button("Analyze Sentiment"):
    if not ticker:
        st.error("Please enter a ticker symbol.")
    else:
        with st.spinner("Scraping data..."):
            yahoo_headlines = scrape_yahoo(ticker)
            reddit_posts = scrape_reddit(ticker)
        
        all_texts = [(text, 'Yahoo') for text in yahoo_headlines] + [(text, 'Reddit') for text in reddit_posts]
        
        if not all_texts:
            st.error("No data scraped. Try again or check sources.")
        else:
            # Store raw data
            store_raw_data(ticker, 'Yahoo', yahoo_headlines)
            store_raw_data(ticker, 'Reddit', reddit_posts)
            
            with st.spinner("Analyzing sentiment with FinBERT..."):
                sentiments = analyze_sentiment([text for text, _ in all_texts])
            
            # Aggregate sentiments
            agg, avg_score = aggregate_sentiments(sentiments)
            signal = get_trading_signal(agg['positive'], agg['negative'])
            
            # Sentiment by source
            source_agg = aggregate_by_source(sentiments, all_texts)
            
            # Ticker mention count
            mention_count = count_mentions(ticker, [text for text, _ in all_texts])
            
            # Display aggregated sentiment
            st.subheader("Aggregated Sentiment")
            df_agg = pd.DataFrame(list(agg.items()), columns=['Sentiment', 'Percentage'])
            st.table(df_agg)
            
            # Pie chart for overall sentiment
            fig, ax = plt.subplots()
            ax.pie(agg.values(), labels=agg.keys(), autopct='%1.1f%%', colors=['#4CAF50', '#F44336', '#B0BEC5'])
            ax.set_title('Overall Sentiment Distribution')
            st.pyplot(fig)
            
            # Bar chart for sentiment by source
            st.subheader("Sentiment by Source")
            sources = ['Yahoo', 'Reddit']
            pos_values = [source_agg[src]['positive'] for src in sources]
            neg_values = [source_agg[src]['negative'] for src in sources]
            neu_values = [source_agg[src]['neutral'] for src in sources]
            
            fig, ax = plt.subplots()
            x = range(len(sources))
            ax.bar([i - 0.3 for i in x], pos_values, width=0.3, label='Positive', color='#4CAF50')
            ax.bar(x, neg_values, width=0.3, label='Negative', color='#F44336')
            ax.bar([i + 0.3 for i in x], neu_values, width=0.3, label='Neutral', color='#B0BEC5')
            ax.set_xticks(x)
            ax.set_xticklabels(sources)
            ax.set_ylabel('Percentage')
            ax.set_title('Sentiment by Source')
            ax.legend()
            st.pyplot(fig)
            
            # Average score and mention count
            st.write(f"Average Sentiment Score: {avg_score:.2f}")
            st.write(f"Ticker Mentions: {mention_count}")
            
            # Trading signal
            st.subheader("Trading Signal")
            st.write(signal)
            
            # Trend analysis (basic: last 24h vs now)
            st.subheader("Sentiment Trend (Last 24h)")
            historical_data = fetch_historical_data(ticker)
            if historical_data:
                hist_df = pd.DataFrame(historical_data, columns=['source', 'text', 'timestamp'])
                hist_texts = hist_df['text'].tolist()
                hist_sentiments = analyze_sentiment(hist_texts)
                hist_agg, hist_avg_score = aggregate_sentiments(hist_sentiments)
                
                # Simple trend comparison
                st.write("Last 24h Sentiment:")
                df_hist_agg = pd.DataFrame(list(hist_agg.items()), columns=['Sentiment', 'Percentage'])
                st.table(df_hist_agg)
                st.write(f"Last 24h Average Sentiment Score: {hist_avg_score:.2f}")
                
                # Trend chart
                fig, ax = plt.subplots()
                times = ['Last 24h', 'Now']
                pos_trend = [hist_agg['positive'], agg['positive']]
                neg_trend = [hist_agg['negative'], agg['negative']]
                ax.plot(times, pos_trend, marker='o', label='Positive', color='#4CAF50')
                ax.plot(times, neg_trend, marker='o', label='Negative', color='#F44336')
                ax.set_ylabel('Percentage')
                ax.set_title('Sentiment Trend')
                ax.legend()
                st.pyplot(fig)
            else:
                st.write("No historical data available for trend analysis.")
            
            # Recent texts with sentiment tags
            st.subheader("Recent Texts with Sentiment")
            for (text, source), (text, sentiment, score) in zip(all_texts[:20], sentiments[:20]):
                st.write(f"[{source}] [{sentiment.upper()}] (Score: {score:.2f}): {text}")

# Notes:
# - For deployment: Run `streamlit run app_v2.py` locally or deploy to Streamlit Cloud using a requirements.txt.
# - For V3: Add multi-ticker support, volume-weighted sentiment, and search history (store queries in SQLite).
# - For V4: Implement a cron job with APScheduler for hourly scraping and live updates.te
