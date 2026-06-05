---
title: Sentiment & Quant Dashboard
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# 📊 Sentiment & Quant Dashboard

A full-stack financial analysis dashboard combining **NLP sentiment analysis** from social media & news with **quantitative technical analysis** of stock prices.

## Features

### 💬 Sentiment Analysis
- Fetches posts from **Reddit** (via PRAW) and **news headlines** (Economic Times scraper)
- Runs financial-domain sentiment model: `distilroberta-finetuned-financial-news-sentiment-analysis`
- Falls back to rule-based sentiment if model unavailable
- Donut charts per source, timeline chart, per-post table

### 📈 Stock Analysis
- **Candlestick chart** with SMA 20/50, Bollinger Bands
- **Volume bars** coloured by direction
- **RSI** (14-period) with overbought/oversold levels
- **MACD** with signal line and histogram
- **Annualised Volatility** (rolling 20-day)
- **Drawdown from peak**
- **OBV** (On-Balance Volume)
- Key stats: Total Return, Sharpe Ratio, Max Drawdown

### 🔗 Combined View
- Pearson correlation between sentiment score and daily price returns
- Overlay chart: price vs net sentiment bar
- Signal summary table (RSI, MACD, MA trend, Sentiment bias, Momentum)

## Setup

### No API keys needed
The app runs with mock data out of the box — great for demos.

### Reddit API (optional, for live data)
1. Go to https://www.reddit.com/prefs/apps and create a "script" app
2. Enter the credentials in the sidebar

### Stock data
Uses `yfinance` with automatic `.NS` (NSE) and `.BO` (BSE) suffix detection.

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```
