# FinancialScorePredictor_UsingSentimentAnalysis
A full-fledged machine learning pipeline to predict Buy / Hold / Sell signals for S&P 500 stocks using sentiment analysis on Twitter & News articles, historical stock performance, and engineered financial indicators. Stock Sentiment-Based Action Predictor

The data is engineered for S&P 500 stocks. The number of unique ticker symbols covered by the model is: 456

**Project Summary :**

This project builds a predictive model that: 
1) Ingests Twitter and News data to compute sentiment scores
2) Merges it with structured stock data and financial indicators
3) Applies lag-based sentiment aggregation
4) Trains a model to classify stock actions as Buy, Hold, or Sell

**Goal:** Provide users with a daily stock strength signal powered by sentiment fluctuations and market data.

**Data Sources & Engineering**

**Sentiment Data: **
1) Twitter: Analyzed using cardiffnlp/twitter-roberta-base-sentiment
2) News: Analyzed using yiyanghkust/finbert-tone

Each row contains: news_score, twitter_score, Weighted final score: 0.65 * news + 0.35 * twitter

**Stock Data:** Includes features like: price, volume, low_bid, high_ask, sp500_return
Engineered Features: ret3: 3-day cumulative return, vol7: 7-day volatility (std), vma7: 7-day average volume, bidask_spread: spread between ask and bid prices
sentiment_3d_avg, sentiment_7d_avg: rolling sentiment windows

**Modeling Approach**
1) Final Model: XGBoost (Bayesian Optimized) - Applied SMOTETomek to balance class imbalance, Tuned with Bayesian Search (skopt)

Performance after optimizing:
ss - from /visualization

Significantly improved over baseline (61%) and unbalanced recall values - ss from /visualization

Interpretability, Feature Importance tracked, Sentiment-based features ranked in top 5 - ss from /visualization


**Sample Predictions**

Preview of model predictions: fetch top 10 rows from /data/actual_vs_prediction.csv

Ticker  | Sentiment | Action
--------|-----------|--------
AAPL    |  0.65     | Buy
MSFT    | -0.32     | Sell
GOOG    |  0.01     | Hold

See data/Actual_vs_Predictions.csv for 100 randomly sampled tickers with actual vs. predicted results.


**Visual Gallery**
ss from /visualization

![Sample Predictions](visualization/sample_predictions_table.png)
