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

**Sentiment Data:**
1) Twitter: Analyzed using cardiffnlp/twitter-roberta-base-sentiment
2) News: Analyzed using yiyanghkust/finbert-tone

Each row contains: news_score, twitter_score, Weighted final score: 0.65 * news + 0.35 * twitter

**Stock Data:** Includes features like: price, volume, low_bid, high_ask, sp500_return
Engineered Features: ret3: 3-day cumulative return, vol7: 7-day volatility (std), vma7: 7-day average volume, bidask_spread: spread between ask and bid prices
sentiment_3d_avg, sentiment_7d_avg: rolling sentiment windows

**Modeling Approach**
1) Final Model: XGBoost (Bayesian Optimized) - Applied SMOTETomek to balance class imbalance, Tuned with Bayesian Search (skopt)

Performance after optimizing:
![Sample Predictions](visualization/sample_predictions_table.png)

Significantly improved over baseline (61%) and unbalanced recall values - ![Sample Predictions](visualization/sample_predictions_table.png)

Interpretability, Feature Importance tracked, Sentiment-based features ranked in top 5 - ![Sample Predictions](visualization/sample_predictions_table.png)


**Sample Predictions**

Preview of model predictions:

| Ticker | Price   | Sentiment Score | Sentiment 3D Avg | Sentiment 7D Avg | Actual Action | Predicted Action |
|--------|---------|------------------|------------------|------------------|----------------|-------------------|
| NFLX   | 685.67  | 0.277            | 0.008            | 0.092            | Buy            | Hold              |
| CME    | 203.28  | 0.505            | -0.067           | -0.265           | Hold           | Sell              |
| CRM    | 218.87  | 0.217            | 0.311            | 0.148            | Buy            | Buy               |
| GOOGL  | 101.22  | -0.314           | 0.029            | -0.105           | Hold           | Buy               |
| CUBE   | 41.81   | 0.650            | 0.078            | 0.144            | Hold           | Hold              |
| AMP    | 397.83  | -0.325           | -0.139           | -0.108           | Buy            | Hold              |
| BR     | 142.78  | 0.650            | 0.032            | 0.217            | Hold           | Hold              |
| AVGO   | 1348.00 | 0.199            | 0.214            | 0.283            | Buy            | Buy               |
| TSLA   | 725.60  | -0.048           | 0.013            | 0.036            | Buy            | Sell              |

See data/Actual_vs_Predictions.csv for 100 randomly sampled tickers with actual vs. predicted results.


**Visual Gallery**

![Sample Predictions](visualization/sample_predictions_table.png)
