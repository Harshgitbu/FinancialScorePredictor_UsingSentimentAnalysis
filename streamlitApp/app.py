import streamlit as st
import pandas as pd
import joblib
from transformers import pipeline

# load assets
model = joblib.load("models/final_xgb_model.pkl")
le = joblib.load("models/label_encoder.pkl")

# Sentiment pipelines
twitter_sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
news_sentiment = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Header
st.set_page_config(page_title="Stock Action Predictor", layout="wide")
st.title("Stock Strength Predictor using Sentiment & Market Signals")
st.markdown("""
A full pipeline combining sentiment analysis (Twitter & News) and historical financial data to predict **Buy / Hold / Sell** signals for S&P 500 stocks.
---
""")

# overview
tabs = st.tabs(["Project Overview", "Sample Predictions", "Try It Yourself"])


with tabs[0]:
    st.header("Project Flow")
    st.markdown("""
    1. **Sentiment Ingestion**: Twitter + News analyzed with domain-specific BERT models
    2. **Stock Data Merge**: Historic S&P 500 prices, engineered technical indicators
    3. **Modeling**: XGBoost + Bayesian Optimization + Class Balancing (SMOTETomek)
    4. **Final Output**: Action label with confidence
    
    ### 🚀 Model Highlights
    - **Best Accuracy**: `71%` using optimized XGBoost
    - **Top Features**: Sentiment rolling avg, price, volatility, bid-ask spread
    
    ### 📈 Feature Importance
    """)
    st.image("visualization/Feature_Imp_final.png", use_column_width=True)
    st.markdown("---")
    st.markdown("### 📊 Performance Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.image("visualization/Baseline_model.png", caption="Baseline")
    with col2:
        st.image("visualization/Bayes_XGB.png", caption="Enhanced Model")

#  Sample Predictions 
with tabs[1]:
    st.header("Random Sample Predictions")
    df_sample = pd.read_csv("data/actual_vs_prediction.csv")
    show_cols = ["ticker", "price", "final_sentiment_score", "sentiment_3d_avg", "sentiment_7d_avg", "Actual_Action", "Predicted_Action"]
    st.dataframe(df_sample[show_cols].sample(10).reset_index(drop=True))

# DIY
with tabs[2]:
    st.header("Test Prediction on Your Text")
    
    user_input = st.text_input("Enter a news or tweet text:")
    selected_ticker = st.selectbox("Select a Ticker:", ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "META"])

    if user_input:
        # Get sentiment
        news_score = news_sentiment(user_input)[0]['score'] if 'label' in news_sentiment(user_input)[0] else 0.0
        twit_score = twitter_sentiment(user_input)[0]['score'] if 'label' in twitter_sentiment(user_input)[0] else 0.0
        final_score = 0.65 * news_score + 0.35 * twit_score

        # Simulate other values
        input_dict = {
            'price': 150.0,
            'volume': 10e6,
            'low_bid': 148.0,
            'high_ask': 152.0,
            'sp500_return': 0.01,
            'news_score': news_score,
            'twitter_score': twit_score,
            'final_sentiment_score': final_score,
            'sentiment_1d': final_score,
            'sentiment_3d_avg': final_score,
            'sentiment_7d_avg': final_score,
            'ret3': 0.03,
            'vol7': 0.02,
            'vma7': 9e6,
            'bidask_spread': 4.0
        }
        input_df = pd.DataFrame([input_dict])

        pred = model.predict(input_df)
        label = le.inverse_transform(pred)[0]

        st.success(f"Predicted Action for {selected_ticker}: **{label}**")
