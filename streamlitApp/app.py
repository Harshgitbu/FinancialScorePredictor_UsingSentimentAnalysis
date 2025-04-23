import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ----------- Load cached models and encoder -----------
@st.cache_resource
def load_model():
    return joblib.load("models/final_xgb_model.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load("models/label_encoder.pkl")

model = load_model()
le = load_encoder()

# ----------- UI Layout -----------
st.set_page_config(page_title="Stock Action Predictor", layout="wide")
st.title("📈 Stock Action Predictor (Buy / Hold / Sell)")
st.markdown("Predict stock strength using sentiment analysis and market data.")

# ----------- Load data -----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/Actual_vs_Predictions.csv")
    return df

sample_df = load_data()
tickers = sorted(sample_df['ticker'].unique())

# ----------- Section 1: Model Summary -----------
st.subheader("Model Overview")
st.markdown("""
- Final Model: **Bayesian-tuned XGBoost**
- Accuracy Achieved: **71%**
- Engineered features: 3-day return, volatility, 7-day avg volume, bid-ask spread
- Balanced classes using **SMOTETomek**
- Sentiment from **Twitter & News**, weighted score = 0.65 * News + 0.35 * Twitter
""")

# ----------- Section 2: View Predictions -----------
st.subheader("Sample Model Predictions")
view_count = st.slider("Select number of rows to preview:", 5, 50, 10)
st.dataframe(sample_df.head(view_count)[[
    'ticker', 'price', 'final_sentiment_score',
    'sentiment_3d_avg', 'sentiment_7d_avg',
    'Actual_Action', 'Predicted_Action'
]])

# ----------- Section 3: Interactive Prediction -----------
st.subheader("🔮 Try Your Own Prediction")
col1, col2 = st.columns(2)

with col1:
    ticker_input = st.selectbox("Choose a Ticker (S&P 500)", tickers)
    twitter_sentiment = st.selectbox("Twitter Sentiment", ["Positive", "Neutral", "Negative"])
    news_sentiment = st.selectbox("News Sentiment", ["Positive", "Neutral", "Negative"])

with col2:
    price = st.number_input("Current Price", value=100.0)
    volume = st.number_input("Volume", value=10000000)
    low_bid = st.number_input("Low Bid", value=98.0)
    high_ask = st.number_input("High Ask", value=102.0)
    sp500_return = st.slider("S&P 500 Daily Return", -0.05, 0.05, 0.01)

# ----------- Convert Sentiment to Score -----------
sentiment_map = {"Positive": 0.65, "Neutral": 0.0, "Negative": -0.65}
news_score = sentiment_map[news_sentiment]
twitter_score = sentiment_map[twitter_sentiment]
final_sentiment = 0.65 * news_score + 0.35 * twitter_score

# ----------- Feature Engineering (default values where missing) -----------
ret3 = 0.02
vol7 = 0.015
vma7 = volume  # assuming current volume as baseline
bidask_spread = high_ask - low_bid

# ----------- Create prediction row -----------
predict_row = pd.DataFrame.from_dict({
    'price': [price],
    'volume': [volume],
    'low_bid': [low_bid],
    'high_ask': [high_ask],
    'sp500_return': [sp500_return],
    'news_score': [news_score],
    'twitter_score': [twitter_score],
    'final_sentiment_score': [final_sentiment],
    'sentiment_1d': [final_sentiment],
    'sentiment_3d_avg': [final_sentiment],
    'sentiment_7d_avg': [final_sentiment],
    'ret3': [ret3],
    'vol7': [vol7],
    'vma7': [vma7],
    'bidask_spread': [bidask_spread]
})

# ----------- Run prediction -----------
if st.button("Predict Action"):
    pred = model.predict(predict_row)
    pred_label = le.inverse_transform(pred)[0]
    st.success(f"📊 Predicted Action for {ticker_input}: **{pred_label}**")

# ----------- Footer -----------
st.markdown("---")
st.caption("Built by Harsh Shah, Ishanay Sharma, Saketh Bolina| Powered by Sentiment + Market Signals")
