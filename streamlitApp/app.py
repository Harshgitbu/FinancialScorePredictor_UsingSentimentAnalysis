import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from transformers import pipeline

# ------------------ Cache assets ------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/final_xgb_model.pkl")
    return model

@st.cache_resource
def load_encoder():
    return joblib.load("models/label_encoder.pkl")

# ------------------ Load assets ------------------
model = load_model()
le = load_encoder()

# Safe-load XGB booster (for advanced cases)
# booster = xgb.Booster()
# booster.load_model("models/final_xgb_model.json")

# ------------------ Sentiment Pipelines ------------------
twitter_sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
news_sentiment = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# ------------------ App Layout ------------------
st.set_page_config(page_title="Stock Score Predictor", layout="centered")
st.title("📈 Stock Sentiment & Action Predictor")
st.markdown("Get Buy / Hold / Sell signals powered by sentiment + stock indicators.")

# ------------------ Input Fields ------------------
ticker = st.selectbox("Select a Ticker (Example: ORCL, TSLA, AAPL)", ["AAPL", "TSLA", "GOOGL", "MSFT", "ORCL"])
tweet = st.text_area("Enter a Tweet or Headline")

if st.button("Predict Action"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet or headline text.")
    else:
        # ------------------ Sentiment Scoring ------------------
        tw_pred = twitter_sentiment(tweet)[0]
        nw_pred = news_sentiment(tweet)[0]

        tw_score = tw_pred['score'] if 'POS' in tw_pred['label'].upper() else (-tw_pred['score'] if 'NEG' in tw_pred['label'].upper() else 0)
        nw_score = nw_pred['score'] if 'POS' in nw_pred['label'].upper() else (-nw_pred['score'] if 'NEG' in nw_pred['label'].upper() else 0)

        final_sentiment = 0.65 * nw_score + 0.35 * tw_score

        # ------------------ Default Financial Feature Values ------------------
        avg_features = {
            'price': 150.0, 'volume': 2e7, 'low_bid': 148.0, 'high_ask': 151.5, 'sp500_return': 0.001,
            'news_score': nw_score, 'twitter_score': tw_score, 'final_sentiment_score': final_sentiment,
            'sentiment_1d': final_sentiment, 'sentiment_3d_avg': final_sentiment,
            'sentiment_7d_avg': final_sentiment, 'ret3': 0.015, 'vol7': 0.02,
            'vma7': 2e7, 'bidask_spread': 3.5
        }

        input_df = pd.DataFrame([avg_features])

        # ------------------ Prediction ------------------
        pred = model.predict(input_df)[0]
        pred_label = le.inverse_transform([pred])[0]

        # ------------------ Display Output ------------------
        st.success(f"Predicted Action for {ticker}: **{pred_label}**")
        st.markdown(f"\n\n**🧠 Sentiment Scores:**\n- News: `{nw_score:.2f}`\n- Twitter: `{tw_score:.2f}`\n- Final Weighted: `{final_sentiment:.2f}`")

        st.markdown("---")
        st.markdown("_Note: Default financial metrics used. Sentiment drives prediction._")

# ------------------ Footer ------------------
st.markdown("---")
st.caption("Built by Harsh Shah, Ishanay Sharma, Saketh Bolina| Powered by Sentiment + Market Signals")
