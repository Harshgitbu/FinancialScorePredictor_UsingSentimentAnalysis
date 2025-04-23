import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

st.set_page_config(page_title="Sentiment-Driven Stock Predictor", layout="wide")

# Load model and encoder
@st.cache_resource
def load_model():
    return joblib.load("models/final_xgb_model.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load("models/label_encoder.pkl")

model = load_model()
le = load_encoder()

# Load actual vs prediction (for display)
@st.cache_data
def load_sample_predictions():
    return pd.read_csv("data/Actual_vs_Prediction.csv")

# Load past stock data (gold layer historical stock sentiment data)
@st.cache_data
def load_gold_data():
    return pd.read_csv("data/ModelDataFile.csv", parse_dates=['date'])

def simulate_sentiment(text):
    text = text.lower()
    pos_words = ['buy', 'gain', 'growth', 'bull', 'profit', 'strong', 'positive']
    neg_words = ['sell', 'loss', 'drop', 'bear', 'crash', 'weak', 'negative']
    score = 0
    for word in pos_words:
        if word in text:
            score += 0.3
    for word in neg_words:
        if word in text:
            score -= 0.3
    return np.clip(score, -1, 1)

# ------------------------ Sidebar Navigation ------------------------
st.sidebar.title("📊 App Navigation")
selection = st.sidebar.radio("Go to", ["📘 Project Overview", "🚀 Make a Prediction", "🔍 Backtesting (Roleplay Mode)", "📈 Model Performance"])

# ------------------------ 1. Project Overview ------------------------
if selection == "📘 Project Overview":
    st.title("📘 Sentiment-Based Stock Action Classifier")
    st.markdown("""
    **Goal:** Predict **Buy / Hold / Sell** actions for S&P 500 stocks using sentiment analysis + historical financial indicators.

    - ✅ Combined Twitter & News sentiment using BERT models (RoBERTa, FinBERT)
    - ✅ Engineered lag-based rolling sentiment indicators
    - ✅ Used SMOTETomek for balancing labels
    - ✅ Final XGBoost model tuned via Bayesian optimization
    - ✅ Achieved 71% accuracy (Balanced Buy/Hold/Sell)
    """)
    
    st.image("visualization/Feature_Imp_final.png", caption="📊 Feature Importance from XGBoost")

# ------------------------ 2. Make a Prediction ------------------------
elif selection == "🚀 Make a Prediction":
    st.title("🚀 Predict a Stock Action")
    
    ticker = st.selectbox("Select a Stock", ['AAPL', 'MSFT', 'GOOG', 'META', 'NFLX', 'ORCL'])
    text = st.text_area("Enter Sentiment (News/Tweet Style)", height=120)

    if st.button("Predict Stock Action"):
        if text.strip():
            score = simulate_sentiment(text)
            final_score = 0.65 * score + 0.35 * score
            input_df = pd.DataFrame([{ 
                'price': 100.0, 'volume': 1_000_000, 'low_bid': 99.0, 'high_ask': 101.0,
                'sp500_return': 0.001, 'news_score': score, 'twitter_score': score,
                'final_sentiment_score': final_score, 'sentiment_1d': score,
                'sentiment_3d_avg': score, 'sentiment_7d_avg': score,
                'ret3': 0.02, 'vol7': 0.03, 'vma7': 1_100_000, 'bidask_spread': 2.0
            }])
            pred = model.predict(input_df)
            label = le.inverse_transform(pred)[0]
            
            st.success(f"📌 Predicted Action for {ticker}: **{label}**")
            st.metric("Sentiment Score", round(score, 3))
        else:
            st.warning("Please enter a sentiment text.")

# ------------------------ 3. Roleplay Mode (Backtest) ------------------------
elif selection == "🔍 Backtesting (Roleplay Mode)":
    st.title("🔍 Roleplay: Sentiment-Driven Prediction in the Past")

    gold_df = load_gold_data()
    unique_tickers = sorted(gold_df['ticker'].unique())
    
    ticker = st.selectbox("Select Stock for Roleplay", unique_tickers)
    past_date = st.date_input("Choose a Date (in Past)", datetime(2023, 4, 1), min_value=datetime(2022,1,1), max_value=datetime(2023,12,1))

    role_df = gold_df[(gold_df['ticker'] == ticker)]
    week_before = role_df[role_df['date'].between(past_date - timedelta(days=7), past_date)]
    week_after = role_df[role_df['date'].between(past_date, past_date + timedelta(days=7))]

    st.markdown(f"### Data 1 Week Before ({past_date - timedelta(days=7)} to {past_date})")
    st.dataframe(week_before[['date', 'price', 'news_score', 'twitter_score', 'final_sentiment_score']])

    st.markdown(f"### Data 1 Week After ({past_date} to {past_date + timedelta(days=7)})")
    st.dataframe(week_after[['date', 'price', 'daily_return']])

# ------------------------ 4. Performance Summary ------------------------
elif selection == "📈 Model Performance":
    st.title("📈 Final Model Evaluation")
    st.markdown("### 1. Confusion Matrix")
    st.image("visualization/Bayes_XGB.png")

    st.markdown("### 2. Accuracy Comparison")
    st.image("visualization/Baseline_model.png")

    st.markdown("### 3. Sample Predictions")
    sample = load_sample_predictions().sample(10)
    st.table(sample[['ticker','final_sentiment_score','Actual_Action','Predicted_Action']])


# ------------------- Footer -------------------
st.markdown("---")
st.caption("Built by Harsh Shah, Ishanay Sharma, Saketh Bolina| Powered by Sentiment + Market Signals")
