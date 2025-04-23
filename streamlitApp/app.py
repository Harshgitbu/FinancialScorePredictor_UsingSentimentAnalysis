import streamlit as st
import pandas as pd
import joblib
from datetime import timedelta
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Load Assets ------------------
@st.cache_resource
def load_model():
    return joblib.load("models/final_xgb_model.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load("models/label_encoder.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("data/ModelDataFile.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

@st.cache_data
def load_sample_preds():
    return pd.read_csv("data/Actual_vs_Predictions.csv")

# Load models and data
model = load_model()
le = load_encoder()
data = load_data()
sample_preds = load_sample_preds()

# Sentiment pipelines (disabled for now to avoid cloud errors)
twitter_sentiment = lambda x: {"label": "Neutral", "score": 0.5}
news_sentiment = lambda x: {"label": "Neutral", "score": 0.5}

# ------------------ UI ------------------
st.set_page_config(layout="wide")

st.sidebar.title("📈 Stock Action Predictor")
section = st.sidebar.radio("Jump to:", ["Overview", "Prediction", "Backtest Mode", "Performance"])

# ------------------ Overview ------------------
if section == "Overview":
    st.title("💼 Stock Sentiment-Based Action Predictor")
    st.markdown("""
    Predict **Buy / Hold / Sell** actions for S&P 500 stocks based on:
    - 🤖 News & Twitter Sentiment (via FinBERT + Twitter RoBERTa)
    - 📉 Stock Indicators (price, volume, volatility, etc.)
    - 🧠 Bayesian-optimized XGBoost classifier
    
    **Final Model Accuracy:** ~71% on historic data (balanced classes)
    """)

    col1, col2, col3 = st.columns(3)
    col1.image("visualization/Bayes_XGB.png", caption="Final Model Accuracy")
    col2.image("visualization/Feature_Imp_final.png", caption="Feature Importance")
    col3.image("visualization/Baseline_model.png", caption="Baseline Model")

# ------------------ Prediction ------------------
elif section == "Prediction":
    st.header("🔮 Make a Prediction")
    tickers = sorted(data['ticker'].unique())
    selected = st.selectbox("Choose a Stock", tickers)
    user_input = st.text_input("Enter recent headline or tweet about this stock:", "Strong Q2 earnings, great future outlook!")

    if st.button("Predict Action"):
        # Simulate sentiment pipeline
        score = 0.7 if "good" in user_input.lower() or "strong" in user_input.lower() else -0.3

        # Pull last row for ticker
        row = data[data['ticker'] == selected].sort_values("date").iloc[-1]

        features = [
            'price','volume','low_bid','high_ask','sp500_return',
            'news_score','twitter_score','final_sentiment_score',
            'sentiment_1d','sentiment_3d_avg','sentiment_7d_avg',
            'ret3','vol7','vma7','bidask_spread'
        ]

        input_data = row[features].copy()
        input_data['news_score'] = score
        input_data['final_sentiment_score'] = 0.65 * score + 0.35 * row['twitter_score']
        input_df = pd.DataFrame([input_data])

        prediction = le.inverse_transform(model.predict(input_df))[0]

        st.success(f"📢 Model Prediction for {selected}: **{prediction}**")

        st.subheader(f"📊 Historical Trend for {selected}")
        hist = data[data['ticker'] == selected].sort_values("date")[-30:]
        st.line_chart(hist.set_index("date")["price"])

# ------------------ Backtest ------------------
elif section == "Backtest Mode":
    st.header("⏪ Roleplay Backtest Mode")
    role_date = st.date_input("Select a past date:", value=pd.to_datetime("2023-06-01"))
    role_ticker = st.selectbox("Pick a stock:", sorted(data['ticker'].unique()), key="role_ticker")

    role_df = data[data['ticker'] == role_ticker].copy()
    role_df['date'] = pd.to_datetime(role_df['date'])

    past_date = pd.to_datetime(role_date)

    week_before = role_df[(role_df['date'] >= past_date - timedelta(days=7)) & (role_df['date'] <= past_date)]
    week_after = role_df[(role_df['date'] > past_date) & (role_df['date'] <= past_date + timedelta(days=7))]

    st.subheader("📅 Week Before")
    st.dataframe(week_before[['date','price','sentiment_1d','final_sentiment_score']])

    st.subheader("📅 Week After")
    st.dataframe(week_after[['date','price','sentiment_1d','final_sentiment_score']])

    fig, ax = plt.subplots(figsize=(10,4))
    plt.plot(role_df['date'], role_df['price'], label="Price", marker='o')
    plt.axvline(past_date, color='red', linestyle='--', label="Selected Date")
    plt.title(f"Price Trend for {role_ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig)

# ------------------ Performance ------------------
elif section == "Performance":
    st.header("📈 Model Evaluation")

    col1, col2 = st.columns(2)
    col1.image("visualization/conf_matrix_final.png", caption="Confusion Matrix")
    col2.image("visualization/Actual_vs_Prediction_SS.png", caption="Sample Predictions")

    st.subheader("📝 Random Sample of Actual vs Predicted")
    sample_view = sample_preds[['ticker','price','final_sentiment_score','sentiment_3d_avg','ret3','vol7','Actual_Action','Predicted_Action']]
    st.dataframe(sample_view.sample(20).reset_index(drop=True))

# ------------------- Footer -------------------
st.markdown("---")
st.caption("Built by Harsh Shah, Ishanay Sharma, Saketh Bolina| Powered by Sentiment + Market Signals")
