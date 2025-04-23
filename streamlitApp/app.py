import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Financial Sentiment Stock Predictor", layout="centered")

# ------------------- Load Model & Label Encoder -------------------
@st.cache_resource
def load_model():
    return joblib.load("models/final_xgb_model.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load("models/label_encoder.pkl")

model = load_model()
le = load_encoder()

# ------------------- Simple Sentiment Simulator -------------------
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

# ------------------- Default Stock Feature Values -------------------
default_inputs = {
    'price': 100.0,
    'volume': 1_000_000,
    'low_bid': 99.0,
    'high_ask': 101.0,
    'sp500_return': 0.001,
    'news_score': 0.1,
    'twitter_score': 0.1,
    'final_sentiment_score': 0.1,
    'sentiment_1d': 0.1,
    'sentiment_3d_avg': 0.1,
    'sentiment_7d_avg': 0.1,
    'ret3': 0.01,
    'vol7': 0.02,
    'vma7': 1_200_000,
    'bidask_spread': 2.0
}

# ------------------- Streamlit App UI -------------------
st.title("📈 Financial Stock Action Predictor")
st.markdown("""
This app uses **sentiment analysis + financial indicators** to predict a stock action:  
**Buy**, **Hold**, or **Sell**  
(Example: "TSLA is up! This is bullish." → Buy)

""")

col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Choose a Stock Ticker", ['AAPL', 'MSFT', 'TSLA', 'GOOG', 'META', 'NFLX', 'ORCL'])

with col2:
    st.write("💡 Tip: Use market terms like _bullish_, _growth_, _crash_, etc.")
    user_text = st.text_area("Enter sentiment text (from Twitter or News):", height=100)

st.markdown("---")

# ------------------- Predict Button -------------------
if st.button("🔍 Predict Stock Action"):
    if not user_text.strip():
        st.warning("Please enter a sample sentiment text.")
    else:
        # Generate sentiment score
        sentiment_score = simulate_sentiment(user_text)

        # Populate final input features
        features = default_inputs.copy()
        features['news_score'] = sentiment_score
        features['twitter_score'] = sentiment_score
        features['final_sentiment_score'] = 0.65 * sentiment_score + 0.35 * sentiment_score

        input_df = pd.DataFrame([features])
        prediction = model.predict(input_df)
        predicted_action = le.inverse_transform(prediction)[0]

        # ------------------- Display Result -------------------
        st.success(f"📌 **Predicted Action for {ticker}: {predicted_action.upper()}**")
        st.metric(label="Inferred Sentiment Score", value=round(sentiment_score, 3))
        st.info("✅ Based on combined sentiment and financial features.")

        # Show the full input snapshot (optional)
        with st.expander("See model input features"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}))

# ------------------- Footer -------------------
st.markdown("---")
st.caption("Built by Harsh Shah, Ishanay Sharma, Saketh Bolina| Powered by Sentiment + Market Signals")
