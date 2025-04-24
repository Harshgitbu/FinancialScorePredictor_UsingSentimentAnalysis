import streamlit as st
import pandas as pd


def run():
    st.title("🧠 Final Verdict Engine")

    # Load data
    combined_df = pd.read_csv("data/combined_verdict_with_fundamentals.csv")
    tech_df = pd.read_csv("data/technical_indicators_wrds_output.csv")

    tickers = combined_df["ticker"].unique()
    selected_ticker = st.selectbox("Select a Ticker:", tickers)

    sentiment_weight = st.slider("Sentiment Weight", 0.0, 1.0, 0.33)
    fundamental_weight = st.slider("Fundamental Weight", 0.0, 1.0, 0.33)
    technical_weight = st.slider("Technical Weight", 0.0, 1.0, 0.34)

    # Normalize weights
    total = sentiment_weight + fundamental_weight + technical_weight
    sentiment_weight /= total
    fundamental_weight /= total
    technical_weight /= total

    # Get scores
    latest_sent = combined_df[combined_df["ticker"] == selected_ticker].sort_values("date").iloc[-1]
    latest_tech = tech_df[tech_df["TICKER"] == selected_ticker].sort_values("date").iloc[-1]

    sentiment_score = latest_sent["final_sentiment_score"]
    fundamental_score = latest_sent["fundamental_score"]
    technical_score = latest_tech["sharpe_ratio"] / 3  # Normalize Sharpe to ~0-1 scale

    final_score = (sentiment_score * sentiment_weight +
                   fundamental_score * fundamental_weight +
                   technical_score * technical_weight)

    st.subheader("Scores")
    st.write(f"Sentiment Score: {sentiment_score:.2f}")
    st.write(f"Fundamental Score: {fundamental_score:.2f}")
    st.write(f"Technical Score (scaled): {technical_score:.2f}")

    st.metric("Final Weighted Score", f"{final_score:.2f}")

    # Final verdict logic
    if final_score > 0.75:
        verdict = "🔼 Strong Buy"
    elif final_score > 0.6:
        verdict = "🟢 Buy"
    elif final_score < 0.4:
        verdict = "🔻 Sell"
    elif final_score < 0.25:
        verdict = "🔴 Strong Sell"
    else:
        verdict = "🟡 Hold"

    st.subheader("Final Verdict")
    st.success(verdict)