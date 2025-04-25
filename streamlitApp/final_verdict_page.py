import streamlit as st
import pandas as pd
import ast

def run():
    st.title("🧠 Final Verdict Engine")

    # Load data
    combined_df = pd.read_csv("data/combined_verdict_with_fundamentals.csv")
    tech_df = pd.read_csv("/data/technical_indicators_wrds_output.csv")

    tickers = combined_df["ticker"].unique()
    txt = open("C:/Users/ishan/Desktop/ISHANAY/BU docs/Spring 2025/Financial_analytics/Project/FinancialScorePredictor_UsingSentimentAnalysis/data/company_name_ticker.txt").read().strip()
    # # Ticker selector
    # selected_ticker = st.selectbox("Select a Ticker:", tickers)
    mapping = ast.literal_eval("{" + txt + "}")

    # 2) Prepare list of tickers
    tickers = list(mapping.keys())

    # 3) Build the selectbox
    selected_ticker = st.selectbox(
        "Select a Company:",
        tickers,
        format_func=lambda t: f"{mapping[t]}\n({t})"
    )

    # 4) Use the selection
    st.write("You picked:", mapping[selected_ticker], f"({selected_ticker})")

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
    if final_score <= -0.026:
        verdict = "🔴 Strong Sell"
    elif final_score <=  0.075:
        verdict = "🔻 Sell"
    elif final_score <=  0.221:
        verdict = "🟡 Hold"
    elif final_score <=  0.371:
        verdict = "🟢 Buy"
    else:
        verdict = "🔼 Strong Buy"

    st.subheader("Final Verdict")
    st.success(verdict)
