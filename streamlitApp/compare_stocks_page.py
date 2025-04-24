import streamlit as st
import pandas as pd
import plotly.express as px


def run():
    st.title("📈 Compare Stocks")

    # Load data
    verdicts = pd.read_csv("C:/Users/ishan/Desktop/ISHANAY/BU docs/Spring 2025/Financial_analytics/Project/FinancialScorePredictor_UsingSentimentAnalysis/data/combined_verdict_with_fundamentals.csv")
    tech = pd.read_csv("C:/Users/ishan/Desktop/ISHANAY/BU docs/Spring 2025/Financial_analytics/Project/FinancialScorePredictor_UsingSentimentAnalysis/data/technical_indicators_wrds_output.csv")

    tickers = verdicts["ticker"].unique()
    selected = st.multiselect("Select up to 5 tickers to compare:", tickers, default=tickers[:2])

    if len(selected) < 2:
        st.info("Please select at least two tickers to compare.")
        return

    df_latest = []
    for ticker in selected:
        v = verdicts[verdicts["ticker"] == ticker].sort_values("date").iloc[-1]
        t = tech[tech["TICKER"] == ticker].sort_values("date").iloc[-1]
        df_latest.append({
            "Ticker": ticker,
            "Sentiment Score": v["final_sentiment_score"],
            "Fundamental Score": v["fundamental_score"],
            "Technical Score": t["sharpe_ratio"] / 3  # Normalize
        })

    df_compare = pd.DataFrame(df_latest)

    st.subheader("Score Comparison Table")
    st.dataframe(df_compare.set_index("Ticker"))

    st.subheader("Radar Chart")
    radar = px.line_polar(df_compare.melt(id_vars="Ticker"),
                          r="value", theta="variable", color="Ticker",
                          line_close=True)
    st.plotly_chart(radar, use_container_width=True)
