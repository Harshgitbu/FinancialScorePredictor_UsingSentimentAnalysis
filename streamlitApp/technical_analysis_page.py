import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def run():
    st.title("📉 Technical Analysis")

    # Load technical indicator data
    tech_df = pd.read_csv("C:/Users/ishan/Desktop/ISHANAY/BU docs/Spring 2025/Financial_analytics/Project/FinancialScorePredictor_UsingSentimentAnalysis/data/technical_indicators_wrds_output.csv")
    tickers = tech_df["TICKER"].unique()

    selected_ticker = st.selectbox("Select a Ticker:", tickers)
    selected = tech_df[tech_df["TICKER"] == selected_ticker]

    if selected.empty:
        st.warning("No technical data available for this ticker.")
        return

    latest = selected.sort_values("date").iloc[-1]

    st.metric("Sharpe Ratio", f"{latest['sharpe_ratio']:.2f}")
    st.markdown(f"**MA Signal**: {latest['ma_signal']}")
    st.markdown(f"**Bollinger Band Signal**: {latest['bb_signal']}")

    # Moving Averages Visual
    st.subheader("20/50/200 Day Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected["date"], y=selected["adj_price"], name="Price"))
    fig.add_trace(go.Scatter(x=selected["date"], y=selected["ma_20"], name="MA 20"))
    fig.add_trace(go.Scatter(x=selected["date"], y=selected["ma_50"], name="MA 50"))
    fig.add_trace(go.Scatter(x=selected["date"], y=selected["ma_200"], name="MA 200"))
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Price", legend_title="Lines")
    st.plotly_chart(fig, use_container_width=True)

    # Similar stocks (based on same MA signal and similar Sharpe ratio)
    st.subheader("Top 5 Technically Similar Stocks")
    threshold = 0.3
    others = tech_df[(tech_df["TICKER"] != selected_ticker) & (tech_df["ma_signal"] == latest["ma_signal"])]
    others["sharpe_diff"] = (others["sharpe_ratio"] - latest["sharpe_ratio"]).abs()
    similar = others.sort_values("sharpe_diff").head(5)

    if not similar.empty:
        st.dataframe(similar[["TICKER", "sharpe_ratio", "ma_signal"]])
    else:
        st.info("No similar stocks found.")