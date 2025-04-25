import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ast
import numpy as np

def run():
    st.title("📉 Technical Analysis")

    # Load technical indicator data
    tech_df = pd.read_csv("C:/Users/ishan/Desktop/ISHANAY/BU docs/Spring 2025/Financial_analytics/Project/FinancialScorePredictor_UsingSentimentAnalysis/data/technical_indicators_wrds_output.csv")
    tickers = tech_df["TICKER"].unique()

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
    selected = tech_df[tech_df["TICKER"] == selected_ticker]

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

    summary = (
    tech_df
    .sort_values("date")
    .drop_duplicates(subset="TICKER", keep="last")
    .reset_index(drop=True)
)

    # 3) Now when you pick `latest`, do it from `summary`
    latest = summary[summary["TICKER"] == selected_ticker].iloc[0]

    # 4) Filter others *also* from the summary
    others = summary[
        (summary["ma_signal"] == latest["ma_signal"]) &
        (summary["TICKER"] != selected_ticker)
    ].copy()

    # 5) Compute sharpe difference & grab the closest five tickers
    others["sharpe_diff"] = np.abs(others["sharpe_ratio"] - latest["sharpe_ratio"])
    similar = others.nsmallest(5, "sharpe_diff")

    # 6) Display those five unique tickers
    st.dataframe(similar[["TICKER","COMNAM","sharpe_ratio","ma_signal"]])