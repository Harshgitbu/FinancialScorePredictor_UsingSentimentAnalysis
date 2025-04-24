import streamlit as st
import pandas as pd
import yfinance as yf


def run():
    st.title("📈 Sentiment Analysis")

    # Load data
    tickers = pd.read_csv("data/yfinance_filtered_tickers.txt", header=None)[0].tolist()
    sentiment_data = pd.read_csv("data/ModelDataFile.csv")

    # Ticker selector
    selected_ticker = st.selectbox("Select a Ticker:", tickers)

    # Live chart
    st.subheader(f"Live Stock Price: {selected_ticker}")
    data = yf.download(selected_ticker, period="6mo")
    st.line_chart(data["Adj Close"])

    # Filter sentiment data
    filtered = sentiment_data[sentiment_data["ticker"] == selected_ticker]

    # Score summary
    st.subheader("Sentiment Score Summary")
    st.dataframe(
        filtered[["date", "final_sentiment_score", "sentiment_1d", "sentiment_3d_avg", "sentiment_7d_avg"]].tail(10)
    )

    # Top tweets and headlines
    st.subheader("Top Contributing Tweets and News")
    if "description" in filtered.columns:
        st.markdown("**Top 3 Tweets:**")
        top_tweets = filtered.sort_values("final_sentiment_score", ascending=False)["description"].head(3)
        for tweet in top_tweets:
            st.info(tweet)

    if "embed_title" in filtered.columns:
        st.markdown("**Top 3 News Headlines:**")
        top_news = filtered.sort_values("final_sentiment_score", ascending=False)["embed_title"].head(3)
        for headline in top_news:
            st.success(headline)
