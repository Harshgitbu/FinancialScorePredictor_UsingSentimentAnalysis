import streamlit as st
import streamlit.components.v1 as components


def run():
    st.title("📂 Project Overview")

    st.header("🔍 Project Purpose & Approach")
    st.markdown("""
    This project combines **Sentiment Analysis**, **Fundamental Analysis**, and **Technical Analysis** to provide an
    intelligent stock recommendation system.

    - **Sentiment Module** uses tweet and news headlines to score sentiment
    - **Fundamentals Module** evaluates valuation, profitability, efficiency, and debt
    - **Technical Module** uses moving averages, Bollinger bands, and Sharpe ratio
    - All modules are fused to create a final weighted `Buy / Hold / Sell` verdict
    """)

    st.header("📁 Methodology Files from GitHub")
    st.markdown("View the core notebooks and code used in building this application:")
    st.markdown("https://github.com/Harshgitbu/FinancialScorePredictor_UsingSentimentAnalysis")

    st.header("📜 Summary & Next Steps")
    st.markdown("""
    - Use the app to explore individual stock diagnostics and compare them.
    - Adjust weights in the final verdict engine to reflect different investor profiles.
    - Future improvements: integration with live trading APIs, forecast modeling, risk simulation.
    """)
