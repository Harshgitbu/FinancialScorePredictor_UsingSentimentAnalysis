import streamlit as st
from pages import sentiment_analysis
from pages import fundamental_analysis
from pages import technical_analysis
from pages import final_verdict
from pages import compare_stocks

st.set_page_config(page_title="Stock Verdict Intelligence", layout="wide")

PAGES = {
    "📈 Sentiment Analysis": sentiment_analysis,
    "📊 Fundamental Analysis": fundamental_analysis,
    "📉 Technical Analysis": technical_analysis,
    "🧠 Final Verdict Engine": final_verdict,
    "📈 Compare Stocks": compare_stocks,
}

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[page].run()
