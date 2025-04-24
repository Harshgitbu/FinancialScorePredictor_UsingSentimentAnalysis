import streamlit as st
from modules import sentiment_analysis
from modules import fundamental_analysis
from modules import technical_analysis
from modules import final_verdict
from modules import compare_stocks

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
