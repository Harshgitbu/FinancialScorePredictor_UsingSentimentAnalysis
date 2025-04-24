import streamlit as st

# from streamlitApp import sentiment_analysis_page, fundamental_analysis_page, technical_analysis_page, final_verdict_page, compare_stocks_page, methodology_page
import sentiment_analysis_page
import fundamental_analysis_page
import technical_analysis_page
import final_verdict_page
import compare_stocks_page
import methodology_page

st.set_page_config(page_title="Stock Verdict Intelligence", layout="wide")

PAGES = {
    "📈 Sentiment Analysis": sentiment_analysis_page,
    "📊 Fundamental Analysis": fundamental_analysis_page,
    "📉 Technical Analysis": technical_analysis_page,
    "🧠 Final Verdict Engine": final_verdict_page,
    "📈 Compare Stocks": compare_stocks_page,
    "📂 Project Overview": methodology_page,
}

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[page].run()
