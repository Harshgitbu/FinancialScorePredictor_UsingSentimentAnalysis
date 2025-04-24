import streamlit as st

from pages import sentiment_analysis, fundamental_analysis, technical_analysis, final_verdict, compare_stocks, project_overview

st.set_page_config(page_title="Stock Verdict Intelligence", layout="wide")

PAGES = {
    "📈 Sentiment Analysis": sentiment_analysis,
    "📊 Fundamental Analysis": fundamental_analysis,
    "📉 Technical Analysis": technical_analysis,
    "🧠 Final Verdict Engine": final_verdict,
    "📈 Compare Stocks": compare_stocks,
    "📂 Project Overview": project_overview,
}

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[page].run()
