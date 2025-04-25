import streamlit as st
from PIL import Image
import time

def run():
    logo = Image.open("data/app_logo.png")
    st.image(logo, width=150)
    st.info("Built by Ishanay Sharma, Harsh Shah and Saketh Bollina")
    st.title("🏠 StockSense")
    st.markdown("""
    Welcome to **StockSense** — your all-in-one platform for:
    
    📈 Real-time **Sentiment Analysis** from Twitter & News
    \n📊 Deep **Fundamental Analysis** of company financials
    \n📉 Smart **Technical Analysis** using Moving Averages, Bollinger Bands, and Sharpe Ratios
    \n🧠 Combined **Final Verdict Engine** with customizable weights
    \n📈 **Compare Stocks** across sentiment, fundamentals, and technicals
    
    StockSense is a working prototype that combines three analytical lenses: market mood, accounting health and price behavior into a single, transparent score. 
    You won’t find price predictions here, instead you get a clear demonstration of how sentiment, fundamentals and technicals can be fused into a data-driven stock scoring engine which can be customized according to your investment mantra.

    ---
    #### 🚀 How to Get Started
    - Select a module from the sidebar 📂
    - Adjust sliders for weights and thresholds
    - Discover top opportunities with AI-driven insights!

    ---
    #### 💡 Tip:
    The **final verdict** fuses **Sentiment + Fundamentals + Technicals** into a simple Buy/Hold/Sell recommendation.
    """)
