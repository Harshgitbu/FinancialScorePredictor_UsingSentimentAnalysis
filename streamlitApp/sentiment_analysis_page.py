import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import ast
import joblib
from sklearn.preprocessing import LabelEncoder

tweets_df = pd.read_csv("data/TwitterDataSentimentScore.csv.zip", compression='zip', parse_dates=["date"])
news_df = pd.read_csv("data/NewsDataSentimentScore.csv.zip", compression='zip', parse_dates=["date"])

# Helper to get top-N tweets
def get_top_tweets(ticker, n=3):
    df = tweets_df[tweets_df["ticker"] == ticker].dropna(subset=["clean_text", "sentiment_score"])
    # sort by absolute contribution if you want largest magnitude, or by positive only:
    top = df.nlargest(n, "sentiment_score")
    return top[["date", "clean_text", "sentiment_score"]]

# Helper to get top-N news
def get_top_news(ticker, n=3):
    df = news_df[news_df["ticker"] == ticker].dropna(subset=["description", "sentiment_score"])
    top = df.nlargest(n, "sentiment_score")
    return top[["date", "description", "sentiment_score"]]

def run():
    st.title("📈 Sentiment Analysis")

    # Load data
    tickers = pd.read_csv("data/yfinance_filtered_tickers.txt", header=None)[0].tolist()
    sentiment_data = pd.read_csv("data/ModelDataFile.csv")

    # Ticker selector
    txt = open("data/company_name_ticker.txt").read().strip()
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

    # Live chart
    ticker_obj = yf.Ticker(selected_ticker)

    # 1) Fetch 1-day intraday data at 1-minute granularity (includes pre/post-market)
    df_intraday = ticker_obj.history(
        period="1d",
        interval="1m",
        prepost=True
    )

    # 2) Build a Plotly figure
    fig = go.Figure()

    # price line
    fig.add_trace(go.Scatter(
        x=df_intraday.index, 
        y=df_intraday["Close"], 
        mode="lines", 
        name="Close"
    ))

    # optional: add a horizontal line at yesterday's close
    prev_close = df_intraday["Close"].iloc[0]
    fig.add_hline(
        y=prev_close,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Prev Close: {prev_close:.2f}",
        annotation_position="bottom right"
    )

    # 3) Update layout to mimic Yahoo style
    fig.update_layout(
        title=f"{selected_ticker} Intraday — 1m Interval (Pre/Post)",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=True,            # show range slider
        xaxis_rangebreaks=[                         # hide overnight gaps
            dict(bounds=["sat","mon"])
        ],
        template="plotly_dark",                     # dark theme
        margin=dict(l=40, r=20, t=60, b=40)
    )

    # 4) Range selector buttons (1h, 3h, 6h, all)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=3, label="3h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    # 5) Render in Streamlit
    st.plotly_chart(fig, use_container_width=True, height=450)

    # Filter sentiment data
    filtered = sentiment_data[sentiment_data["ticker"] == selected_ticker]

    # Score summary
    # st.subheader("Sentiment Score Summary")
    # st.dataframe(
    #     filtered[["date", "final_sentiment_score", "sentiment_1d", "sentiment_3d_avg", "sentiment_7d_avg"]].tail(10)
    # )
    st.subheader("Sentiment Score Summary")
    latest = filtered.sort_values("date").iloc[-1]

    # 2) Create three side-by-side metrics
    c1, c2, c3 = st.columns(3, gap="large")
    c1.metric("Final Sentiment", f"{latest['final_sentiment_score']:.2f}")
    c2.metric("3-Day Avg",        f"{latest['sentiment_3d_avg']:.2f}")
    c3.metric("7-Day Avg",        f"{latest['sentiment_7d_avg']:.2f}")
    # last_n = 5
    # recent = filtered.sort_values("date").tail(last_n)

    # for _, row in recent.iterrows():
    #     fs  = row["final_sentiment_score"]
    #     s1  = row["sentiment_1d"]
    #     s3  = row["sentiment_3d_avg"]
    #     s7  = row["sentiment_7d_avg"]

    #     c1, c2, c3 = st.columns(3, gap="small")
    #     c1.metric("Final Sentiment",  f"{fs:.2f}")
    #     c2.metric("3-Day Avg",        f"{s3:.2f}")
    #     c3.metric("7-Day Avg",        f"{s7:.2f}")
    #     st.markdown("---")

    # ------------------ Load Assets ------------------
    @st.cache_resource
    def load_model():
        return joblib.load("models/final_xgb_model.pkl")
    
    @st.cache_resource
    def load_encoder():
        return joblib.load("models/label_encoder.pkl")
    
    @st.cache_data
    def load_data():
        df = pd.read_csv("data/ModelDataFile.csv")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
        # Add missing engineered features if not present
        if 'ret3' not in df.columns:
            df['ret3'] = (df.groupby('ticker')['daily_return']
                            .rolling(3, min_periods=1).sum()
                            .reset_index(0, drop=True))
        if 'vol7' not in df.columns:
            df['vol7'] = (df.groupby('ticker')['daily_return']
                            .rolling(7, min_periods=1).std()
                            .reset_index(0, drop=True)).fillna(0)
        if 'vma7' not in df.columns:
            df['vma7'] = (df.groupby('ticker')['volume']
                            .rolling(7, min_periods=1).mean()
                            .reset_index(0, drop=True))
        if 'bidask_spread' not in df.columns:
            df['bidask_spread'] = df['high_ask'] - df['low_bid']
    
        return df
    
    
    # Load models and data
    model = load_model()
    le = load_encoder()
    data = load_data()

    # Mock sentiment pipelines (Fast load without transformers)
    twitter_sentiment = lambda x: {"label": "Neutral", "score": 0.5}
    news_sentiment = lambda x: {"label": "Neutral", "score": 0.5}

    # Predict Action
    # Load model and encoder (cached if needed)
    model = load("models/final_xgb_model.pkl")
    le = load("models/label_encoder.pkl")
    
    # User input for headline
    st.header("🔮 Make a Prediction")
    tickers = sorted(data['ticker'].unique())
    selected = st.selectbox("Choose a Stock", tickers)
    user_input = st.text_input("Enter a recent headline/tweet:", "Strong Q2 earnings, great future outlook!")

    if st.button("Predict Action"):
        score = 0.7 if "good" in user_input.lower() or "strong" in user_input.lower() else -0.3

        row = data[data['ticker'] == selected].sort_values("date").iloc[-1]

        features = [
            'price','volume','low_bid','high_ask','sp500_return',
            'news_score','twitter_score','final_sentiment_score',
            'sentiment_1d','sentiment_3d_avg','sentiment_7d_avg',
            'ret3','vol7','vma7','bidask_spread'
        ]

        input_data = row[features].copy()
        input_data['news_score'] = score
        input_data['final_sentiment_score'] = 0.65 * score + 0.35 * row['twitter_score']
        input_df = pd.DataFrame([input_data])

        prediction = le.inverse_transform(model.predict(input_df))[0]

        st.success(f"📢 Model Prediction for {selected}: **{prediction}**")

        st.subheader(f"📊 Last 30 Days: {selected}")
        hist = data[data['ticker'] == selected].sort_values("date")[-30:]
        st.line_chart(hist.set_index("date")["price"])


    # Top tweets and headlines
    # st.subheader("Top Contributing Tweets and News")
    # if "description" in filtered.columns:
    #     st.markdown("**Top 3 Tweets:**")
    #     top_tweets = filtered.sort_values("final_sentiment_score", ascending=False)["description"].head(3)
    #     for tweet in top_tweets:
    #         st.info(tweet)

    # if "embed_title" in filtered.columns:
    #     st.markdown("**Top 3 News Headlines:**")
    #     top_news = filtered.sort_values("final_sentiment_score", ascending=False)["embed_title"].head(3)
    #     for headline in top_news:
    #         st.success(headline)
    st.subheader("Top 3 Tweets by Sentiment")
    top_t = get_top_tweets(selected_ticker, 3)
    for _, row in top_t.iterrows():
        st.info(f"{row['clean_text']}\n\n Score: {row['sentiment_score']:.2f}")

    st.subheader("Top 3 News Headlines by Sentiment")
    top_n = get_top_news(selected_ticker, 3)
    for _, row in top_n.iterrows():
        st.success(f"{row['description']}\n\n Score: {row['sentiment_score']:.2f}")
