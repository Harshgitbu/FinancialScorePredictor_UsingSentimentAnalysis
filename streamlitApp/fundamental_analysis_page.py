import streamlit as st
import pandas as pd


def run():
    st.title("📊 Fundamental Analysis")

    # Load data
    fundamentals = pd.read_csv("C:/Users/ishan/Desktop/ISHANAY/BU docs/Spring 2025/Financial_analytics/Project/FinancialScorePredictor_UsingSentimentAnalysis/data/fundamental_scores_wrds.csv")
    tickers = fundamentals["tic"].unique()

    # Ticker selector
    selected_ticker = st.selectbox("Select a Ticker:", tickers)
    selected = fundamentals[fundamentals["tic"] == selected_ticker]

    if selected.empty:
        st.warning("No fundamental data available for this ticker.")
        return

    latest = selected.sort_values("datadate").iloc[-1]

    # Slider weights
    st.markdown("### Weight the Fundamental Categories")
    val_weight = st.slider("Valuation", 0.0, 1.0, 0.25)
    prof_weight = st.slider("Profitability", 0.0, 1.0, 0.25)
    eff_weight = st.slider("Efficiency", 0.0, 1.0, 0.25)
    debt_weight = st.slider("Debt", 0.0, 1.0, 0.25)

    # Sub-scores
    val_score = latest[["score_pe", "score_pb", "score_ev_ebitda"]].mean()
    prof_score = latest[["score_roce", "score_margin"]].mean()
    eff_score = latest[["score_turnover", "score_inventory", "score_accruals"]].mean()
    debt_score = latest[["score_dte", "score_cov"]].mean()

    # Final fundamental score
    final_score = (
        val_weight * val_score +
        prof_weight * prof_score +
        eff_weight * eff_score +
        debt_weight * debt_score
    )

    st.metric("Final Fundamental Score", f"{final_score:.2f}")

    st.markdown("### Raw Financial Ratios")
    cols = ["tic","datadate","score_pe","score_pb","score_ev_ebitda","score_roce","score_margin","score_turnover","score_inventory","score_accruals","score_dte","score_cov"]
    st.dataframe(pd.DataFrame(latest[cols]).rename(columns={latest.name: "Value"}))
