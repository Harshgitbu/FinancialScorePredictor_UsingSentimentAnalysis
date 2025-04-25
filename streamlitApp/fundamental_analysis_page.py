import streamlit as st
import pandas as pd
import ast


def run():
    st.title("📊 Fundamental Analysis")

    # Load data
    fundamentals = pd.read_csv("data/fundamental_scores_wrds.csv")
    tickers = fundamentals["tic"].unique()
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
    debt_weight = st.slider("Solvency", 0.0, 1.0, 0.25)

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
    # cols = ["tic","datadate","score_pe","score_pb","score_ev_ebitda","score_roce","score_margin","score_turnover","score_inventory","score_accruals","score_dte","score_cov"]
    # st.dataframe(pd.DataFrame(latest[cols]).rename(columns={latest.name: "Value"}))
    cols = [
    "score_pe","score_pb","score_ev_ebitda",
    "score_roce","score_margin",
    "score_turnover","score_inventory","score_accruals",
    "score_dte","score_cov"]

    # Mapping from raw column → display name
    rename_map = {
        "score_pe":        "P/E Score",
        "score_pb":        "P/B Score",
        "score_ev_ebitda": "EV/EBITDA Score",
        "score_roce":      "ROCE Score",
        "score_margin":    "Net Margin Score",
        "score_turnover":  "Asset Turnover Score",
        "score_inventory": "Inventory Turnover Score",
        "score_accruals":  "Accruals Score",
        "score_dte":       "Debt/Equity Score",
        "score_cov":       "Interest Coverage Score",
    }

    # One-line explanations for each metric
    explanation_map = {
    "P/E Score":             "Higher → more expensive valuation | Lower → cheaper relative to earnings",
    "P/B Score":             "Higher → premium to book | Lower → discount to book",
    "EV/EBITDA Score":       "Higher → pricier vs cash flow | Lower → more value-oriented",
    "ROCE Score":            "Higher → efficient capital use | Lower → less efficient returns",
    "Net Margin Score":      "Higher → stronger profitability | Lower → weaker profit margins",
    "Asset Turnover Score":  "Higher → better asset efficiency | Lower → under-utilized assets",
    "Inventory Turnover Score":"Higher → good inventory management | Lower → slow stock turnover",
    "Accruals Score":        "Higher → more accruals (lower quality) | Lower → higher earnings quality",
    "Debt/Equity Score":     "Higher → more leverage risk | Lower → more conservative balance sheet",
    "Interest Coverage Score":"Higher → easier debt servicing | Lower → potential cash flow strain",
}

    # Build a display DataFrame
    data = {
        "Metric": [],
        "Score": [],
        "Explanation": []
    }
    for raw in cols:
        disp = rename_map[raw]
        data["Metric"].append(disp)
        data["Score"].append(latest[raw])
        data["Explanation"].append(explanation_map[disp])

    df_display = pd.DataFrame(data)

    # Show as a nice table
    st.subheader("Fundamental Scores Breakdown")
    st.dataframe(df_display, use_container_width=True)
