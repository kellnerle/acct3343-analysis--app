import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Overvalued Stock Screen", layout="wide")
st.title("Overvalued Stock Screen")

# ---------- 1. Load data ----------
@st.cache_data
def load_data():
    return pd.read_csv("universe.csv")

df = load_data()

# ---------- 2. Measures (tested in Colab first) ----------
def safe_div(a, b):
    b = b.where(b > 0)            # zero or negative denominator -> missing, not infinity
    return a / b

def add_measures(data):
    d = data.copy()
    ta = d["total_assets"]
    # BUY side: quality and value
    d["GPA"] = safe_div(d["gross_profit"], ta)
    d["BM"] = safe_div(d["equity"], d["market_cap"])
    # SELL side: earnings quality and distress
    d["STA"] = safe_div(d["net_income"] - d["operating_cash_flow"], ta)
    noa = d["total_debt"] + d["equity"] - d["cash"]
    d["SNOA"] = safe_div(noa, d["total_assets_prev"])
    d["Altman_Z"] = (1.2 * safe_div(d["current_assets"] - d["current_liabilities"], ta)
                     + 1.4 * safe_div(d["retained_earnings"], ta)
                     + 3.3 * safe_div(d["ebit"], ta)
                     + 0.6 * safe_div(d["market_cap"], d["total_liabilities"])
                     + 1.0 * safe_div(d["revenue"], ta))
    # Percentile ranks (0-100) across the universe; missing stays missing
    for col in ["GPA", "BM", "STA", "SNOA"]:
        d[col + "_pct"] = d[col].rank(pct=True) * 100
    return d

df = add_measures(df)

# ---------- 3. Sidebar filters ----------
st.sidebar.header("Filters")
sectors = sorted(df["sector"].dropna().unique())
chosen = st.sidebar.multiselect("Sectors to include", sectors, default=sectors)
min_cap = st.sidebar.number_input("Minimum market cap ($ millions)", 0, 100000, 300, step=100)

st.sidebar.subheader("SELL screen")
acc_cut = st.sidebar.slider("Accrual percentile at or above", 50, 99, 80)
z_cut = st.sidebar.slider("Altman Z below", 0.0, 4.0, 1.81, 0.01)
flags_needed = st.sidebar.slider("Red flags required (of 3)", 1, 3, 2)

st.sidebar.subheader("BUY screen")
w_gpa = st.sidebar.slider("Weight on profitability (vs value)", 0.0, 1.0, 0.5, 0.05)
n_buy = st.sidebar.slider("Number of BUY names", 5, 50, 20)

screen = df[df["sector"].isin(chosen) & (df["market_cap"] >= min_cap * 1e6)].copy()

# ---------- 4. SELL screen: earnings-quality red flags ----------
screen["flag_STA"] = screen["STA_pct"] >= acc_cut
screen["flag_SNOA"] = screen["SNOA_pct"] >= acc_cut
screen["flag_Z"] = screen["Altman_Z"] < z_cut
screen["red_flags"] = screen[["flag_STA", "flag_SNOA", "flag_Z"]].sum(axis=1)
sell = (screen[screen["red_flags"] >= flags_needed]
        .sort_values(["red_flags", "STA_pct"], ascending=False))

# ---------- 5. BUY screen: a different analysis (quality + value rank) ----------
buy_pool = screen[(screen["red_flags"] == 0)].dropna(subset=["GPA_pct", "BM_pct"]).copy()
buy_pool["buy_score"] = w_gpa * buy_pool["GPA_pct"] + (1 - w_gpa) * buy_pool["BM_pct"]
buy = buy_pool.sort_values("buy_score", ascending=False).head(n_buy)

# ---------- 6. Headline numbers ----------
c1, c2, c3 = st.columns(3)
c1.metric("Companies screened", len(screen))
c2.metric("SELL candidates", len(sell))
c3.metric("BUY candidates", len(buy))

# ---------- 7. Tables ----------
left, right = st.columns(2)
with left:
    st.subheader("SELL candidates")
    st.caption("Flagged for high accruals, high net operating assets, and/or distress")
    st.dataframe(sell[["ticker", "name", "sector", "red_flags", "STA", "SNOA", "Altman_Z"]]
                 .round(3), hide_index=True)
with right:
    st.subheader("BUY candidates")
    st.caption("Top combined rank on gross profitability and book-to-market, no red flags")
    st.dataframe(buy[["ticker", "name", "sector", "buy_score", "GPA", "BM"]]
                 .round(3), hide_index=True)

# ---------- 8. Chart ----------
st.subheader("Where the candidates sit")
plot = screen.dropna(subset=["STA", "Altman_Z"]).copy()
plot["group"] = np.where(plot["ticker"].isin(sell["ticker"]), "SELL",
                 np.where(plot["ticker"].isin(buy["ticker"]), "BUY", "Other"))
fig = px.scatter(plot, x="STA", y="Altman_Z", color="group", hover_name="ticker",
                 color_discrete_map={"SELL": "#C4501F", "BUY": "#1F5F8B", "Other": "#C8C8C8"})
fig.add_hline(y=z_cut, line_dash="dash")
st.plotly_chart(fig, use_container_width=True)

# ---------- 9. Company drill-down ----------
st.subheader("Company drill-down")
pick = st.selectbox("Choose a company", screen["ticker"].sort_values())
row = screen[screen["ticker"] == pick].iloc[0]
st.write(f"**{row['name']}** · {row['sector']} · {row['industry']}")
st.dataframe(pd.DataFrame({
    "Measure": ["GPA", "B/M", "STA", "Scaled NOA", "Altman Z", "Red flags"],
    "Value": [row["GPA"], row["BM"], row["STA"], row["SNOA"], row["Altman_Z"], row["red_flags"]],
    "Percentile": [row["GPA_pct"], row["BM_pct"], row["STA_pct"], row["SNOA_pct"], None, None],
}).round(2), hide_index=True)

# ---------- 10. Conclusions and methodology ----------
st.subheader("Investment conclusions")
st.markdown("""
**Top SELL:** *Write your case here — why the flags matter for this company, what the
cash flow statement shows, and what would make you wrong.*

**Top BUY:** *Write your case here.*
""")
with st.expander("Methodology and formulas"):
    st.markdown("""
- **Data:** universe.csv, built in Colab with yfinance (date pulled, firms included, exclusions).
- **GPA** = gross profit / total assets.  **B/M** = book equity / market cap.
- **STA** = (net income − operating cash flow) / total assets.
- **Scaled NOA** = (total debt + equity − cash) / prior-year total assets.
- **Altman Z** = 1.2·WC/TA + 1.4·RE/TA + 3.3·EBIT/TA + 0.6·MVE/TL + 1.0·Sales/TA.
- Missing inputs stay missing; a firm with a missing measure cannot trigger that flag.
""")
