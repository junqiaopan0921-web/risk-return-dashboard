import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def download_price_data(tickers, start, end):
    """Download adjusted close prices from Yahoo Finance."""
    if len(tickers) == 0:
        return pd.DataFrame()
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        data = data.to_frame(name=tickers[0])
    return data.dropna(how="all")

def compute_returns(price_df):
    return price_df.pct_change().dropna()

def annualized_return(returns, periods_per_year=252):
    mean_period = returns.mean()
    return (1 + mean_period) ** periods_per_year - 1

def annualized_volatility(returns, periods_per_year=252):
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - risk_free_rate) / ann_vol

def max_drawdown(price_series):
    cum_max = price_series.cummax()
    drawdown = price_series / cum_max - 1
    return drawdown.min()

def series_max_drawdown(returns):
    cum = (1 + returns).cumprod()
    return max_drawdown(cum)

def cvar_historical(returns, alpha=0.95):
    sorted_returns = returns.sort_values()
    var_cutoff = sorted_returns.quantile(1 - alpha)
    tail = sorted_returns[sorted_returns <= var_cutoff]
    if len(tail) == 0:
        return np.nan
    return tail.mean()

def portfolio_metrics(returns_df, weights, risk_free_rate=0.0, alpha=0.95, periods_per_year=252):
    weights = np.array(weights)
    weights = weights / weights.sum()
    port_returns = (returns_df * weights).sum(axis=1)
    ann_ret = annualized_return(port_returns, periods_per_year)
    ann_vol = annualized_volatility(port_returns, periods_per_year)
    sharpe = sharpe_ratio(port_returns, risk_free_rate, periods_per_year)
    mdd = series_max_drawdown(port_returns)
    cvar = cvar_historical(port_returns, alpha)
    return {
        "Annualized Return": ann_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": mdd,
        f"CVaR {int(alpha*100)}%": cvar,
    }, port_returns

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="StockPeers-style Dashboard", layout="wide")
st.markdown(
    """
    <style>
    /* Dark background for main page */
    .main .block-container {
        background-color: #121212;
        color: white;
    }
    /* Dark scrollbar for sidebar */
    .css-1d391kg {
        background-color: #1e1e1e;
    }
    </style>
    """, unsafe_allow_html=True
)
st.title("📊 Interactive Risk–Return Dashboard")
st.markdown(
    "Analyze how different assets balance **risk and return**, "
    "build custom portfolios, and compare them to benchmarks like **SPY** and **QQQ**."
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Controls")
today = date.today()

default_universe = ["SPY", "QQQ", "DIA", "XLF", "XLE", "XLK", "XLI", "XLV", "XLY", "XLP"]
extra_tickers_str = st.sidebar.text_input("Extra tickers (comma-separated):", "")
extra_tickers = [t.strip().upper() for t in extra_tickers_str.split(",") if t.strip()]
universe = sorted(list(set(default_universe + extra_tickers)))

selected_tickers = st.sidebar.multiselect("Assets to analyze", universe, default=["SPY", "QQQ", "XLK"])
start_date = st.sidebar.date_input("Start date", today - timedelta(days=5*365))
end_date = st.sidebar.date_input("End date", today)
rf_rate = st.sidebar.slider("Risk-free rate (%)", 0.0, 5.0, 0.0, 0.25)/100
alpha = st.sidebar.slider("CVaR confidence level", 0.90, 0.99, 0.95, 0.01)

weight_mode = st.sidebar.radio("Portfolio weighting", ["Equal weight", "Custom weights"])
custom_weights = {}
if weight_mode == "Custom weights" and selected_tickers:
    st.sidebar.markdown("#### Custom weights")
    for t in selected_tickers:
        custom_weights[t] = st.sidebar.number_input(
            f"Weight for {t}", 0.0, 1.0, 1.0 / len(selected_tickers), 0.01
        )

# -------------------------------------------------
# Validation
# -------------------------------------------------
if not selected_tickers:
    st.warning("Select at least one asset.")
    st.stop()
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# -------------------------------------------------
# Data download
# -------------------------------------------------
with st.spinner("Downloading price data..."):
    prices = download_price_data(selected_tickers, start_date, end_date)
if prices.empty:
    st.error("No price data available.")
    st.stop()
returns = compute_returns(prices)

# Ensure DataFrames for single ticker
if isinstance(prices, pd.Series):
    prices = prices.to_frame()
if isinstance(returns, pd.Series):
    returns = returns.to_frame()
cum_returns = (1 + returns).cumprod()
if isinstance(cum_returns, pd.Series):
    cum_returns = cum_returns.to_frame()

# -------------------------------------------------
# Price & cumulative returns charts (side-by-side)
# -------------------------------------------------
st.subheader("📈 Price History & Cumulative Returns")
col1, col2 = st.columns(2)
with col1:
    price_fig = px.line(
        prices,
        x=prices.index,
        y=prices.columns,
        labels={"value":"Price ($)", "index":"Date", "variable":"Asset"},
        title="Adjusted Close Prices",
        template="plotly_dark"
    )
    st.plotly_chart(price_fig, use_container_width=True)
with col2:
    cum_fig = px.line(
        cum_returns,
        x=cum_returns.index,
        y=cum_returns.columns,
        labels={"value":"Growth of $1", "index":"Date", "variable":"Asset"},
        title="Cumulative Growth",
        template="plotly_dark"
    )
    st.plotly_chart(cum_fig, use_container_width=True)

# -------------------------------------------------
# Asset-level metrics
# -------------------------------------------------
st.subheader("📊 Annualized Risk & Return Metrics")
asset_df = pd.DataFrame({
    "Annualized Return": [annualized_return(returns[c]) for c in returns.columns],
    "Annualized Volatility": [annualized_volatility(returns[c]) for c in returns.columns],
    "Sharpe Ratio": [sharpe_ratio(returns[c], rf_rate) for c in returns.columns],
    "Max Drawdown": [series_max_drawdown(returns[c]) for c in returns.columns],
    f"CVaR {int(alpha*100)}%": [cvar_historical(returns[c], alpha) for c in returns.columns]
}, index=returns.columns)
st.dataframe(asset_df.style.format({
    "Annualized Return": "{:.2%}",
    "Annualized Volatility": "{:.2%}",
    "Sharpe Ratio": "{:.2f}",
    "Max Drawdown": "{:.2%}",
    f"CVaR {int(alpha*100)}%": "{:.2%}",
}))

# -------------------------------------------------
# Risk vs Return scatter plot
# -------------------------------------------------
st.subheader("📊 Risk vs Return Scatter Plot")
scatter_fig = px.scatter(
    asset_df,
    x="Annualized Volatility",
    y="Annualized Return",
    text=asset_df.index,
    size="Annualized Return",
    hover_data=["Sharpe Ratio", "Max Drawdown"],
    labels={"Annualized Volatility":"Risk", "Annualized Return":"Return"},
    title="Risk vs Return",
    template="plotly_dark"
)
scatter_fig.update_traces(textposition="top center")
st.plotly_chart(scatter_fig, use_container_width=True)

# -------------------------------------------------
# Portfolio vs benchmarks
# -------------------------------------------------
st.subheader("💼 Custom Portfolio vs Benchmarks")
if weight_mode == "Equal weight":
    weights = [1/len(selected_tickers)] * len(selected_tickers)
else:
    weights = [custom_weights[t] for t in selected_tickers]
    if sum(weights) == 0:
        st.error("All custom weights are zero. Adjust them in the sidebar.")
        st.stop()

portfolio_info, port_returns = portfolio_metrics(returns[selected_tickers], weights, rf_rate, alpha)
benchmarks = {}
for bench in ["SPY","QQQ"]:
    if bench in returns.columns:
        bm_info, bm_ret = portfolio_metrics(returns[[bench]], [1.0], rf_rate, alpha)
        benchmarks[bench] = (bm_info, bm_ret)

port_df = pd.DataFrame([{"Portfolio":"Custom Portfolio", **portfolio_info}] + 
                       [{"Portfolio":name, **info} for name,(info,_) in benchmarks.items()])
st.dataframe(port_df.style.format({
    "Annualized Return":"{:.2%}",
    "Annualized Volatility":"{:.2%}",
    "Sharpe Ratio":"{:.2f}",
    "Max Drawdown":"{:.2%}",
    f"CVaR {int(alpha*100)}%":"{:.2%}",
}))

# -------------------------------------------------
# Cumulative returns portfolio vs benchmarks
# -------------------------------------------------
st.subheader("💹 Cumulative Returns: Portfolio vs Benchmarks")
cum_df = pd.DataFrame({"Custom Portfolio": (1+port_returns).cumprod()})
for name,(_,bm_ret) in benchmarks.items():
    cum_df[name] = (1+bm_ret).cumprod()
cum_fig2 = px.line(
    cum_df,
    x=cum_df.index,
    y=cum_df.columns,
    labels={"value":"Growth of $1","index":"Date","variable":"Portfolio"},
    title="Portfolio vs Benchmarks",
    template="plotly_dark"
)
st.plotly_chart(cum_fig2, use_container_width=True)

# -------------------------------------------------
# Footer explanation
# -------------------------------------------------
st.markdown("---")
st.markdown(
    f"""
### How to interpret this dashboard
- **Annualized return** – average yearly growth.
- **Volatility** – how much returns fluctuate (risk).
- **Sharpe ratio** – return per unit of risk after subtracting {rf_rate:.2%} risk-free rate.
- **Max drawdown** – worst peak-to-trough loss.
- **CVaR {int(alpha*100)}%** – average loss on worst days.

Use the sidebar to:
- Change the **time period** and **assets**
- Switch between **equal-weight** and **custom-weight** portfolios
- Compare your portfolio to benchmarks like **SPY** and **QQQ**
"""
)
