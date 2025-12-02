import streamlit as st
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
 data = yf.download(
 tickers,
 start=start,
 end=end,
 auto_adjust=True,
 progress=False,
 )
 if isinstance(data.columns, pd.MultiIndex):
 data = data["Close"]
 else:
 data = data.to_frame(name=tickers[0])
 return data.dropna(how="all")
def compute_returns(price_df):
 """Daily returns."""
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
def portfolio_metrics(returns_df, weights, risk_free_rate=0.0, alpha=0.95,
periods_per_year=252):
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
# Page config & simple styling
# -------------------------------------------------
st.set_page_config(page_title="Risk–Return Dashboard", layout="wide")
st.title("📊 Interactive Risk–Return Dashboard")
st.markdown(
 """
Analyze how different assets balance **risk and return**, build your own
portfolios,
and compare them to benchmarks like **SPY** and **QQQ** using historical
data from Yahoo Finance.
"""
)
# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.sidebar.header("Controls")
default_universe = [
 "SPY", "QQQ", "DIA",
 "XLF", "XLE", "XLK", "XLI", "XLV",
 "XLY", "XLP", "IWM",
]
extra_tickers_str = st.sidebar.text_input(
 "Extra tickers (comma-separated, e.g. AAPL,MSFT,TSLA):",
 "",
)
extra_tickers = [t.strip().upper() for t in extra_tickers_str.split(",")
if t.strip()]
universe = sorted(list(set(default_universe + extra_tickers)))
selected_tickers = st.sidebar.multiselect(
 "Assets to analyze",
 options=universe,
 default=["SPY", "QQQ", "XLK"],
)
today = date.today()
start_date = st.sidebar.date_input("Start date", today - timedelta(days=5
* 365))
end_date = st.sidebar.date_input("End date", today)
rf_rate = st.sidebar.slider(
 "Risk-free rate (annualized, %)",
 min_value=0.0,
 max_value=5.0,
 value=0.0,
 step=0.25,
) / 100
alpha = st.sidebar.slider(
 "CVaR confidence level",
 min_value=0.90,
 max_value=0.99,
 value=0.95,
 step=0.01,
)
weight_mode = st.sidebar.radio(
 "Portfolio weighting",
 options=["Equal weight", "Custom weights"],
)
custom_weights = {}
if weight_mode == "Custom weights" and selected_tickers:
 st.sidebar.markdown("#### Custom weights")
 for t in selected_tickers:
 custom_weights[t] = st.sidebar.number_input(
 f"Weight for {t}",
 min_value=0.0,
 max_value=1.0,
 value=1.0 / len(selected_tickers),
 step=0.01,
 )
if not selected_tickers:
 st.warning("Select at least one asset.")
 st.stop()
if start_date >= end_date:
 st.error("Start date must be before end date.")
 st.stop()
# -------------------------------------------------
# Data loading
# -------------------------------------------------
with st.spinner("Downloading price data from Yahoo Finance..."):
 prices = download_price_data(selected_tickers, start_date, end_date)
if prices.empty:
 st.error("No price data for this selection (check tickers or date
range).")
 st.stop()
returns = compute_returns(prices)
# -------------------------------------------------
# Price chart
# -------------------------------------------------
st.subheader("Price history")
price_fig = px.line(
 prices,
 labels={"value": "Price", "index": "Date", "variable": "Asset"},
)
st.plotly_chart(price_fig, use_container_width=True)
# -------------------------------------------------
# Asset-level metrics
# -------------------------------------------------
st.subheader("Risk–return metrics (individual assets)")
asset_rows = []
for col in returns.columns:
 r = returns[col]
 asset_rows.append(
 {
 "Asset": col,
 "Annualized Return": annualized_return(r),
 "Annualized Volatility": annualized_volatility(r),
 "Sharpe Ratio": sharpe_ratio(r, rf_rate),
 "Max Drawdown": series_max_drawdown(r),
 f"CVaR {int(alpha*100)}%": cvar_historical(r, alpha),
 }
 )
asset_df = pd.DataFrame(asset_rows)
st.dataframe(
 asset_df.style.format(
 {
 "Annualized Return": "{:.2%}",
 "Annualized Volatility": "{:.2%}",
 "Sharpe Ratio": "{:.2f}",
 "Max Drawdown": "{:.2%}",
 f"CVaR {int(alpha*100)}%": "{:.2%}",
 }
 ),
 use_container_width=True,
)
# -------------------------------------------------
# Scatter plot: Risk vs Return
# -------------------------------------------------
st.subheader("Risk–return scatter plot (assets)")
scatter_fig = px.scatter(
 asset_df,
 x="Annualized Volatility",
 y="Annualized Return",
 text="Asset",
 hover_data=["Sharpe Ratio", "Max Drawdown"],
 labels={
 "Annualized Volatility": "Risk (volatility)",
 "Annualized Return": "Return",
 },
)
scatter_fig.update_traces(textposition="top center")
st.plotly_chart(scatter_fig, use_container_width=True)
# -------------------------------------------------
# Custom portfolio vs benchmarks
# -------------------------------------------------
st.subheader("Custom portfolio vs benchmarks")
if weight_mode == "Equal weight":
 weights = [1 / len(selected_tickers)] * len(selected_tickers)
else:
 weights = [custom_weights[t] for t in selected_tickers]
 if sum(weights) == 0:
 st.error("All custom weights are zero. Adjust them in the
sidebar.")
 st.stop()
portfolio_info, port_returns = portfolio_metrics(
 returns[selected_tickers],
 weights,
 risk_free_rate=rf_rate,
 alpha=alpha,
)
benchmarks = {}
for bench in ["SPY", "QQQ"]:
 if bench in returns.columns:
 bm_info, bm_ret = portfolio_metrics(
 returns[[bench]],
 [1.0],
 risk_free_rate=rf_rate,
 alpha=alpha,
 )
 benchmarks[bench] = (bm_info, bm_ret)
rows = [{"Portfolio": "Custom portfolio", **portfolio_info}]
for name, (info, _) in benchmarks.items():
 rows.append({"Portfolio": name, **info})
port_df = pd.DataFrame(rows)
st.dataframe(
 port_df.style.format(
 {
 "Annualized Return": "{:.2%}",
 "Annualized Volatility": "{:.2%}",
 "Sharpe Ratio": "{:.2f}",
 "Max Drawdown": "{:.2%}",
 f"CVaR {int(alpha*100)}%": "{:.2%}",
 }
 ),
 use_container_width=True,
)
# -------------------------------------------------
# Cumulative returns
# -------------------------------------------------
st.subheader("Cumulative returns: custom portfolio vs benchmarks")
cum_df = pd.DataFrame()
cum_df["Custom portfolio"] = (1 + port_returns).cumprod()
for name, (_, bm_ret) in benchmarks.items():
 cum_df[name] = (1 + bm_ret).cumprod()
cum_fig = px.line(
 cum_df,
 labels={"value": "Growth of $1", "index": "Date", "variable":
"Portfolio"},
)
st.plotly_chart(cum_fig, use_container_width=True)
# -------------------------------------------------
# Explanation
# -------------------------------------------------
st.markdown("---")
st.markdown(
 f"""
### How to interpret this dashboard
- **Annualized return** – average yearly growth over the selected period.
- **Volatility** – how much returns fluctuate (risk).
- **Sharpe ratio** – return per unit of risk, after subtracting a
{rf_rate:.2%} risk-free rate.
- **Max drawdown** – worst peak-to-trough loss over the period.
- **CVaR {int(alpha*100)}%** – average loss on the very bad days (tail
risk).
Use the sidebar to:
- Change the **time period** and **assets**
- Switch between **equal-weight** and **custom-weight** portfolios
- See how your custom mix compares to benchmarks like **SPY** and **QQQ**
"""
)
