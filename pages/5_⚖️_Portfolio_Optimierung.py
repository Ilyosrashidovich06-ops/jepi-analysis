import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import load_monthly_returns, load_risk_free, load_vix, START_DATE
from utils.portfolio_math import (
    efficient_frontier, max_sharpe_weights, min_variance_weights,
    optimal_weights_given_delta, simulate_portfolio, summary_stats,
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
)
from utils.plot_utils import JEPI_COLOR, SPY_COLOR, AGG_COLOR, BG_COLOR, SECONDARY_BG, TEXT_COLOR, GRID_COLOR

st.set_page_config(page_title="Portfolio-Optimierung — JEPI", page_icon="⚖️", layout="wide")

st.markdown("## Wie viel JEPI sollte in einem optimalen Portfolio sein?")
st.caption("Markowitz Mean-Variance-Optimierung mit 4-Asset-Universum")
st.divider()

ASSETS = ["SPY", "JEPI", "AGG", "Cash"]
COLORS_ASSETS = [SPY_COLOR, JEPI_COLOR, AGG_COLOR, "#a78bfa"]

@st.cache_data(ttl=3600)
def load_returns_all():
    spy_m  = load_monthly_returns("SPY")
    jepi_m = load_monthly_returns("JEPI")
    agg_m  = load_monthly_returns("AGG")
    rf     = load_risk_free()
    rf_m   = rf.resample("ME").mean()
    rf_m.index = pd.to_datetime(rf_m.index).tz_localize(None)
    rf_m.name = "Cash"
    return spy_m, jepi_m, agg_m, rf_m

with st.spinner("Lade Monatsdaten und optimiere Portfolios…"):
    spy_m, jepi_m, agg_m, rf_m = load_returns_all()

rf_scalar = float(rf_m.mean()) * 12  # annualized

# Align all assets on common monthly dates
returns_df = pd.DataFrame({
    "SPY": spy_m, "JEPI": jepi_m, "AGG": agg_m, "Cash": rf_m,
}).dropna()

returns_no_jepi = returns_df[["SPY", "AGG", "Cash"]].dropna()

st.markdown(f"""
**Annahmen:**
- Zeitraum: {START_DATE} bis heute (monatliche Renditen)
- Erwartete Renditen = Stichprobenmittel (annualisiert)
- Kovarianzmatrix = Stichprobenkovarianz (annualisiert)
- Risikofreirate: {rf_scalar*100:.2f}% p.a. (aktueller 1M-Treasury)
- Keine Leerverkäufe, Gewichte ∈ [0, 1], Σ = 1
""")

# ── Efficient Frontier ────────────────────────────────────────────────────────
st.subheader("Effiziente Grenze — mit und ohne JEPI")

with st.spinner("Berechne effiziente Grenzen…"):
    ef_with    = efficient_frontier(returns_df, n_points=60)
    ef_without = efficient_frontier(returns_no_jepi, n_points=60)

    w_ms_with    = max_sharpe_weights(returns_df, rf=rf_scalar)
    w_ms_without = max_sharpe_weights(returns_no_jepi, rf=rf_scalar)
    w_mv_with    = min_variance_weights(returns_df)
    w_mv_without = min_variance_weights(returns_no_jepi)

    mean_with    = returns_df.mean().values * 12
    cov_with     = returns_df.cov().values * 12
    mean_without = returns_no_jepi.mean().values * 12
    cov_without  = returns_no_jepi.cov().values * 12

def port_stats(w, mean, cov):
    r = np.dot(w, mean)
    v = np.sqrt(w @ cov @ w)
    return r, v

ms_r_with, ms_v_with = port_stats(w_ms_with, mean_with, cov_with)
ms_r_without, ms_v_without = port_stats(w_ms_without, mean_without, cov_without)
mv_r_with, mv_v_with = port_stats(w_mv_with, mean_with, cov_with)
mv_r_without, mv_v_without = port_stats(w_mv_without, mean_without, cov_without)

fig_ef = go.Figure()
if not ef_with.empty:
    fig_ef.add_trace(go.Scatter(x=ef_with["vol"]*100, y=ef_with["ret"]*100,
                                mode="lines", name="EF mit JEPI",
                                line=dict(color=JEPI_COLOR, width=2.5)))
if not ef_without.empty:
    fig_ef.add_trace(go.Scatter(x=ef_without["vol"]*100, y=ef_without["ret"]*100,
                                mode="lines", name="EF ohne JEPI",
                                line=dict(color=SPY_COLOR, width=2, dash="dash")))

# Max Sharpe stars
fig_ef.add_trace(go.Scatter(x=[ms_v_with*100], y=[ms_r_with*100],
                             mode="markers", marker=dict(symbol="star", size=18, color=JEPI_COLOR),
                             name="Max-Sharpe (mit JEPI)"))
fig_ef.add_trace(go.Scatter(x=[ms_v_without*100], y=[ms_r_without*100],
                             mode="markers", marker=dict(symbol="star", size=18, color=SPY_COLOR),
                             name="Max-Sharpe (ohne JEPI)"))
# Min Variance diamonds
fig_ef.add_trace(go.Scatter(x=[mv_v_with*100], y=[mv_r_with*100],
                             mode="markers", marker=dict(symbol="diamond", size=14, color=JEPI_COLOR, opacity=0.7),
                             name="Min-Varianz (mit JEPI)"))
fig_ef.add_trace(go.Scatter(x=[mv_v_without*100], y=[mv_r_without*100],
                             mode="markers", marker=dict(symbol="diamond", size=14, color=SPY_COLOR, opacity=0.7),
                             name="Min-Varianz (ohne JEPI)"))

fig_ef.update_layout(paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
                     font=dict(color=TEXT_COLOR), height=480, hovermode="closest",
                     xaxis=dict(gridcolor=GRID_COLOR, title="Volatilität ann. (%)", ticksuffix="%"),
                     yaxis=dict(gridcolor=GRID_COLOR, title="Erwartete Rendite ann. (%)", ticksuffix="%"),
                     legend=dict(bgcolor="rgba(30,41,59,0.9)", bordercolor=GRID_COLOR, borderwidth=1),
                     margin=dict(l=60, r=30, t=20, b=60))
st.plotly_chart(fig_ef, use_container_width=True)
st.caption("Der Stern markiert das maximale Sharpe-Ratio-Portfolio. Wenn die EF mit JEPI weiter links liegt, verbessert JEPI das Risiko-Rendite-Profil.")

st.divider()

# ── Max Sharpe Weights ─────────────────────────────────────────────────────────
st.subheader("Optimale Gewichtungen — Max-Sharpe")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Mit JEPI (4 Assets)**")
    w_labels = ASSETS
    fig_w1 = go.Figure(go.Bar(
        x=[f"{w*100:.1f}%" for w in w_ms_with],
        y=w_labels,
        orientation="h",
        marker_color=COLORS_ASSETS,
        text=[f"{w*100:.1f}%" for w in w_ms_with],
        textposition="auto",
    ))
    fig_w1.update_layout(paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
                          font=dict(color=TEXT_COLOR), height=220,
                          xaxis=dict(gridcolor=GRID_COLOR, ticksuffix="%"),
                          yaxis=dict(gridcolor=GRID_COLOR),
                          margin=dict(l=60, r=20, t=10, b=30))
    st.plotly_chart(fig_w1, use_container_width=True)

with col2:
    st.markdown("**Ohne JEPI (3 Assets)**")
    fig_w2 = go.Figure(go.Bar(
        x=[f"{w*100:.1f}%" for w in w_ms_without],
        y=["SPY", "AGG", "Cash"],
        orientation="h",
        marker_color=[SPY_COLOR, AGG_COLOR, "#a78bfa"],
        text=[f"{w*100:.1f}%" for w in w_ms_without],
        textposition="auto",
    ))
    fig_w2.update_layout(paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
                          font=dict(color=TEXT_COLOR), height=220,
                          xaxis=dict(gridcolor=GRID_COLOR, ticksuffix="%"),
                          yaxis=dict(gridcolor=GRID_COLOR),
                          margin=dict(l=60, r=20, t=10, b=30))
    st.plotly_chart(fig_w2, use_container_width=True)

st.divider()

# ── Risk-Aversion Slider ────────────────────────────────────────────────────────
st.subheader("Optimale Gewichtung nach Risikoaversion (δ)")

delta = st.slider(
    "Risikoaversionskoeffizient δ (1 = risikofreudig, 10 = sehr risikoscheu)",
    min_value=1.0, max_value=10.0, value=3.0, step=0.5,
)
st.caption("Vorsichtigere Anleger (δ hoch) werden stärker in Anleihen und Cash allokieren als risikofreudige.")

w_delta = optimal_weights_given_delta(returns_df, delta=delta)

fig_delta = go.Figure(go.Bar(
    x=ASSETS, y=w_delta * 100,
    marker_color=COLORS_ASSETS,
    text=[f"{w*100:.1f}%" for w in w_delta],
    textposition="outside",
    textfont=dict(color=TEXT_COLOR),
))
fig_delta.update_layout(paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
                         font=dict(color=TEXT_COLOR), height=340,
                         xaxis=dict(gridcolor=GRID_COLOR, title="Asset"),
                         yaxis=dict(gridcolor=GRID_COLOR, title="Gewicht (%)", ticksuffix="%", range=[0, 110]),
                         margin=dict(l=60, r=30, t=20, b=60))
st.plotly_chart(fig_delta, use_container_width=True)

st.divider()

# ── VRP-konditionierte Optimierung ────────────────────────────────────────────
st.subheader("JEPI-Gewicht nach VRP-Quartil")
st.caption("Ändert sich das optimale JEPI-Gewicht je nach Volatilitätsumfeld?")

with st.spinner("Berechne VRP-Quartile…"):
    vix = load_vix()
    from utils.portfolio_math import realized_vol
    from utils.data_loader import load_prices
    spy_p = load_prices("SPY")
    if not spy_p.empty:
        spy_close = spy_p["Close"].copy()
        spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)
        spy_ret_d = spy_close.pct_change().dropna()
        rv_30 = realized_vol(spy_ret_d, 30)
        vix_aligned = vix.reindex(rv_30.index, method="ffill")
        vrp_daily = (vix_aligned / 100 - rv_30).dropna()
        vrp_monthly = vrp_daily.resample("ME").mean()
        vrp_monthly.index = pd.to_datetime(vrp_monthly.index).tz_localize(None)

        returns_vrp = returns_df.copy()
        vrp_aligned = vrp_monthly.reindex(returns_vrp.index, method="ffill").dropna()
        returns_vrp = returns_vrp.loc[vrp_aligned.index]

        quartile_jepi_weights = []
        quartile_labels = ["Q1\n(niedrigste VRP)", "Q2", "Q3", "Q4\n(höchste VRP)"]
        quartiles = pd.qcut(vrp_aligned, 4, labels=False)

        for q in range(4):
            mask = quartiles == q
            subset = returns_vrp[mask.values]
            if len(subset) >= 5:
                w_q = max_sharpe_weights(subset, rf=rf_scalar)
                jepi_w = w_q[ASSETS.index("JEPI")]
            else:
                jepi_w = 0.0
            quartile_jepi_weights.append(jepi_w * 100)

        fig_vrp_q = go.Figure(go.Bar(
            x=["Q1 (niedrigste VRP)", "Q2", "Q3", "Q4 (höchste VRP)"],
            y=quartile_jepi_weights,
            marker_color=[JEPI_COLOR] * 4,
            text=[f"{w:.1f}%" for w in quartile_jepi_weights],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR),
        ))
        fig_vrp_q.update_layout(paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
                                 font=dict(color=TEXT_COLOR), height=360,
                                 xaxis=dict(gridcolor=GRID_COLOR, title="VRP-Quartil"),
                                 yaxis=dict(gridcolor=GRID_COLOR, title="Optimales JEPI-Gewicht (%)", ticksuffix="%", range=[0, 100]),
                                 margin=dict(l=60, r=30, t=20, b=60))
        st.plotly_chart(fig_vrp_q, use_container_width=True)
        st.caption(
            "Erwartet: Höheres JEPI-Gewicht bei hoher VRP (viel zu verdienen), "
            "niedriges bei komprimierter VRP (Prämien zu klein für die Opportunitätskosten)."
        )
    else:
        st.warning("SPY-Preisdaten für VRP-Quartil-Analyse fehlen.")

st.divider()

# ── Backtest ───────────────────────────────────────────────────────────────────
st.subheader("Backtest: drei Portfolios im Vergleich")
st.caption("100 % SPY | 60/40 (SPY/AGG) | 50/30/10/10 (SPY/AGG/JEPI/Cash)")

if not returns_df.empty:
    common_idx = returns_df.dropna().index

    def run_backtest(weights, name):
        cols = ASSETS
        aligned = returns_df[cols].dropna()
        port_ret = aligned @ weights
        cum_wealth = (1 + port_ret).cumprod() * 10_000
        cum_wealth.name = name
        return port_ret, cum_wealth

    w_spy      = np.array([1.0, 0.0, 0.0, 0.0])
    w_6040     = np.array([0.6, 0.0, 0.4, 0.0])
    w_jepi_mix = np.array([0.50, 0.10, 0.30, 0.10])

    ret_spy,   cum_spy   = run_backtest(w_spy,      "100% SPY")
    ret_6040,  cum_6040  = run_backtest(w_6040,     "60/40 (SPY/AGG)")
    ret_jepi,  cum_jepi  = run_backtest(w_jepi_mix, "50/30/10/10 (+JEPI)")

    fig_bt = go.Figure()
    for series, color, name in [
        (cum_spy, SPY_COLOR, "100% SPY"),
        (cum_6040, AGG_COLOR, "60/40 (SPY/AGG)"),
        (cum_jepi, JEPI_COLOR, "50/30/10/10 (+JEPI)"),
    ]:
        fig_bt.add_trace(go.Scatter(x=series.index, y=series,
                                    name=name, line=dict(color=color, width=2.5)))
    fig_bt.update_layout(paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
                          font=dict(color=TEXT_COLOR), height=400, hovermode="x unified",
                          xaxis=dict(gridcolor=GRID_COLOR),
                          yaxis=dict(gridcolor=GRID_COLOR, title="Portfolio-Wert (USD)", tickprefix="$"),
                          legend=dict(bgcolor="rgba(30,41,59,0.9)", bordercolor=GRID_COLOR, borderwidth=1),
                          margin=dict(l=60, r=30, t=20, b=60))
    st.plotly_chart(fig_bt, use_container_width=True)

    # Summary stats table
    bt_rows = {}
    for name, ret, prices in [
        ("100% SPY",        ret_spy,   cum_spy),
        ("60/40",           ret_6040,  cum_6040),
        ("50/30/10/10 JEPI", ret_jepi, cum_jepi),
    ]:
        bt_rows[name] = {
            "Ann. Rendite (%)":    round(ret.mean() * 12 * 100, 2),
            "Ann. Volatilität (%)": round(ret.std() * np.sqrt(12) * 100, 2),
            "Sharpe":              round(sharpe_ratio(ret, rf_scalar / 12) * np.sqrt(12), 3),
            "Max Drawdown (%)":    round(max_drawdown(prices) * 100, 2),
        }
    st.dataframe(pd.DataFrame(bt_rows).T, use_container_width=True)

st.divider()
st.info("[Team: discuss allocation recommendation — Markowitz (1952): Unter welchen Marktbedingungen ist eine JEPI-Beimischung rational? Welche Anlegertypen profitieren?]")
