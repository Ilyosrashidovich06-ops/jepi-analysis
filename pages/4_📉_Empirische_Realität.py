import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import (
    load_total_returns, load_prices, load_dividends,
    load_vix, load_risk_free, load_monthly_returns, START_DATE,
)
from utils.portfolio_math import (
    realized_vol, summary_stats, drawdown_series,
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
)
from utils.black_scholes import bs_call_price
from utils.plot_utils import JEPI_COLOR, SPY_COLOR, AGG_COLOR, BG_COLOR, SECONDARY_BG, TEXT_COLOR, GRID_COLOR

st.set_page_config(page_title="Empirische Realität — JEPI", page_icon="📉", layout="wide")

st.markdown("## Empirische Realität: Was sagen die Daten?")
st.caption("JEPI vs. SPY vs. AGG — vollständige Analyse seit Mai 2020")
st.divider()

# ── Load all data ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all():
    jepi_tr = load_total_returns("JEPI")
    spy_tr  = load_total_returns("SPY")
    agg_tr  = load_total_returns("AGG")

    jepi_p  = load_prices("JEPI")
    spy_p   = load_prices("SPY")
    agg_p   = load_prices("AGG")

    jepi_d  = load_dividends("JEPI")
    spy_d   = load_dividends("SPY")

    vix     = load_vix()
    rf      = load_risk_free()

    jepi_m  = load_monthly_returns("JEPI")
    spy_m   = load_monthly_returns("SPY")
    agg_m   = load_monthly_returns("AGG")

    return jepi_tr, spy_tr, agg_tr, jepi_p, spy_p, agg_p, jepi_d, spy_d, vix, rf, jepi_m, spy_m, agg_m

with st.spinner("Lade Marktdaten für alle Ticker…"):
    (jepi_tr, spy_tr, agg_tr,
     jepi_p, spy_p, agg_p,
     jepi_d, spy_d,
     vix, rf,
     jepi_m, spy_m, agg_m) = load_all()

def _close(df):
    if df.empty: return pd.Series(dtype=float)
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s

jepi_close = _close(jepi_p)
spy_close  = _close(spy_p)
agg_close  = _close(agg_p)

rf_latest = float(rf.iloc[-1]) if not rf.empty else 0.045

# ── 1. Kumulierte Gesamtrendite ───────────────────────────────────────────────
st.subheader("1. Kumulierte Gesamtrendite")
st.caption("Reinvestierte Ausschüttungen eingerechnet — Basis = 100 am 20. Mai 2020")

if not jepi_tr.empty and not spy_tr.empty:
    tr = pd.DataFrame({"JEPI": jepi_tr * 100, "SPY": spy_tr * 100}).dropna()
    if not agg_tr.empty:
        tr["AGG"] = agg_tr.reindex(tr.index) * 100

    fig1 = go.Figure()
    for col, color in [("JEPI", JEPI_COLOR), ("SPY", SPY_COLOR), ("AGG", AGG_COLOR)]:
        if col in tr.columns:
            fig1.add_trace(go.Scatter(x=tr.index, y=tr[col], name=col,
                                      line=dict(color=color, width=2.5)))

    # Annotate 2022 drawdown
    dd_dates = tr.index[(tr.index.year == 2022) & (tr.index.month == 10)]
    if len(dd_dates) > 0:
        dd_d = dd_dates[0]
        fig1.add_annotation(x=dd_d, y=float(tr.loc[dd_d, "SPY"]),
                            text="2022 Drawdown", showarrow=True, arrowhead=2,
                            ax=40, ay=-50, font=dict(color=TEXT_COLOR, size=11),
                            bgcolor=SECONDARY_BG, bordercolor=GRID_COLOR, borderwidth=1)
    # Rally annotation
    rl_dates = tr.index[(tr.index.year == 2024) & (tr.index.month == 1)]
    if len(rl_dates) > 0:
        rl_d = rl_dates[0]
        fig1.add_annotation(x=rl_d, y=float(tr.loc[rl_d, "SPY"]),
                            text="2023–2024 Rally", showarrow=True, arrowhead=2,
                            ax=-40, ay=-50, font=dict(color=TEXT_COLOR, size=11),
                            bgcolor=SECONDARY_BG, bordercolor=GRID_COLOR, borderwidth=1)

    fig1.update_layout(paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
                       font=dict(color=TEXT_COLOR), height=400, hovermode="x unified",
                       xaxis=dict(gridcolor=GRID_COLOR),
                       yaxis=dict(gridcolor=GRID_COLOR, title="Index (Basis = 100)"),
                       legend=dict(bgcolor="rgba(30,41,59,0.9)", bordercolor=GRID_COLOR, borderwidth=1),
                       margin=dict(l=60, r=30, t=20, b=60))
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Die Gesamtrendite berücksichtigt reinvestierte Dividenden. JEPI hinkt beim Kurswachstum hinterher — aber die monatlichen Ausschüttungen mildern den Unterschied.")

st.divider()

# ── 2. Rollierende Volatilität ─────────────────────────────────────────────────
st.subheader("2. Rollierende Volatilität (30 Tage)")
st.caption("Annualisierte realisierte Volatilität — JEPI vs. SPY")

if not jepi_close.empty and not spy_close.empty:
    jepi_rv = realized_vol(jepi_close.pct_change().dropna()) * 100
    spy_rv  = realized_vol(spy_close.pct_change().dropna()) * 100

    df_rv = pd.DataFrame({"JEPI": jepi_rv, "SPY": spy_rv}).dropna()
    fig2 = go.Figure()
    for col, color in [("JEPI", JEPI_COLOR), ("SPY", SPY_COLOR)]:
        fig2.add_trace(go.Scatter(x=df_rv.index, y=df_rv[col], name=f"{col} Vol.",
                                  line=dict(color=color, width=2)))
    fig2.update_layout(paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
                       font=dict(color=TEXT_COLOR), height=360, hovermode="x unified",
                       xaxis=dict(gridcolor=GRID_COLOR),
                       yaxis=dict(gridcolor=GRID_COLOR, title="Volatilität ann. (%)", ticksuffix="%"),
                       legend=dict(bgcolor="rgba(30,41,59,0.9)", bordercolor=GRID_COLOR, borderwidth=1),
                       margin=dict(l=60, r=30, t=20, b=60))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("JEPI weist deutlich geringere Schwankungen auf — der Covered-Call-Overlay dämpft die Volatilität strukturell.")

st.divider()

# ── 3. Capture-Ratio Scatter ──────────────────────────────────────────────────
st.subheader("3. Capture-Ratio — Der zentrale Chart der Präsentation")
st.caption("Jeder Punkt = ein Kalendermonat. X = SPY-Rendite, Y = JEPI-Rendite")

if not jepi_m.empty and not spy_m.empty:
    scatter_df = pd.DataFrame({"SPY": spy_m, "JEPI": jepi_m}).dropna()
    scatter_df["color"] = np.where(scatter_df["SPY"] > 0, SPY_COLOR, "#ef4444")

    up   = scatter_df[scatter_df["SPY"] > 0]
    down = scatter_df[scatter_df["SPY"] <= 0]

    fig3 = go.Figure()
    # 45° line
    lim = max(abs(scatter_df["SPY"].min()), abs(scatter_df["SPY"].max())) * 1.1
    fig3.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim],
                              mode="lines", line=dict(color="#94a3b8", width=1, dash="dot"),
                              name="45°-Linie (1:1)", showlegend=True))

    # Scatter points
    for label, subset, color in [("Positiver SPY-Monat", up, SPY_COLOR),
                                   ("Negativer SPY-Monat", down, "#ef4444")]:
        fig3.add_trace(go.Scatter(
            x=subset["SPY"] * 100, y=subset["JEPI"] * 100,
            mode="markers",
            name=label,
            marker=dict(color=color, size=8, opacity=0.75),
            text=subset.index.strftime("%b %Y"),
            hovertemplate="<b>%{text}</b><br>SPY: %{x:.1f}%<br>JEPI: %{y:.1f}%<extra></extra>",
        ))

    # OLS fits
    for subset, color, label in [(up, "#22c55e", "OLS (Aufwärtsmonate)"),
                                   (down, "#f59e0b", "OLS (Abwärtsmonate)")]:
        if len(subset) >= 3:
            m, b, *_ = stats.linregress(subset["SPY"], subset["JEPI"])
            xs = np.array([subset["SPY"].min(), subset["SPY"].max()])
            ys = m * xs + b
            fig3.add_trace(go.Scatter(x=xs * 100, y=ys * 100,
                                      mode="lines", line=dict(color=color, width=2.5, dash="dash"),
                                      name=label))

    # Current slider point
    fig3.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
        font=dict(color=TEXT_COLOR), height=480, hovermode="closest",
        xaxis=dict(gridcolor=GRID_COLOR, title="SPY Monatsrendite (%)", ticksuffix="%", zeroline=True, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, title="JEPI Monatsrendite (%)", ticksuffix="%", zeroline=True, zerolinecolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(30,41,59,0.9)", bordercolor=GRID_COLOR, borderwidth=1),
        margin=dict(l=60, r=30, t=20, b=60),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "Die Asymmetrie ist deutlich sichtbar: In negativen Marktmonaten folgt JEPI dem S&P fast 1:1 — "
        "in starken Aufwärtsmonaten kapped der Covered Call die Partizipation (flachere OLS-Linie rechts)."
    )

st.divider()

# ── 4. Drawdown ───────────────────────────────────────────────────────────────
st.subheader("4. Maximaler Drawdown")
st.caption("Peak-to-Trough-Verlust von JEPI und SPY")

if not jepi_close.empty and not spy_close.empty:
    jepi_dd = drawdown_series(jepi_close) * 100
    spy_dd  = drawdown_series(spy_close)  * 100

    fig4 = go.Figure()
    for name, series, color in [("SPY", spy_dd, SPY_COLOR), ("JEPI", jepi_dd, JEPI_COLOR)]:
        fig4.add_trace(go.Scatter(
            x=series.index, y=series,
            name=name, line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=color + "20",
        ))
    fig4.update_layout(paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
                       font=dict(color=TEXT_COLOR), height=360, hovermode="x unified",
                       xaxis=dict(gridcolor=GRID_COLOR),
                       yaxis=dict(gridcolor=GRID_COLOR, title="Drawdown (%)", ticksuffix="%"),
                       legend=dict(bgcolor="rgba(30,41,59,0.9)", bordercolor=GRID_COLOR, borderwidth=1),
                       margin=dict(l=60, r=30, t=20, b=60))
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("JEPI verzeichnete geringere maximale Verluste als der S&P 500 — ein echter Vorteil für risikoscheue Anleger.")

st.divider()

# ── 5. Rendite-Dekomposition ──────────────────────────────────────────────────
st.subheader("5. Geschätzte Monatsrendite-Dekomposition (JEPI)")
st.caption("Equity-Sleeve + BS-Optionsprämie + Upside-Cap-Kosten (Schätzung)")

if not spy_m.empty and not jepi_m.empty:
    vix_monthly = vix.resample("ME").mean() / 100
    rf_monthly  = rf.resample("ME").mean()
    spy_month_price = spy_close.resample("ME").first()

    decomp_rows = []
    for date in spy_m.index:
        spy_ret = float(spy_m.loc[date])
        eq_contrib = spy_ret * 0.85  # ~85% equity exposure
        try:
            s_val  = float(spy_month_price.asof(date))
            iv_val = float(vix_monthly.asof(date))
            rf_val = float(rf_monthly.asof(date))
            prem   = bs_call_price(s_val, s_val * 1.02, 1/12, rf_val, iv_val) / s_val * 0.20
        except Exception:
            prem = 0.0
        upside_giveway = -max(0, spy_ret - 0.02) * 0.20
        decomp_rows.append({
            "Datum": date,
            "Aktien (~85%)": eq_contrib * 100,
            "Optionsprämie (BS)": prem * 100,
            "Upside-Verlust (~20%)": upside_giveway * 100,
        })

    decomp_df = pd.DataFrame(decomp_rows).set_index("Datum")

    fig5 = go.Figure()
    colors_decomp = [SPY_COLOR, "#22c55e", "#ef4444"]
    for col, color in zip(decomp_df.columns, colors_decomp):
        fig5.add_trace(go.Bar(x=decomp_df.index, y=decomp_df[col], name=col,
                              marker_color=color))
    fig5.update_layout(
        barmode="relative",
        paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
        font=dict(color=TEXT_COLOR), height=420, hovermode="x unified",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, title="Renditebeitrag (%)", ticksuffix="%"),
        legend=dict(bgcolor="rgba(30,41,59,0.9)", bordercolor=GRID_COLOR, borderwidth=1),
        margin=dict(l=60, r=30, t=20, b=60),
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("⚠️ Schätzwerte. Equity-Sleeve = SPY-Rendite × 0,85; Prämie = BS-Preis × 0,20 ELN-Gewicht; Upside-Verlust = max(0, SPY-Return − 2 %) × 0,20.")

st.divider()

# ── 6. Distributions overlay ──────────────────────────────────────────────────
st.subheader("6. JEPI-Kurs mit monatlichen Ausschüttungen")
st.caption("Ausschüttungen treten aus dem NAV aus — die monatlichen Balken zeigen das")

if not jepi_close.empty and not jepi_d.empty:
    jepi_d_aligned = jepi_d.copy()
    jepi_d_aligned.index = pd.to_datetime(jepi_d_aligned.index).tz_localize(None)
    jepi_monthly_div = jepi_d_aligned.resample("ME").sum()

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=jepi_close.index, y=jepi_close,
        name="JEPI Kurs (NAV)", line=dict(color=JEPI_COLOR, width=2.5),
        yaxis="y1",
    ))
    fig6.add_trace(go.Bar(
        x=jepi_monthly_div.index, y=jepi_monthly_div,
        name="Monatliche Ausschüttung (USD/Anteil)",
        marker_color="#f97316", opacity=0.6,
        yaxis="y2",
    ))
    fig6.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
        font=dict(color=TEXT_COLOR), height=420, hovermode="x unified",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, title="Kurs (USD)", side="left"),
        yaxis2=dict(gridcolor=GRID_COLOR, title="Ausschüttung (USD)", side="right", overlaying="y"),
        legend=dict(bgcolor="rgba(30,41,59,0.9)", bordercolor=GRID_COLOR, borderwidth=1),
        margin=dict(l=60, r=80, t=20, b=60),
    )
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("Ausschüttungen entstehen nicht aus dem Nichts — sie verringern den NAV unmittelbar nach Ausschüttungsdatum.")

st.divider()

# ── 7. Summary statistics ─────────────────────────────────────────────────────
st.subheader("7. Kennzahlen im Überblick")
st.caption("JEPI vs. SPY vs. 60/40 (SPY/AGG, monatlich rebalanciert) — seit Mai 2020")

if not jepi_close.empty and not spy_close.empty:
    jepi_ret_d = jepi_close.pct_change().dropna()
    spy_ret_d  = spy_close.pct_change().dropna()

    rows = {}
    for name, prices, ret_d in [
        ("JEPI", jepi_close, jepi_ret_d),
        ("SPY",  spy_close,  spy_ret_d),
    ]:
        rows[name] = {
            "Ann. Rendite (%)":    round(ret_d.mean() * 252 * 100, 2),
            "Ann. Volatilität (%)": round(ret_d.std() * np.sqrt(252) * 100, 2),
            "Sharpe Ratio":        round(sharpe_ratio(ret_d, rf_latest), 3),
            "Sortino Ratio":       round(sortino_ratio(ret_d, rf_latest), 3),
            "Max Drawdown (%)":    round(max_drawdown(prices) * 100, 2),
            "Calmar Ratio":        round(calmar_ratio(ret_d, prices), 3),
        }

    # 60/40 portfolio
    if not agg_m.empty and not spy_m.empty:
        common = spy_m.index.intersection(agg_m.index)
        port_m = spy_m.loc[common] * 0.6 + agg_m.loc[common] * 0.4
        port_prices = (1 + port_m).cumprod()
        port_ret_d = port_m
        rows["60/40 (SPY/AGG)"] = {
            "Ann. Rendite (%)":    round(port_ret_d.mean() * 12 * 100, 2),
            "Ann. Volatilität (%)": round(port_ret_d.std() * np.sqrt(12) * 100, 2),
            "Sharpe Ratio":        round(sharpe_ratio(port_ret_d, rf_latest / 12) * np.sqrt(12), 3),
            "Sortino Ratio":       round(sortino_ratio(port_ret_d, rf_latest / 12) * np.sqrt(12), 3),
            "Max Drawdown (%)":    round(max_drawdown(port_prices) * 100, 2),
            "Calmar Ratio":        round(calmar_ratio(port_ret_d, port_prices) * 12, 3),
        }

    stats_df = pd.DataFrame(rows).T
    st.dataframe(stats_df, use_container_width=True)

st.divider()
st.info("[Team: synthesize empirical findings here — was ist das zentrale empirische Ergebnis? Wie gut erklärt die VRP die JEPI-Rendite?]")
