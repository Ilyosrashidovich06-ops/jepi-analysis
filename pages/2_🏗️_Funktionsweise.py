import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plot_utils import JEPI_COLOR, SPY_COLOR, BG_COLOR, SECONDARY_BG, TEXT_COLOR, GRID_COLOR

st.set_page_config(page_title="Funktionsweise — JEPI", page_icon="🏗️", layout="wide")

st.markdown("## Was macht JEPI eigentlich mit deinem Geld?")
st.caption("How JEPI works — visual overview for every audience")
st.divider()

# ── Sankey: money flow ────────────────────────────────────────────────────────
st.subheader("Geldfluss im JEPI-ETF")
st.caption("Von der Investition zur monatlichen Ausschüttung")

fig_sankey = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        pad=25, thickness=22,
        label=[
            "Investor",
            "JEPI ETF",
            "Defensive S&P-500-Aktien (~80 %)",
            "Equity-Linked Notes – ELNs (~20 %)",
            "Monatliche Call-Optionsprämie",
            "Ausschüttung an Investor",
        ],
        color=[
            "#6366f1",  # Investor
            JEPI_COLOR,  # JEPI
            SPY_COLOR,   # Aktien
            "#a78bfa",   # ELNs
            "#22c55e",   # Prämie
            JEPI_COLOR,  # Ausschüttung
        ],
        x=[0.0, 0.25, 0.55, 0.55, 0.78, 1.0],
        y=[0.5,  0.5,  0.2,  0.75, 0.75, 0.5],
    ),
    link=dict(
        source=[0, 1, 1, 3, 4],
        target=[1, 2, 3, 4, 5],
        value=[100, 80, 20, 20, 20],
        color=[
            "rgba(249,115,22,0.3)",
            "rgba(59,130,246,0.3)",
            "rgba(167,139,250,0.3)",
            "rgba(34,197,94,0.3)",
            "rgba(249,115,22,0.3)",
        ],
        label=[
            "100 % Anlagekapital",
            "~80 % Aktienportfolio",
            "~20 % in ELNs",
            "Optionsprämien-Einnahmen",
            "Monatliche Ausschüttung",
        ],
    ),
))
fig_sankey.update_layout(
    paper_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR, size=13),
    height=420,
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig_sankey, use_container_width=True)

st.info(
    "**Was sind ELNs?** Equity-Linked Notes sind strukturierte Anleihen, die JPMorgan intern "
    "ausgibt. Sie replizieren den Payoff verkaufter Index-Call-Optionen, ohne dass JEPI selbst "
    "regulierte Optionen im Portfolio halten muss."
)

st.divider()

# ── Comparison table ──────────────────────────────────────────────────────────
st.subheader("JEPI vs. SPY auf einen Blick")

comparison = pd.DataFrame({
    "Metrik": [
        "AUM",
        "Ausschüttungsrendite",
        "Kostenquote (TER)",
        "Beta vs. S&P 500",
        "Strategie",
        "Auflage",
    ],
    "JEPI": [
        "45,6 Mrd. USD",
        "~8,3 %",
        "0,35 %",
        "~0,48",
        "Aktive Aktien + Covered Calls (via ELNs)",
        "Mai 2020",
    ],
    "SPY": [
        "600+ Mrd. USD",
        "~1,3 %",
        "0,09 %",
        "1,00",
        "Passive S&P-500-Replikation",
        "Januar 1993",
    ],
})

st.dataframe(
    comparison.set_index("Metrik"),
    use_container_width=True,
    height=260,
)

st.divider()

# ── Top 10 holdings ───────────────────────────────────────────────────────────
st.subheader("Top-10-Positionen in JEPI")
st.caption("Stand: Mai 2025 (repräsentativer Snapshot — Quelle: JPMorgan Asset Management)")

holdings = pd.DataFrame({
    "Aktie": [
        "Microsoft", "Amazon", "Mastercard", "Visa",
        "Progressive", "Trane Technologies", "ServiceNow",
        "S&P Global", "Meta Platforms", "AbbVie",
    ],
    "Gewichtung (%)": [1.9, 1.8, 1.7, 1.7, 1.6, 1.6, 1.5, 1.5, 1.5, 1.4],
}).sort_values("Gewichtung (%)")

fig_holdings = go.Figure(go.Bar(
    x=holdings["Gewichtung (%)"],
    y=holdings["Aktie"],
    orientation="h",
    marker=dict(
        color=holdings["Gewichtung (%)"],
        colorscale=[[0, "#1e40af"], [1, JEPI_COLOR]],
        showscale=False,
    ),
    text=holdings["Gewichtung (%)"].apply(lambda v: f"{v:.1f}%"),
    textposition="outside",
    textfont=dict(color=TEXT_COLOR),
))
fig_holdings.update_layout(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=SECONDARY_BG,
    font=dict(color=TEXT_COLOR),
    xaxis=dict(gridcolor=GRID_COLOR, ticksuffix="%", title="Portfoliogewicht"),
    yaxis=dict(gridcolor=GRID_COLOR),
    margin=dict(l=140, r=60, t=20, b=40),
    height=380,
)
st.plotly_chart(fig_holdings, use_container_width=True)

st.divider()

# ── Visual analogy ────────────────────────────────────────────────────────────
st.info(
    "🏠 **Analogy:** Covered Calls verkaufen ist wie deine Dachterrasse vermieten: "
    "Du bekommst monatlich Miete, aber wenn jemand sie für 1 Mio. EUR kaufen will, "
    "kannst du den Deal nicht annehmen — du hast sie bereits vermietet."
)

st.divider()

# ── Interactive payoff diagram ────────────────────────────────────────────────
st.subheader("Interaktiver Payoff-Vergleich: JEPI vs. SPY")
st.caption("Zeigt, wie der Covered-Call die Rendite nach oben begrenzt — und nach unten nur begrenzt schützt")

strike_pct = 2.0  # 2 % OTM strike
slider_val = st.slider(
    "Hypothetische monatliche Aktienrendite (%)",
    min_value=-15.0, max_value=15.0, value=0.0, step=0.5,
    format="%.1f%%",
)

monthly_returns_range = np.linspace(-15, 15, 300)
spy_payoff  = monthly_returns_range
# JEPI approximation: full downside, capped upside (cap at ~2 % + premium received ~0.7 %)
cap = strike_pct + 0.7
jepi_payoff = np.where(monthly_returns_range <= cap, monthly_returns_range - 0.7, cap)
# Slight downside cushion from premium received
jepi_payoff = np.where(monthly_returns_range < 0, monthly_returns_range + 0.7, jepi_payoff)

fig_payoff = go.Figure()
fig_payoff.add_trace(go.Scatter(
    x=monthly_returns_range, y=spy_payoff,
    name="SPY (ungesichert)", line=dict(color=SPY_COLOR, width=2.5, dash="dot"),
))
fig_payoff.add_trace(go.Scatter(
    x=monthly_returns_range, y=jepi_payoff,
    name="JEPI (Covered Call)", line=dict(color=JEPI_COLOR, width=3),
    fill="tonexty", fillcolor="rgba(249,115,22,0.07)",
))

# Mark user slider
spy_at_slider  = slider_val
jepi_at_slider = float(np.interp(slider_val, monthly_returns_range, jepi_payoff))
fig_payoff.add_vline(x=slider_val, line_dash="dash", line_color="#94a3b8", line_width=1)
fig_payoff.add_trace(go.Scatter(
    x=[slider_val, slider_val], y=[spy_at_slider, jepi_at_slider],
    mode="markers",
    marker=dict(size=10, color=[SPY_COLOR, JEPI_COLOR]),
    showlegend=False,
))
fig_payoff.add_annotation(
    x=slider_val, y=max(spy_at_slider, jepi_at_slider) + 1.5,
    text=f"SPY: {spy_at_slider:+.1f}% | JEPI: {jepi_at_slider:+.1f}%",
    showarrow=False, font=dict(color=TEXT_COLOR, size=12),
    bgcolor=SECONDARY_BG, bordercolor=GRID_COLOR, borderwidth=1,
)
fig_payoff.add_vline(x=cap, line_dash="longdash", line_color="#22c55e", line_width=1,
                     annotation_text=f"Cap ≈ +{cap:.1f}%", annotation_position="top right",
                     annotation_font_color="#22c55e")
fig_payoff.update_layout(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=SECONDARY_BG,
    font=dict(color=TEXT_COLOR),
    xaxis=dict(gridcolor=GRID_COLOR, title="Marktrendite (%)", ticksuffix="%"),
    yaxis=dict(gridcolor=GRID_COLOR, title="JEPI-/SPY-Rendite (%)", ticksuffix="%"),
    legend=dict(bgcolor="rgba(30,41,59,0.9)", bordercolor=GRID_COLOR, borderwidth=1),
    margin=dict(l=60, r=30, t=20, b=60),
    height=420,
    hovermode="x unified",
)
st.plotly_chart(fig_payoff, use_container_width=True)

st.caption(
    "Die Asymmetrie ist der Kern der Forschungsfrage: Aufwärtspotenzial wird gekappt, "
    "Abwärtsrisiko bleibt (fast) vollständig erhalten."
)
