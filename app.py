import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_total_returns, load_prices
from utils.plot_utils import JEPI_COLOR, SPY_COLOR, BG_COLOR, SECONDARY_BG, TEXT_COLOR, GRID_COLOR, add_annotation

st.set_page_config(
    page_title="JEPI Analyse — FRA-UAS 2026",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Title ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2.6rem; margin-bottom:0;'>Die 45-Milliarden-Dollar-Frage</h1>
<p style='color:#94a3b8; font-size:1.1rem; margin-top:0.2rem;'>
    Is JEPI's yield real income or a repackaged return?
</p>
""", unsafe_allow_html=True)

st.divider()

# ── KPI cards ────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        label="💼 JEPI Verwaltetes Vermögen",
        value="45,6 Mrd. USD",
        delta="Schnellstwachsender aktiver ETF aller Zeiten",
    )
with c2:
    st.metric(
        label="📈 Ausschüttungsrendite",
        value="8,3 %",
        delta="vs. S&P 500: ~1,3 %",
    )
with c3:
    st.metric(
        label="💸 Mittelzuflüsse seit Mai 2020",
        value=">40 Mrd. USD",
        delta="In nur 4 Jahren",
    )

st.divider()

# ── Research question ────────────────────────────────────────────────────────
st.info(
    "**Forschungsfrage:** Stellt die Ausschüttungsrendite eines Covered-Call-ETF ökonomisch "
    "tatsächlich generiertes Einkommen dar — oder lässt sie sich vollständig als systematische "
    "Umwandlung der Aktienrendite des Anlegers in laufende Auszahlungen rekonstruieren, "
    "finanziert durch den Verkauf der Aufwärtsbeteiligung?"
)

st.divider()

# ── Hero chart: JEPI vs SPY total return ────────────────────────────────────
st.subheader("Gesamtrendite seit Auflage (Mai 2020)")
st.caption("Kursentwicklung + reinvestierte Ausschüttungen, indexiert auf 100")

with st.spinner("Lade Marktdaten…"):
    jepi_tr = load_total_returns("JEPI")
    spy_tr  = load_total_returns("SPY")

if not jepi_tr.empty and not spy_tr.empty:
    # Align on common dates
    tr = pd.DataFrame({"JEPI": jepi_tr, "SPY": spy_tr}).dropna()
    tr = tr * 100  # index to 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tr.index, y=tr["JEPI"],
        name="JEPI (Gesamtrendite)",
        line=dict(color=JEPI_COLOR, width=3),
        fill="tozeroy",
        fillcolor="rgba(249,115,22,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=tr.index, y=tr["SPY"],
        name="SPY (Gesamtrendite)",
        line=dict(color=SPY_COLOR, width=2.5, dash="dot"),
        opacity=0.75,
    ))

    # Annotations
    drawdown_date = "2022-10-01"
    rally_date = "2024-01-01"

    if drawdown_date in tr.index.astype(str).tolist():
        dd_idx = tr.index[tr.index.astype(str) == drawdown_date][0]
    else:
        dd_idx = tr.index[tr.index.year == 2022].min() if any(tr.index.year == 2022) else tr.index[len(tr)//2]

    if rally_date in tr.index.astype(str).tolist():
        rally_idx = tr.index[tr.index.astype(str) == rally_date][0]
    else:
        rally_idx = tr.index[tr.index.year == 2024].min() if any(tr.index.year == 2024) else tr.index[-1]

    fig.add_annotation(
        x=dd_idx, y=float(tr.loc[dd_idx, "SPY"]),
        text="2022 Drawdown", showarrow=True,
        arrowhead=2, arrowcolor=TEXT_COLOR, ax=40, ay=-50,
        font=dict(color=TEXT_COLOR, size=11),
        bgcolor=SECONDARY_BG, bordercolor=GRID_COLOR, borderwidth=1,
    )
    fig.add_annotation(
        x=rally_idx, y=float(tr.loc[rally_idx, "SPY"]),
        text="2023–2024 Rally", showarrow=True,
        arrowhead=2, arrowcolor=TEXT_COLOR, ax=-40, ay=-50,
        font=dict(color=TEXT_COLOR, size=11),
        bgcolor=SECONDARY_BG, bordercolor=GRID_COLOR, borderwidth=1,
    )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=SECONDARY_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, ticksuffix=""),
        legend=dict(bgcolor="rgba(30,41,59,0.9)", bordercolor=GRID_COLOR, borderwidth=1),
        margin=dict(l=60, r=30, t=40, b=60),
        hovermode="x unified",
        height=450,
        yaxis_title="Indexiert (Basis = 100)",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Marktdaten konnten nicht geladen werden. Bitte Internetverbindung prüfen.")

st.divider()

# ── Navigation hint ──────────────────────────────────────────────────────────
st.markdown(
    "<p style='color:#94a3b8; text-align:center; font-size:0.95rem;'>"
    "Navigiere mit der Seitenleiste durch die Präsentation →</p>",
    unsafe_allow_html=True,
)
