import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plot_utils import JEPI_COLOR, SPY_COLOR, AGG_COLOR, BG_COLOR, SECONDARY_BG, TEXT_COLOR, GRID_COLOR

st.set_page_config(page_title="Das Verdikt — JEPI", page_icon="🎯", layout="wide")

st.markdown("## Einkommen oder Illusion? Unsere Antwort")
st.caption("The Verdict — what the data says about JEPI's yield")
st.divider()

# ── Claim vs reality table ─────────────────────────────────────────────────────
st.subheader("Marketing-Versprechen vs. Empirischer Befund")

verdict_data = {
    "JEPI Marketing-Versprechen": [
        '"8,3 % jährliche Ausschüttung als Einkommen"',
        '"Geringere Volatilität als der S&P 500"',
        '"Geeignet für einkommensorientierte Anleger"',
        '"Aktive Stockpicking-Komponente schafft Alpha"',
    ],
    "Empirischer Befund": [
        "[Team: Füllt aus — z.B.: 'Ausschüttung entspricht zu ~X% der BS-Optionsprämie']",
        "[Team: Füllt aus — z.B.: 'Bestätigt: JEPI-Vol. ~11% vs. SPY ~18% im Beobachtungszeitraum']",
        "[Team: Füllt aus — z.B.: 'Bedingt — bei hoher VRP ja, bei niedrigem VIX fraglich']",
        "[Team: Füllt aus — z.B.: 'Nicht nachgewiesen — Rendite erklärbar ohne Alpha-Annahme']",
    ],
    "Bewertung": ["🟡 Teilweise", "🟢 Bestätigt", "🟡 Bedingt", "🔴 Fraglich"],
}

verdict_df = pd.DataFrame(verdict_data)
st.dataframe(verdict_df, use_container_width=True, hide_index=True, height=200)

st.divider()

# ── Allocation matrix (2×2 heatmap) ──────────────────────────────────────────
st.subheader("Allokationsmatrix: Wann lohnt sich JEPI?")
st.caption("Empfohlene JEPI-Gewichtung je nach Anlagehorizont und VRP-Regime")

matrix_z = [
    [5, 20],   # kurzfristiger Horizont: niedrige / hohe VRP
    [5, 15],   # langfristiger Horizont
]
matrix_text = [
    ["~5 %\n(kaum)", "~15–20 %\n(interessant)"],
    ["~5 %\n(kaum)", "~10–15 %\n(moderat)"],
]

fig_heatmap = go.Figure(go.Heatmap(
    z=matrix_z,
    x=["Niedrige VRP\n(VIX < 15)", "Hohe VRP\n(VIX > 20)"],
    y=["Langfristiger Horizont\n(> 5 Jahre)", "Kurzfristiger Horizont\n(< 2 Jahre)"],
    text=matrix_text,
    texttemplate="%{text}",
    colorscale=[[0, "#1e3a5f"], [0.5, "#f97316"], [1, "#22c55e"]],
    showscale=False,
    textfont=dict(color=TEXT_COLOR, size=15),
))
fig_heatmap.update_layout(
    paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
    font=dict(color=TEXT_COLOR, size=13),
    height=300,
    margin=dict(l=180, r=30, t=30, b=80),
    xaxis=dict(side="bottom"),
)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("""
**Lesehilfe:**
- **Hohe VRP + kurzfristiger Horizont** → JEPI liefert strukturell hohe Prämien; laufende Ausschüttungen kompensieren das fehlende Kurswachstum.
- **Niedrige VRP + langfristiger Horizont** → Aufwärtsbeteiligung wird zu teuer verkauft; reine Aktien‐ oder gemischte Portfolios dominieren.
""")

st.divider()

# ── Limitationen ───────────────────────────────────────────────────────────────
with st.expander("Limitationen unserer Analyse"):
    st.markdown("""
- **Kurze Stichprobe:** JEPI besteht erst seit Mai 2020 — nur ~5 Jahre, kein vollständiger Marktzyklus.
- **Keine Steuerbetrachtung:** Ausschüttungen können steuerlich schlechter behandelt werden als Kursgewinne (je nach Jurisdiktion).
- **ELN-Struktur ≠ Direktoptionen:** JEPI nutzt Equity-Linked Notes statt direkte Optionen; das genaue Pricing der ELNs ist nicht öffentlich.
- **Survivalship Bias:** Wir analysieren nur JEPI, nicht gescheiterte ähnliche ETFs.
- **Makro-Regime:** Der Analysezeitraum umfasst eine ungewöhnlich lockere Geldpolitik (2020–2022) und dann scharfe Zinserhöhungen — VRP-Muster können nicht-stationär sein.
- **Einfondus-Fokus:** Keine Breitenanalyse ähnlicher Produkte (XYLD, RYLD, QYLD).
""")

st.divider()

# ── Final headline callout ─────────────────────────────────────────────────────
st.subheader("Unsere Schlussfolgerung")

st.info(
    "[Team: insert headline finding — z.B.: 'Die 8,3 % Rendite entspricht zu ~X % der "
    "theoretischen Black-Scholes-Optionsprämie. Die Covered-Call-Strategie ist damit keine "
    "magische Einkommensquelle, sondern eine systematische Transformation von Kursrendite "
    "in laufende Ausschüttungen — finanziert durch den Verkauf der Aufwärtsbeteiligung.']"
)

st.divider()

# ── Research question recall ───────────────────────────────────────────────────
st.markdown("### Forschungsfrage — revisited")
st.markdown("""
> *Stellt die Ausschüttungsrendite eines Covered-Call-ETF ökonomisch tatsächlich generiertes
> Einkommen dar — oder lässt sie sich vollständig als systematische Umwandlung der
> Aktienrendite des Anlegers in laufende Auszahlungen rekonstruieren, finanziert durch den
> Verkauf der Aufwärtsbeteiligung?*
""")

st.info("[Team: synthesize final answer to the Forschungsfrage in 3–5 Sätzen]")

st.divider()
st.markdown(
    "<p style='color:#94a3b8; text-align:center;'>"
    "JEPI Analyse — FRA-UAS SoSe 2026 | Portfoliomanagement bei Prof. Benedikt Grimus"
    "</p>",
    unsafe_allow_html=True,
)
