import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import load_vix, load_risk_free, load_prices, get_latest_price, get_latest_vix, get_latest_rf
from utils.black_scholes import bs_call_price, bs_delta, bs_theta, bs_vega
from utils.portfolio_math import realized_vol
from utils.plot_utils import JEPI_COLOR, SPY_COLOR, BG_COLOR, SECONDARY_BG, TEXT_COLOR, GRID_COLOR

st.set_page_config(page_title="Die Mathematik — JEPI", page_icon="🧮", layout="wide")

st.markdown("## Woher kommt die 8,3 % Rendite wirklich?")
st.caption("Black-Scholes, Volatilitätsrisikoprämie und die Mathematik der Covered Calls")
st.divider()

# ── Black-Scholes formula ─────────────────────────────────────────────────────
st.subheader("Das Black-Scholes-Modell")
st.latex(r"C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)")
col1, col2 = st.columns(2)
with col1:
    st.latex(r"d_1 = \frac{\ln(S/K) + (r + \tfrac{1}{2}\sigma^2)\,T}{\sigma\sqrt{T}}")
with col2:
    st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")

st.markdown("""
**Variablen:**
- $S$ = aktueller Kurs des Basiswerts
- $K$ = Ausübungspreis (Strike)
- $T$ = Restlaufzeit in Jahren
- $r$ = risikofreier Zinssatz
- $\\sigma$ = implizite Volatilität
- $N(\\cdot)$ = kumulative Standardnormalverteilung
""")

st.divider()

# ── Interactive BS calculator ─────────────────────────────────────────────────
st.subheader("Interaktiver Black-Scholes-Rechner")
st.caption("Bewege die Schieberegler — alle Werte aktualisieren sich sofort")

spy_price = get_latest_price("SPY")
vix_val   = get_latest_vix()
rf_val    = get_latest_rf()

if np.isnan(spy_price): spy_price = 540.0
if np.isnan(vix_val):   vix_val   = 18.0
if np.isnan(rf_val):    rf_val    = 0.045

c1, c2, c3 = st.columns(3)
with c1:
    S = st.slider("S — Spot-Preis (USD)", 400.0, 700.0, float(round(spy_price, 0)), step=1.0)
    K = st.slider("K — Strike (USD)", 400.0, 750.0, float(round(spy_price * 1.02, 0)), step=1.0)
with c2:
    sigma = st.slider("σ — Implizite Volatilität (%)", 5.0, 60.0, float(round(vix_val, 1)), step=0.5) / 100
    T = st.slider("T — Restlaufzeit (Tage)", 1, 365, 30, step=1) / 365
with c3:
    r = st.slider("r — Risikofreier Zins (%)", 0.0, 10.0, float(round(rf_val * 100, 2)), step=0.05) / 100

call_price = bs_call_price(S, K, T, r, sigma)
delta      = bs_delta(S, K, T, r, sigma, "call")
theta      = bs_theta(S, K, T, r, sigma, "call")
vega       = bs_vega(S, K, T, r, sigma)

moneyness = (S / K - 1) * 100

r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Call-Preis", f"${call_price:.2f}")
r2.metric("Delta (Δ)", f"{delta:.4f}")
r3.metric("Theta (Θ) / Tag", f"${theta:.4f}")
r4.metric("Vega (ν) / 1%σ", f"${vega:.4f}")
r5.metric("Moneyness", f"{moneyness:+.1f}%", delta="OTM" if S < K else "ITM")

# Call price as % of spot (relevant for JEPI)
premium_pct = call_price / S * 100
st.markdown(
    f"**Prämie als % des Spotpreises:** `{premium_pct:.2f}%` "
    f"→ Annualisiert: `{premium_pct * 12:.1f}%` (bei monatlichen Calls)"
)

st.divider()

# ── VRP section ────────────────────────────────────────────────────────────────
st.subheader("Volatilitätsrisikoprämie (VRP)")
st.caption("Implizite Volatilität (VIX) vs. 30-Tage realisierte Volatilität des S&P 500 seit 2020")

with st.spinner("Lade Volatilitätsdaten…"):
    vix_series    = load_vix()
    spy_prices_df = load_prices("SPY")

if not vix_series.empty and not spy_prices_df.empty:
    spy_close = spy_prices_df["Close"].copy()
    spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)
    spy_ret = spy_close.pct_change().dropna()

    rv_30 = realized_vol(spy_ret, 30).dropna() * 100  # in %

    vix_aligned = vix_series.reindex(rv_30.index, method="ffill").dropna()
    rv_aligned  = rv_30.reindex(vix_aligned.index).dropna()
    vix_aligned = vix_aligned.reindex(rv_aligned.index)

    vrp = vix_aligned - rv_aligned

    fig_vrp = go.Figure()
    fig_vrp.add_trace(go.Scatter(
        x=vix_aligned.index, y=vix_aligned,
        name="VIX (impl. Vol.)", line=dict(color=JEPI_COLOR, width=2),
    ))
    fig_vrp.add_trace(go.Scatter(
        x=rv_aligned.index, y=rv_aligned,
        name="Realisierte Vol. (30d)", line=dict(color=SPY_COLOR, width=2),
    ))
    # Shade VRP
    fig_vrp.add_trace(go.Scatter(
        x=list(vix_aligned.index) + list(rv_aligned.index[::-1]),
        y=list(vix_aligned) + list(rv_aligned[::-1]),
        fill="toself",
        fillcolor="rgba(249,115,22,0.12)",
        line=dict(width=0),
        name="VRP (Differenz)",
        showlegend=True,
    ))

    vrp_mean = vrp.mean()
    vrp_pos_pct = (vrp > 0).mean() * 100
    fig_vrp.add_annotation(
        x=vix_aligned.index[len(vix_aligned)//2],
        y=float(vix_aligned.max()) * 0.9,
        text=f"Ø VRP: {vrp_mean:.1f}% | VRP > 0 in {vrp_pos_pct:.0f}% der Handelstage",
        showarrow=False,
        font=dict(color=TEXT_COLOR, size=12),
        bgcolor=SECONDARY_BG, bordercolor=GRID_COLOR, borderwidth=1,
    )
    fig_vrp.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, ticksuffix="%", title="Volatilität (ann., %)"),
        legend=dict(bgcolor="rgba(30,41,59,0.9)", bordercolor=GRID_COLOR, borderwidth=1),
        margin=dict(l=60, r=30, t=20, b=60),
        height=400, hovermode="x unified",
    )
    st.plotly_chart(fig_vrp, use_container_width=True)

    st.markdown(
        f"> **Ergebnis:** Im Schnitt lag die implizite Volatilität (VIX) **{vrp_mean:.1f} Prozentpunkte** "
        f"über der realisierten Volatilität. In **{vrp_pos_pct:.0f} %** der Handelstage war die VRP positiv. "
        "Diese systematische Differenz ist die strukturelle Renditequelle der verkauften Calls."
    )
else:
    st.warning("Volatilitätsdaten konnten nicht geladen werden.")

st.divider()

# ── BS-predicted vs JEPI actual distributions ─────────────────────────────────
st.subheader("BS-Modellprämie vs. JEPI-Ausschüttung")
st.caption(
    "Monatliche BS-Prämie eines 2 %-OTM-Calls auf SPY (Laufzeit: 1 Monat, σ = VIX/100) "
    "vs. tatsächliche JEPI-Ausschüttung je Anteil"
)

from utils.data_loader import load_dividends, load_prices as _lp

with st.spinner("Berechne BS-Prämien…"):
    spy_monthly = spy_prices_df["Close"].resample("MS").first().copy()
    spy_monthly.index = pd.to_datetime(spy_monthly.index).tz_localize(None)
    vix_monthly = vix_series.resample("MS").first()
    vix_monthly.index = pd.to_datetime(vix_monthly.index).tz_localize(None)

    rf_series = load_risk_free()
    rf_monthly = rf_series.resample("MS").first()
    rf_monthly.index = pd.to_datetime(rf_monthly.index).tz_localize(None)

    common_idx = spy_monthly.index.intersection(vix_monthly.index).intersection(rf_monthly.index)

    bs_premiums = []
    for date in common_idx:
        s  = float(spy_monthly.loc[date])
        k  = s * 1.02
        iv = float(vix_monthly.loc[date]) / 100
        rr = float(rf_monthly.loc[date])
        p  = bs_call_price(s, k, 1/12, rr, iv)
        bs_premiums.append({"date": date, "BS Prämie (% Spot)": p / s * 100})

    bs_df = pd.DataFrame(bs_premiums).set_index("date")

    # JEPI monthly dividends (aggregate per month)
    jepi_divs   = load_dividends("JEPI")
    jepi_prices = load_prices("JEPI")

if not jepi_divs.empty and not jepi_prices.empty:
    jepi_close = jepi_prices["Close"].copy()
    jepi_close.index = pd.to_datetime(jepi_close.index).tz_localize(None)
    jepi_divs.index  = pd.to_datetime(jepi_divs.index).tz_localize(None)

    # Monthly div yield (div / price at start of month)
    monthly_div = jepi_divs.resample("MS").sum()
    monthly_price = jepi_close.resample("MS").first()
    monthly_div_yield = (monthly_div / monthly_price * 100).dropna()
    monthly_div_yield.name = "JEPI Ausschüttungsrendite (% NAV)"

    combined = bs_df.join(monthly_div_yield, how="inner")

    fig_bs = go.Figure()
    fig_bs.add_trace(go.Scatter(
        x=combined.index, y=combined["BS Prämie (% Spot)"],
        name="BS-Modellprämie (2 % OTM Call, 1M)",
        line=dict(color=SPY_COLOR, width=2.5, dash="dot"),
    ))
    fig_bs.add_trace(go.Scatter(
        x=combined.index, y=combined["JEPI Ausschüttungsrendite (% NAV)"],
        name="JEPI Ausschüttungsrendite",
        line=dict(color=JEPI_COLOR, width=2.5),
        fill="tonexty", fillcolor="rgba(249,115,22,0.08)",
    ))
    fig_bs.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=SECONDARY_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, ticksuffix="%", title="% des Kurswerts"),
        legend=dict(bgcolor="rgba(30,41,59,0.9)", bordercolor=GRID_COLOR, borderwidth=1),
        margin=dict(l=60, r=30, t=20, b=60),
        height=380, hovermode="x unified",
    )
    st.plotly_chart(fig_bs, use_container_width=True)
else:
    st.warning("JEPI-Dividendendaten nicht verfügbar.")

st.info("[Team: interpret the fit here — Hull (2021, Kap. 15): Wie gut approximiert das BS-Modell die tatsächliche Prämie? Welche Abweichungen sind systematisch?]")

st.divider()

# ── Glossary ──────────────────────────────────────────────────────────────────
st.subheader("Glossar")
with st.expander("Schlüsselbegriffe erklären"):
    st.markdown("""
**Volatilitätsrisikoprämie (VRP)**
Die Differenz zwischen impliziter Volatilität (vom Markt erwartet) und realisierter Volatilität.
Verkäufer von Optionen kassieren diese Prämie systematisch, weil Käufer bereit sind, für Schutz zu bezahlen.

**Implizite Volatilität (IV)**
Die vom Markt eingepreiste erwartete Schwankungsbreite des Basiswerts über die Laufzeit der Option.
Abgeleitet aus dem Optionspreis mittels Black-Scholes.

**OTM (Out of the Money)**
Eine Call-Option ist OTM, wenn der Strike über dem aktuellen Kurs liegt.
JEPI verkauft typischerweise ~2 % OTM Calls, um etwas Aufwärtspotenzial zu erhalten.

**Equity-Linked Note (ELN)**
Eine strukturierte Anleihe, die den Payoff einer verkauften Call-Option realisiert.
JEPI nutzt ELNs statt direkt Optionen zu handeln — regulatorisch einfacher für einen ETF.

**Theta-Decay**
Der tägliche Wertverlust einer Option durch Zeitablauf (Zeitwertverfall).
Als Optionsverkäufer profitiert JEPI davon: Die Prämie „zerfällt" zugunsten des Verkäufers.
""")
