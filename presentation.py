"""
presentation.py — "Die 45-Milliarden-Dollar-Frage"
Portfolio Management SS 2026 · Frankfurt UAS · Prof. Grimus
Run: streamlit run presentation.py
"""
import os, warnings
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import plotly.graph_objects as go
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Die 45-Mrd.-Frage — JEPI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS: light/projector-friendly, no sidebar ─────────────────────────────────
st.markdown("""
<style>
  section[data-testid="stSidebar"]          { display:none !important; }
  button[data-testid="collapsedControl"]    { display:none !important; }

  /* Projector-ready typography */
  html, body, [class*="css"] { font-size: 17px !important; }
  h1  { font-size:2.7rem  !important; color:#003087 !important; font-weight:800 !important;
        line-height:1.2 !important; margin-bottom:0.3rem !important; }
  h2  { font-size:2.0rem  !important; color:#003087 !important; font-weight:700 !important;
        margin-top:2rem  !important; }
  h3  { font-size:1.45rem !important; color:#1a1a2e !important; font-weight:600 !important; }
  p, li { font-size:1.05rem !important; line-height:1.75 !important; }

  /* Section divider */
  .sec-rule { border:none; border-top:3px solid #003087; margin:2rem 0 1.5rem 0; }

  /* KPI cards */
  .kpi { background:#eef1f8; border:2px solid #003087; border-radius:12px;
         padding:22px 16px; text-align:center; }
  .kpi-val { font-size:2.3rem; font-weight:800; color:#003087; }
  .kpi-lbl { font-size:0.85rem; color:#4a5568; margin-top:6px; }

  /* Finding & note boxes */
  .find { background:#f0fff4; border-left:5px solid #38a169; border-radius:0 8px 8px 0;
          padding:14px 18px; margin:14px 0; font-size:1.05rem; color:#1a4731; }
  .note { background:#fffbeb; border-left:5px solid #d97706; border-radius:0 8px 8px 0;
          padding:14px 18px; margin:14px 0; font-size:1.05rem; color:#78350f; }
  .info { background:#eff6ff; border-left:5px solid #003087; border-radius:0 8px 8px 0;
          padding:14px 18px; margin:14px 0; font-size:1.05rem; color:#1e3a5f; }

  /* Conclusion cards */
  .conc { background:#ffffff; border:2px solid #003087; border-radius:14px;
          padding:22px; min-height:180px; }
  .conc-num  { font-size:0.75rem; color:#003087; text-transform:uppercase;
               letter-spacing:3px; font-weight:700; margin-bottom:6px; }
  .conc-head { font-size:1.2rem; font-weight:700; color:#1a1a2e; margin-bottom:10px; }
  .conc-body { font-size:1rem; color:#374151; line-height:1.65; }

  /* Author strip */
  .authors { color:#4a5568; font-size:1.0rem; margin-top:0.4rem; }
</style>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RESULTS — alle Zahlen aus dem Paper; KEINE Änderungen vornehmen           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

SPEAKERS = ["Leon Ye", "Georgios Pelekanos", "Tomas Palmer", "Ilyos Umurzakov"]

T1 = {
    "JEPI": dict(ret=10.58, vol=10.30, skew=-0.110, kurt=-0.29, sharpe=0.74,  mdd=-13.7),
    "SPY":  dict(ret=17.91, vol=15.70, skew=-0.291, kurt=-0.27, sharpe=0.95,  mdd=-24.5),
    "AGG":  dict(ret= 0.23, vol= 6.02, skew= 0.088, kurt= 0.30, sharpe=-0.45, mdd=-18.4),
}
T2 = dict(corr=0.508, rom=0.572, act_pct=0.710, rec_pct=0.410)
T3 = [
    ("ATM (0 %)", "15 %", 0.522, 0.646, False),
    ("ATM (0 %)", "20 %", 0.511, 0.812, False),
    ("ATM (0 %)", "25 %", 0.504, 0.977, False),
    ("2 % OTM",   "15 %", 0.519, 0.467, False),
    ("2 % OTM",   "20 %", 0.508, 0.572, True),
    ("2 % OTM",   "25 %", 0.500, 0.678, False),
    ("4 % OTM",   "15 %", 0.519, 0.344, False),
    ("4 % OTM",   "20 %", 0.507, 0.408, False),
    ("4 % OTM",   "25 %", 0.499, 0.472, False),
]
T4 = dict(
    up_cap=0.592, dn_cap=0.594,
    up_beta=0.474, up_r2=0.38, up_n=48,
    dn_beta=0.556, dn_r2=0.50, dn_n=24,
    asymmetry=1.17,
)
T5 = [
    ("Gesamtstichprobe",         72, 100.0,  0.0, 0.0,  0.0, 0.95),
    ("Niedr. VRP (≤ Median)",    36,  74.2, 13.6, 0.0, 12.2, 1.77),
    ("Hohe VRP (> Median)",      36,  49.8,  0.0, 0.0, 50.2, 0.05),
    ("Niedr. VRP (Terzil 1)",    24,  53.6, 46.4, 0.0,  0.0, 1.99),
    ("Mittl. VRP (Terzil 2)",    24,   0.0, 96.6, 0.0,  3.4, 0.36),
    ("Hohe VRP (Terzil 3)",      24,  60.8,  0.0, 0.0, 39.2, 0.51),
]

# ── Colors ────────────────────────────────────────────────────────────────────
CJ = "#003087"; CS = "#C8102E"; CA = "#228B22"; CC = "#888888"
CR = "#E8621A"  # reconstruction line
BG = "#ffffff"; BG2 = "#f4f6fb"; BD = "#d1d5db"

# ── Chart base layout (light) ─────────────────────────────────────────────────
def lo(h=480, title=""):
    return dict(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color="#1a1a2e", size=13),
        xaxis=dict(gridcolor=BD, linecolor="#adb5bd", linewidth=1),
        yaxis=dict(gridcolor=BD, linecolor="#adb5bd", linewidth=1),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=BD, borderwidth=1,
                    font=dict(size=12)),
        margin=dict(l=65, r=25, t=50 if title else 30, b=65),
        hovermode="x unified", height=h,
        title=dict(text=title, font=dict(size=15, color="#003087")) if title else None,
    )

# ── Data loading ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")

@st.cache_data
def load_monthly():
    return pd.read_csv(os.path.join(_DATA, "jepi_monthly.csv"), index_col=0, parse_dates=True)

@st.cache_data
def load_daily():
    return pd.read_csv(os.path.join(_DATA, "jepi_daily.csv"), index_col=0, parse_dates=True)

@st.cache_data(ttl=300)
def load_live():
    try:
        raw = yf.download(["JEPI","SPY"], period="1y", auto_adjust=True, progress=False)
        close = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
        return close[["JEPI","SPY"]].dropna(), None
    except Exception as e:
        return None, str(e)

monthly = load_monthly()
daily   = load_daily()

# ── Computation helpers ───────────────────────────────────────────────────────
def _bs(S, K, r, q, sig, T=1/12):
    if sig <= 0 or T <= 0: return max(S-K, 0.0)
    d1 = (np.log(S/K)+(r-q+0.5*sig**2)*T)/(sig*np.sqrt(T)); d2=d1-sig*np.sqrt(T)
    return float(S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2))

@st.cache_data
def recon_df(_m, moneyness=0.02, overlay=0.20):
    rows=[]
    for dt,row in _m.iterrows():
        sig = row["VIX_ms"] if (not pd.isna(row["VIX_ms"]) and row["VIX_ms"]>0) else 0.20
        r   = row["RF_m"]   if not pd.isna(row["RF_m"]) else 0.03
        q   = 0.015 if pd.isna(row.get("SPY_div_yield", np.nan)) else row["SPY_div_yield"]
        prem = _bs(1.0, 1.0*(1+moneyness), r, q, sig)*overlay
        rows.append({"date":dt, "actual":row["JEPI_dist_yield"],
                     "premium":prem, "equity_div":q/12, "reconstructed":prem+q/12})
    df=pd.DataFrame(rows).set_index("date")
    return df.dropna(subset=["actual","reconstructed"])

@st.cache_data
def annual_returns(_m):
    ann=((1+_m[["JEPI_ret","SPY_ret"]]).groupby(_m.index.year).prod()-1)*100
    ann.columns=["JEPI","SPY"]; return ann

@st.cache_data
def frontier(_m):
    cols=["JEPI_ret","SPY_ret","AGG_ret"]; data=_m[cols].dropna()
    mu=data.mean().values*12; Sig=data.cov().values*12
    rf=float(_m["RF_m"].reindex(data.index).ffill().mean())
    n=len(mu); mu_a=np.append(mu,rf)
    Sa=np.block([[Sig,np.zeros((n,1))],[np.zeros((1,n)),np.array([[0.0]])]])
    def ns(w):
        v=max(float(w@Sa@w)**0.5,1e-10); return -(float(w@mu_a)-rf)/v
    best_sr, best_w = -np.inf, None
    rng = np.random.default_rng(42)
    for _ in range(30):
        w0  = rng.dirichlet(np.ones(n+1))
        res = minimize(ns, w0, method="SLSQP", bounds=[(0,1)]*(n+1),
                       constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                       options={"ftol":1e-12,"maxiter":2000})
        if res.success and -res.fun > best_sr:
            best_sr = -res.fun
            best_w  = res.x
    targets=np.linspace(rf,float(max(mu_a))*0.99,150); fv,fr=[],[]
    for tgt in targets:
        r2=minimize(lambda w:float(w@Sa@w)**0.5,x0=np.ones(n+1)/(n+1),method="SLSQP",
                    bounds=[(0,1)]*(n+1),
                    constraints=[{"type":"eq","fun":lambda w:w.sum()-1},
                                  {"type":"ineq","fun":lambda w,t=tgt:float(w@mu_a)-t}],
                    options={"ftol":1e-12,"maxiter":2000})
        if r2.success: fv.append(float(r2.fun)); fr.append(float(r2.x@mu_a))
    vols=np.sqrt(np.diag(Sig))
    vt=float(np.sqrt(float(best_w@Sa@best_w))); rt=float(best_w@mu_a)
    return mu,vols,rf,best_w,best_sr,np.array(fv),np.array(fr),vt,rt

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LAYOUT — scrollable single page                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── HEADER ────────────────────────────────────────────────────────────────────
logo_path = os.path.join(_HERE, "fra_logo.png")
col_logo, col_mid, col_right = st.columns([1.2, 3, 1])
with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=190)
with col_mid:
    st.markdown(
        "<h1 style='margin-bottom:0;'>Die 45-Milliarden-Dollar-Frage</h1>"
        "<p style='font-size:1.25rem;color:#003087;margin-top:4px;'>"
        "Generiert JEPI echtes Einkommen — oder verpackt er Aktienrendite neu?</p>",
        unsafe_allow_html=True,
    )
with col_right:
    st.markdown(
        "<div style='text-align:right;padding-top:18px;'>"
        "<span style='font-size:0.9rem;color:#4a5568;'>"
        "Portfolio Management SS 2026<br>Frankfurt UAS · Prof. Grimus<br>"
        + " · ".join(SPEAKERS) +
        "</span></div>",
        unsafe_allow_html=True,
    )

st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1,k2,k3,k4 = st.columns(4)
for col,(val,lbl) in zip([k1,k2,k3,k4],[
    ("45,6 Mrd. USD", "Verwaltetes Vermögen (AUM)"),
    ("~8,3 %",        "Ausschüttungsrendite p.a."),
    ("72 Monate",     "Mai 2020 – Mai 2026"),
    ("3 Teilfragen",  "Forschungsstruktur"),
]):
    col.markdown(
        f"<div class='kpi'><div class='kpi-val'>{val}</div>"
        f"<div class='kpi-lbl'>{lbl}</div></div>",
        unsafe_allow_html=True,
    )

# ── EINLEITUNG ────────────────────────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Was ist JEPI?")

col_txt, col_cc = st.columns([1.1, 0.9])
with col_txt:
    st.markdown("""
**JPMorgan Equity Premium Income ETF** — aktiv verwalteter ETF mit zwei Renditequellen:

1. **Defensives S&P-500-Aktienportfolio** — breite Streuung, leicht defensiv
2. **Monatlicher Covered-Call-Overlay** — Verkauf von Kaufoptionen auf den S&P 500;
   die Optionsprämien werden als monatliche Ausschüttung weitergeleitet

Mit **45,6 Mrd. USD** AUM und **~8,3 % Ausschüttungsrendite** wurde JEPI zum
schnellstwachsenden aktiven ETF aller Zeiten — und stellt eine grundlegende Frage:
    """)
    st.markdown("""
<div class="info">
<strong>Forschungsfrage:</strong>
Ist die hohe Ausschüttungsrendite ökonomisch <em>echtes</em> Einkommen —
oder eine systematische Umwandlung von Kursrendite in laufende Auszahlungen,
finanziert durch den Verkauf der Aufwärtsbeteiligung?
</div>
    """, unsafe_allow_html=True)

with col_cc:
    st.markdown("**Auszahlungsprofil eines Covered Call**")
    xs = np.linspace(80, 125, 300)
    S0, K, prem = 100.0, 102.0, 1.5
    fig_cc = go.Figure()
    fig_cc.add_trace(go.Scatter(x=xs, y=xs-S0, name="Long Aktie",
        line=dict(color=CS, width=1.8, dash="dash")))
    fig_cc.add_trace(go.Scatter(x=xs, y=-np.maximum(xs-K,0)+prem, name="Short Call + Prämie",
        line=dict(color=CC, width=1.8, dash="dot")))
    fig_cc.add_trace(go.Scatter(x=xs, y=(xs-S0)+(-np.maximum(xs-K,0)+prem),
        name="Covered Call (gesamt)", line=dict(color=CJ, width=3),
        fill="tozeroy", fillcolor="rgba(0,48,135,0.08)"))
    fig_cc.add_vline(x=K, line=dict(color="#374151", width=1.2, dash="dot"),
                     annotation_text=f"Strike K={K:.0f}",
                     annotation_font=dict(size=12, color="#374151"),
                     annotation_position="top right")
    layout_cc = lo(h=310)
    layout_cc["xaxis"]["title"] = "S&P-500-Kurs bei Verfall"
    layout_cc["yaxis"]["title"] = "Gewinn / Verlust"
    layout_cc["hovermode"] = "x"
    fig_cc.update_layout(**layout_cc)
    st.plotly_chart(fig_cc, use_container_width=True)
    st.markdown("""
<div class="note">
Die Prämie stammt aus der eigenen Aufwärtsbeteiligung des Anlegers — kein "freies" Einkommen.
</div>
    """, unsafe_allow_html=True)

# ── DESKRIPTIVE STATISTIK ─────────────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Datenbasis & Deskriptive Statistik")

t1_df = pd.DataFrame({
    "Kennzahl": [
        "Ann. Rendite (%)", "Ann. Volatilität (%)", "Schiefe (monatl.)",
        "Überschuss-Kurtosis", "Sharpe Ratio", "Max. Drawdown (%)",
    ],
    "JEPI": [T1["JEPI"]["ret"],T1["JEPI"]["vol"],T1["JEPI"]["skew"],
             T1["JEPI"]["kurt"],T1["JEPI"]["sharpe"],T1["JEPI"]["mdd"]],
    "SPY":  [T1["SPY"]["ret"], T1["SPY"]["vol"], T1["SPY"]["skew"],
             T1["SPY"]["kurt"], T1["SPY"]["sharpe"], T1["SPY"]["mdd"]],
    "AGG":  [T1["AGG"]["ret"], T1["AGG"]["vol"], T1["AGG"]["skew"],
             T1["AGG"]["kurt"], T1["AGG"]["sharpe"], T1["AGG"]["mdd"]],
})

def _hl_t1(df):
    s = pd.DataFrame("", index=df.index, columns=df.columns)
    s.iloc[4, 1:] = "background-color:#dbeafe;font-weight:700;color:#003087"
    return s

st.dataframe(
    t1_df.style.apply(_hl_t1, axis=None)
         .format({"JEPI":"{:.2f}","SPY":"{:.2f}","AGG":"{:.2f}"}),
    hide_index=True, use_container_width=True,
)
st.markdown("""
<div class="find">
JEPI bietet ca. <strong>65 % der SPY-Rendite bei nur 66 % der Volatilität</strong>
— aber SPY bleibt nach Sharpe Ratio im Untersuchungszeitraum der stärkere Baustein (0,95 vs. 0,74).
</div>
""", unsafe_allow_html=True)

# ── LIVE JEPI CHART ───────────────────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Live Kursverlauf — JEPI & SPY (letzte 12 Monate)")

live_data, live_err = load_live()

if live_data is not None and not live_data.empty:
    jepi_now = float(live_data["JEPI"].iloc[-1])
    spy_now  = float(live_data["SPY"].iloc[-1])
    jepi_chg = float((live_data["JEPI"].iloc[-1]/live_data["JEPI"].iloc[-2]-1)*100)
    spy_chg  = float((live_data["SPY"].iloc[-1]/live_data["SPY"].iloc[-2]-1)*100)

    mc1,mc2,mc3 = st.columns([1,1,3])
    mc1.metric("JEPI Aktuell", f"${jepi_now:.2f}", f"{jepi_chg:+.2f} % (1T)")
    mc2.metric("SPY Aktuell",  f"${spy_now:.2f}",  f"{spy_chg:+.2f} % (1T)")
    with mc3:
        st.markdown(
            f"<p style='color:#4a5568;font-size:0.9rem;padding-top:18px;'>"
            f"Letzter Datenpunkt: {live_data.index[-1].strftime('%d.%m.%Y')} · "
            f"Automatische Aktualisierung alle 5 Minuten</p>",
            unsafe_allow_html=True,
        )

    # Normalise to 100
    b_j = live_data["JEPI"].iloc[0]; b_s = live_data["SPY"].iloc[0]
    ma50 = (live_data["JEPI"]/b_j*100).rolling(50, min_periods=1).mean()

    fig_live = go.Figure()
    fig_live.add_trace(go.Scatter(
        x=live_data.index, y=live_data["JEPI"]/b_j*100,
        name="JEPI", line=dict(color=CJ, width=2.8),
        fill="tozeroy", fillcolor="rgba(0,48,135,0.07)",
    ))
    fig_live.add_trace(go.Scatter(
        x=live_data.index, y=live_data["SPY"]/b_s*100,
        name="SPY", line=dict(color=CS, width=2.2, dash="dot"),
    ))
    fig_live.add_trace(go.Scatter(
        x=live_data.index, y=ma50,
        name="50-Tage-MA (JEPI)", line=dict(color="#9b59b6", width=1.5, dash="dash"),
        opacity=0.8,
    ))
    ll = lo(h=440, title="Gesamtrendite indexiert (Basis = 100, letzte 12 Monate)")
    ll["xaxis"]["title"] = "Datum"; ll["yaxis"]["title"] = "Indexiert (Basis = 100)"
    fig_live.update_layout(**ll)
    st.plotly_chart(fig_live, use_container_width=True)
else:
    st.warning(f"Live-Daten nicht verfügbar ({live_err}). Bitte Internetverbindung prüfen.")

# ── ABBILDUNG 1: AUSSCHÜTTUNGSREKONSTRUKTION ──────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Abbildung 1 — Ausschüttungsrekonstruktion")

m1a,m1b,m1c = st.columns(3)
m1a.metric("Zeitliche Korrelation",   f"{T2['corr']:.3f}",
           help="Modell vs. tatsächliche Ausschüttung")
m1b.metric("Mittleres Verhältnis",    f"{T2['rom']:.1%}",
           help="Ø Rekonstruiert / Tatsächlich")
m1c.metric("Erklärte Ausschüttung",   "~57 %",
           help="Zentrale Black-Scholes-Schätzung (2 % OTM, 20 % Overlay)")

rc = recon_df(monthly)
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=rc.index, y=rc["actual"]*100, name="Tatsächlich (JEPI)",
    marker_color=CJ, opacity=0.75,
    hovertemplate="%{x|%b %Y}: %{y:.3f} %<extra>Tatsächlich</extra>",
))
fig1.add_trace(go.Scatter(
    x=rc.index, y=rc["reconstructed"]*100, name="Rekonstruiert (BS + Div.)",
    line=dict(color=CR, width=2.8),
    hovertemplate="%{x|%b %Y}: %{y:.3f} %<extra>Rekonstruiert</extra>",
))
fig1.add_trace(go.Scatter(
    x=rc.index, y=rc["premium"]*100, name="Optionsprämie",
    line=dict(color="#f59e0b", width=1.6, dash="dash"), opacity=0.9,
))
fig1.add_trace(go.Scatter(
    x=rc.index, y=rc["equity_div"]*100, name="Aktiendividende",
    line=dict(color=CA, width=1.6, dash="dot"), opacity=0.9,
))
fig1.add_annotation(
    x=0.01, y=0.97, xref="paper", yref="paper", showarrow=False,
    text=f"Korrelation: {T2['corr']:.3f}  |  Verhältnis: {T2['rom']:.3f}",
    font=dict(size=13, color="#1a1a2e"), bgcolor="white",
    bordercolor=BD, borderwidth=1, align="left",
)
l1 = lo(h=460, title="JEPI Tatsächliche vs. Rekonstruierte Ausschüttungsrendite (% des NAV)")
l1["xaxis"]["title"]="Datum"; l1["yaxis"]["title"]="Monatliche Rendite (% des NAV)"
l1["yaxis"]["ticksuffix"]=" %"
fig1.update_layout(**l1)
st.plotly_chart(fig1, use_container_width=True)

col_f1a, col_f1b = st.columns(2)
with col_f1a:
    st.markdown("""
<div class="find">
<strong>Befund 1:</strong> Das Black-Scholes-Modell erklärt ~57 % der durchschnittlichen
Ausschüttung (Korrelation 0,51). Hohe Volatilität → höhere Prämien → höhere Ausschüttung.
JEPI <em>glättet</em> die Auszahlungen jedoch aktiv.
</div>
    """, unsafe_allow_html=True)
with col_f1b:
    t3_df = pd.DataFrame([
        {"Moneyness": m+((" ← Zentral") if c else ""), "Overlay": ov,
         "Korrelation": f"{co:.3f}", "Verhältnis": f"{rm:.3f}"}
        for m,ov,co,rm,c in T3
    ])
    cix = next(i for i,r in enumerate(T3) if r[4])
    def _hl3(df):
        s=pd.DataFrame("",index=df.index,columns=df.columns)
        s.iloc[cix]="background-color:#dbeafe;font-weight:700"; return s
    st.caption("Tabelle 3 — Sensitivitätsgitter")
    st.dataframe(t3_df.style.apply(_hl3,axis=None), hide_index=True, use_container_width=True)

# ── ABBILDUNG 2: JAHRESRENDITEN ───────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Abbildung 2 — Jahresrenditen: JEPI vs. SPY")

ann = annual_returns(monthly)
years = [str(y) for y in ann.index]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=years, y=ann["JEPI"], name="JEPI", marker_color=CJ, offsetgroup=0,
    text=[f"{v:.1f}%" for v in ann["JEPI"]], textposition="outside",
    textfont=dict(size=12, color=CJ),
))
fig2.add_trace(go.Bar(
    x=years, y=ann["SPY"], name="SPY", marker_color=CS, offsetgroup=1,
    text=[f"{v:.1f}%" for v in ann["SPY"]], textposition="outside",
    textfont=dict(size=12, color=CS),
))
fig2.add_hline(y=0, line=dict(color="#374151", width=0.8))
for yr,note in [("2020","* Mai–Dez."),("2026","* Jan.–Mai")]:
    if yr in years:
        fig2.add_annotation(x=yr, y=float(min(ann.loc[int(yr)].min(),0))-9,
            text=note, showarrow=False, font=dict(size=10, color="#6b7280"))
l2 = lo(h=460, title="Kalenderjahrrenditen (Teilperioden 2020* und 2026* beachten)")
l2["xaxis"]["title"]="Kalenderjahr"; l2["yaxis"]["title"]="Jahresrendite (%)"
l2["yaxis"]["ticksuffix"]=" %"; l2["barmode"]="group"; l2["hovermode"]="x unified"
fig2.update_layout(**l2)
st.plotly_chart(fig2, use_container_width=True)

# Capture & beta metrics
cc1,cc2,cc3,cc4 = st.columns(4)
cc1.metric("Aufwärts Capture",  f"{T4['up_cap']:.1%}", help=f"N={T4['up_n']} Monate")
cc2.metric("Abwärts Capture",   f"{T4['dn_cap']:.1%}", help=f"N={T4['dn_n']} Monate")
cc3.metric("Beta (Aufwärts)",   f"{T4['up_beta']:.3f}")
cc4.metric("Beta (Abwärts)",    f"{T4['dn_beta']:.3f}", delta=f"Asymmetrie {T4['asymmetry']:.2f}×")

st.markdown("""
<div class="note">
<strong>Befund 2:</strong> Capture Ratios ~59 % — nahezu <em>symmetrisch</em> auf- und abwärts.
Abwärts-Beta (0,556) übersteigt Aufwärts-Beta (0,474), Asymmetrie = 1,17×.
Kein statistisch bedeutsamer Kapitalschutz in Abwärtsmärkten.
</div>
""", unsafe_allow_html=True)

# ── ABBILDUNG 3: EFFIZIENZLINIE ───────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Abbildung 3 — Markowitz-Effizienzlinie & VRP-Regimes")

col_tbl, col_fig = st.columns([1, 1.7])

with col_tbl:
    st.markdown("**Was ist die VRP?**")
    st.markdown("""
**Volatilitätsrisikoprämie (VRP)** = VIX − Realisierte Volatilität

- **Hohe VRP** → Optionen teuer → JEPI kassiert attraktive Prämien
- **Niedrige VRP** → Prämien kompensieren das Risiko kaum
    """)
    st.markdown("**Tabelle 5 — Optimale Gewichte nach VRP-Regime**")
    t5_df = pd.DataFrame([
        {"Regime":lbl,"N":n,"SPY%":spy,"JEPI%":jepi,"Cash%":cash,"Sharpe":sr}
        for lbl,n,spy,jepi,_,cash,sr in T5
    ])
    def _hl5(df):
        s=pd.DataFrame("",index=df.index,columns=df.columns)
        for i,row in t5_df.iterrows():
            if row["JEPI%"]>0:
                s.at[i,"JEPI%"]="background-color:#dbeafe;font-weight:700;color:#003087"
        return s
    st.dataframe(
        t5_df.style.apply(_hl5,axis=None)
             .format({"SPY%":"{:.1f}","JEPI%":"{:.1f}","Cash%":"{:.1f}","Sharpe":"{:.2f}"}),
        hide_index=True, use_container_width=True,
    )

    # Stacked bar of regime weights
    fig_w = go.Figure()
    for asset,col_c,vals in [
        ("SPY",  CS, [r[2] for r in T5]),
        ("JEPI", CJ, [r[3] for r in T5]),
        ("Cash", CC, [r[5] for r in T5]),
    ]:
        fig_w.add_trace(go.Bar(
            name=asset, x=[r[0] for r in T5], y=vals, marker_color=col_c,
        ))
    lw = lo(h=230, title="")
    lw["barmode"]="stack"; lw["xaxis"]["tickfont"]=dict(size=9)
    lw["yaxis"]["title"]="Gewichtung (%)"; lw["yaxis"]["ticksuffix"]=" %"
    lw["hovermode"]="x unified"; lw["margin"]["b"]=80
    fig_w.update_layout(**lw)
    st.plotly_chart(fig_w, use_container_width=True)

with col_fig:
    with st.spinner("Berechne Effizienzlinie …"):
        mu, vols_i, rf, bw, bsr, fv, fr, vt, rt = frontier(monthly)

    fig3 = go.Figure()
    if len(fv)>1:
        fig3.add_trace(go.Scatter(
            x=np.array(fv)*100, y=np.array(fr)*100, mode="lines",
            name="Effizienzlinie (Long-only)", line=dict(color="#003087", width=2.8),
        ))
    if vt>0:
        cal_v=np.linspace(0,vt*1.8,60)
        fig3.add_trace(go.Scatter(
            x=cal_v*100, y=(rf+(rt-rf)/vt*cal_v)*100, mode="lines",
            name="Capital Allocation Line", line=dict(color="#9b59b6",width=1.8,dash="dash"),
            opacity=0.85,
        ))
    for nm,av,ar,ac in zip(
        ["JEPI","SPY","AGG"], vols_i*100, mu*100, [CJ,CS,CA]
    ):
        fig3.add_trace(go.Scatter(
            x=[av], y=[ar], mode="markers+text", name=nm,
            text=[f"  {nm}"], textposition="middle right",
            marker=dict(color=ac, size=14, line=dict(color="white",width=1.5)),
            textfont=dict(size=13, color=ac),
        ))
    fig3.add_trace(go.Scatter(
        x=[vt*100], y=[rt*100], mode="markers",
        name=f"Tangenzportfolio (Sharpe ≈ {bsr:.2f})",
        marker=dict(color="gold", size=20, symbol="star",
                    line=dict(color="#374151", width=1.5)),
    ))
    fig3.add_trace(go.Scatter(
        x=[0], y=[rf*100], mode="markers+text",
        text=[f"  RF ({rf*100:.1f} %)"], textposition="middle right",
        marker=dict(color=CC, size=10, symbol="diamond"),
        textfont=dict(size=11, color=CC), showlegend=False,
    ))
    l3 = lo(h=500, title="Effizienzlinie: SPY, JEPI, AGG + Cash (Long-only)")
    l3["xaxis"]["title"]="Ann. Volatilität (%)"; l3["xaxis"]["ticksuffix"]=" %"
    l3["yaxis"]["title"]="Ann. Erwartungsrendite (%)"; l3["yaxis"]["ticksuffix"]=" %"
    l3["hovermode"]="closest"; l3["legend"]["font"]=dict(size=11)
    fig3.update_layout(**l3)
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("""
<div class="find">
<strong>Befund 3:</strong> In der Gesamtstichprobe erhält JEPI <strong>0 % Gewichtung</strong>.
Im Niedrig-VRP-Terzil steigt die Allokation auf <strong>46,4 %</strong> (Sharpe 1,99).
JEPI ist ein konditionaler Diversifikationsbaustein — kein universelles Einkommensinstrument.
</div>
""", unsafe_allow_html=True)

# ── GESAMTFAZIT ───────────────────────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Gesamtfazit")

c1,c2,c3 = st.columns(3)
for col,(num,head,body) in zip([c1,c2,c3],[
    ("Befund 1","Ausschüttungsrekonstruktion",
     "Black-Scholes erklärt ~57 % der durchschnittlichen Ausschüttung "
     "(Korrelation 0,51). Die Prämie ist rekonstruierbar — JEPI glättet sie aktiv."),
    ("Befund 2","Renditeprofil & Asymmetrie",
     "Capture Ratios ~59 % — symmetrisch. Abwärts-Beta (0,556) > Aufwärts-Beta (0,474), "
     "Asymmetrie 1,17×. Kein bedeutsamer Kapitalschutz beobachtbar."),
    ("Befund 3","Markowitz-Allokation",
     "0 % JEPI in der Gesamtstichprobe. Niedrig-VRP-Terzil: 46,4 % JEPI, Sharpe 1,99. "
     "JEPI ist ein konditionaler Baustein — kein universelles Einkommensinstrument."),
]):
    col.markdown(
        f"<div class='conc'>"
        f"<div class='conc-num'>{num}</div>"
        f"<div class='conc-head'>{head}</div>"
        f"<div class='conc-body'>{body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<p style='text-align:center;color:#9ca3af;font-size:0.85rem;margin-top:3rem;'>"
    "Frankfurt University of Applied Sciences · Portfolio Management SS 2026 · Prof. Grimus</p>",
    unsafe_allow_html=True,
)
