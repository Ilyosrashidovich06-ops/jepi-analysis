"""
presentation.py — "Die 45-Milliarden-Dollar-Frage"
Portfolio Management SS 2026 · Frankfurt UAS · Prof. Grimus
Run: streamlit run presentation.py
CSVs: ../jepi_monthly.csv and ../jepi_daily.csv (one directory up)
"""
import os, warnings
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RESULTS — alle Zahlen aus dem Paper; KEINE Änderungen vornehmen           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

SPEAKERS = [
    "Leon Ye",
    "Georgios Pelekanos",
    "Tomas Palmer",
    "Ilyos Umurzakov",
]

# Tabelle 1 — Deskriptive Statistik (annualisiert, Mai 2020 – Mai 2026)
T1 = {
    "JEPI": dict(ret=10.58, vol=10.30, skew=-0.110, kurt=-0.29, sharpe=0.74,  mdd=-13.7),
    "SPY":  dict(ret=17.91, vol=15.70, skew=-0.291, kurt=-0.27, sharpe=0.95,  mdd=-24.5),
    "AGG":  dict(ret= 0.23, vol= 6.02, skew= 0.088, kurt= 0.30, sharpe=-0.45, mdd=-18.4),
}

# Tabelle 2 — Zentralfall (2 % OTM, 20 % Overlay)
T2 = dict(corr=0.508, rom=0.572, act_pct=0.710, rec_pct=0.410)

# Tabelle 3 — Sensitivitätsgitter  (moneyness, overlay, corr, rom, is_central)
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

# Tabelle 4 — Renditeprofil
T4 = dict(
    up_cap=0.592, dn_cap=0.594,
    up_beta=0.474, up_r2=0.38, up_n=48,
    dn_beta=0.556, dn_r2=0.50, dn_n=24,
    asymmetry=1.17,
)

# Tabelle 5 — Optimale Gewichte nach VRP-Regime
T5 = [
    # label, N, SPY%, JEPI%, AGG%, Cash%, Sharpe
    ("Gesamtstichprobe",         72, 100.0,  0.0, 0.0,  0.0, 0.95),
    ("Niedrige VRP (≤ Median)",  36,  74.2, 13.6, 0.0, 12.2, 1.77),
    ("Hohe VRP (> Median)",      36,  49.8,  0.0, 0.0, 50.2, 0.05),
    ("Niedrige VRP (Terzil 1)",  24,  53.6, 46.4, 0.0,  0.0, 1.99),
    ("Mittlere VRP (Terzil 2)",  24,   0.0, 96.6, 0.0,  3.4, 0.36),
    ("Hohe VRP (Terzil 3)",      24,  60.8,  0.0, 0.0, 39.2, 0.51),
]

# ── Farben ────────────────────────────────────────────────────────────────────
C_JEPI  = "#003087"
C_SPY   = "#C8102E"
C_AGG   = "#228B22"
C_CASH  = "#888888"
C_RECON = "#E8621A"
BG      = "#0d1117"
BG2     = "#161b22"
BORDER  = "#30363d"
TXT     = "#e6edf3"
MUTED   = "#8b949e"
ACCENT  = "#1f6feb"
GREEN   = "#3fb950"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Die 45-Mrd.-Frage — JEPI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  html,body,[class*="css"]{{font-family:'Segoe UI',system-ui,sans-serif;}}
  .stApp{{background:{BG};color:{TXT};}}
  [data-testid="stSidebar"]{{background:{BG2};}}
  [data-testid="stSidebar"] *{{color:{TXT}!important;}}
  h1{{color:#79c0ff!important;font-weight:800;}}
  h2{{color:#a5d6ff!important;font-weight:700;}}
  h3{{color:{TXT}!important;font-weight:600;}}
  .card{{background:{BG2};border:1px solid {BORDER};border-radius:12px;
         padding:16px 20px;margin:6px 0;}}
  .kpi{{background:linear-gradient(135deg,#1a2744 0%,{BG} 100%);
        border:1px solid {ACCENT};border-radius:10px;
        padding:16px;text-align:center;}}
  .kpi-val{{font-size:2.1rem;font-weight:800;color:#58a6ff;}}
  .kpi-lbl{{font-size:0.80rem;color:{MUTED};margin-top:4px;}}
  .finding{{background:#0d2a1a;border:1px solid #2ea043;
            border-left:4px solid {GREEN};border-radius:8px;
            padding:12px 16px;margin:8px 0;color:#aff1b6;}}
  .warn{{background:#2a1a0d;border:1px solid #d29922;
         border-left:4px solid #f0883e;border-radius:8px;
         padding:12px 16px;margin:8px 0;color:#ffa657;}}
  .tag{{font-size:0.70rem;color:{ACCENT};text-transform:uppercase;
        letter-spacing:3px;font-weight:700;}}
  hr{{border-color:{BORDER};}}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = 0

PAGES = [
    "Titelfolie",
    "Teil 1 · Einleitung & Daten",
    "Teil 2 · Ausschüttungsrekonstruktion",
    "Teil 3 · Renditeprofil",
    "Teil 4 · Markowitz & Fazit",
]

# ── Data loading ──────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_DATA   = os.path.join(_HERE, "data")   # CSVs in data/

@st.cache_data
def load_monthly():
    return pd.read_csv(os.path.join(_DATA, "jepi_monthly.csv"),
                       index_col=0, parse_dates=True)

@st.cache_data
def load_daily():
    return pd.read_csv(os.path.join(_DATA, "jepi_daily.csv"),
                       index_col=0, parse_dates=True)

monthly = load_monthly()
daily   = load_daily()

# ── Computation helpers ───────────────────────────────────────────────────────
def _bs_call(S, K, r, q, sigma, T=1/12):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return float(S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))

@st.cache_data
def compute_recon(_monthly, moneyness=0.02, overlay=0.20):
    rows = []
    for dt, row in _monthly.iterrows():
        sig = row["VIX_ms"] if (not pd.isna(row["VIX_ms"]) and row["VIX_ms"] > 0) else 0.20
        r   = row["RF_m"]   if not pd.isna(row["RF_m"])   else 0.03
        q   = row.get("SPY_div_yield", 0.015)
        q   = 0.015 if pd.isna(q) else q
        prem = _bs_call(1.0, 1.0*(1+moneyness), r, q, sig) * overlay
        rows.append({"date": dt,
                     "actual":        row["JEPI_dist_yield"],
                     "premium":       prem,
                     "equity_div":    q / 12,
                     "reconstructed": prem + q/12})
    df = pd.DataFrame(rows).set_index("date")
    return df.dropna(subset=["actual", "reconstructed"])

@st.cache_data
def compute_annual_returns(_monthly):
    ann = ((1 + _monthly[["JEPI_ret","SPY_ret"]]).groupby(_monthly.index.year).prod() - 1)*100
    ann.columns = ["JEPI","SPY"]
    return ann

@st.cache_data
def compute_frontier(_monthly):
    cols = ["JEPI_ret","SPY_ret","AGG_ret"]
    data = _monthly[cols].dropna()
    mu   = data.mean().values * 12
    Sig  = data.cov().values  * 12
    rf   = float(_monthly["RF_m"].reindex(data.index).ffill().mean())
    n    = len(mu)
    mu_a = np.append(mu, rf)
    Sa   = np.block([[Sig, np.zeros((n,1))], [np.zeros((1,n)), np.array([[0.0]])]])

    def neg_sr(w):
        v = np.sqrt(max(float(w @ Sa @ w), 1e-20))
        return -(float(w @ mu_a) - rf) / v

    rng = np.random.default_rng(42)
    best_w, best_sr = None, -np.inf
    for _ in range(30):
        w0  = rng.dirichlet(np.ones(n+1))
        res = minimize(neg_sr, w0, method="SLSQP",
                       bounds=[(0,1)]*(n+1),
                       constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                       options={"ftol":1e-12,"maxiter":2000})
        if res.success and -res.fun > best_sr:
            best_sr, best_w = -res.fun, res.x

    targets = np.linspace(rf, float(max(mu_a))*0.99, 150)
    fv, fr = [], []
    for tgt in targets:
        res = minimize(lambda w: np.sqrt(max(float(w @ Sa @ w), 1e-20)),
                       x0=np.ones(n+1)/(n+1), method="SLSQP",
                       bounds=[(0,1)]*(n+1),
                       constraints=[
                           {"type":"eq",  "fun": lambda w: w.sum()-1},
                           {"type":"ineq","fun": lambda w, t=tgt: float(w@mu_a) - t},
                       ],
                       options={"ftol":1e-12,"maxiter":2000})
        if res.success:
            fv.append(float(res.fun))
            fr.append(float(res.x @ mu_a))

    v_tang   = float(np.sqrt(float(best_w @ Sa @ best_w)))
    ret_tang = float(best_w @ mu_a)
    vols_ind = np.sqrt(np.diag(Sig))
    return mu, vols_ind, rf, best_w, best_sr, np.array(fv), np.array(fr), v_tang, ret_tang

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.80rem;margin-bottom:0;'>"
        "Portfolio Management SS 2026<br>Frankfurt UAS · Prof. Grimus</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Agenda**")
    sel = st.radio("", PAGES, index=st.session_state.page, label_visibility="collapsed")
    st.session_state.page = PAGES.index(sel)
    st.markdown("---")
    st.markdown("**Referenten**")
    for i, spk in enumerate(SPEAKERS):
        active = (i + 1 == st.session_state.page)
        col = "#58a6ff" if active else MUTED
        st.markdown(
            f"<p style='margin:2px 0;color:{col};font-size:0.86rem;'>"
            f"Teil {i+1}: {spk}</p>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.74rem;'>"
        "72 Monatsbeobachtungen<br>Mai 2020 – Mai 2026</p>",
        unsafe_allow_html=True,
    )

# ── Plotly base layout ────────────────────────────────────────────────────────
def base_layout(height=380, title=""):
    return dict(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color=TXT, size=11),
        xaxis=dict(gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER),
        legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor=BORDER, borderwidth=1),
        margin=dict(l=60, r=20, t=40 if title else 20, b=55),
        hovermode="x unified",
        height=height,
        title=dict(text=title, font=dict(size=13)) if title else None,
    )

# ── Navigation ────────────────────────────────────────────────────────────────
def nav():
    st.markdown("---")
    cl, _, cr = st.columns([1, 6, 1])
    with cl:
        if st.session_state.page > 0 and st.button("← Zurück", use_container_width=True):
            st.session_state.page -= 1
            st.rerun()
    with cr:
        if st.session_state.page < len(PAGES)-1 and st.button(
                "Weiter →", use_container_width=True, type="primary"):
            st.session_state.page += 1
            st.rerun()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 0 — Titelfolie                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def page_title():
    st.markdown(
        f"<h1 style='font-size:2.7rem;text-align:center;margin-bottom:0.2rem;'>"
        "Die 45-Milliarden-Dollar-Frage</h1>"
        f"<p style='font-size:1.2rem;text-align:center;color:#a5d6ff;margin-top:0;'>"
        "Generiert JEPI echtes Einkommen — oder verpackt er Aktienrendite neu?</p>"
        f"<p style='text-align:center;color:{MUTED};font-size:0.88rem;'>"
        "Leon Ye · Georgios Pelekanos · Tomas Palmer · Ilyos Umurzakov<br>"
        "Portfolio Management SS 2026 · Frankfurt UAS · Prof. Grimus</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("45,6 Mrd. USD", "Verwaltetes Vermögen"),
        ("~8,3 %",        "Ausschüttungsrendite p.a."),
        ("72 Monate",     "Untersuchungszeitraum"),
        ("3 Teilfragen",  "Forschungsstruktur"),
    ]
    for col, (val, lbl) in zip([c1,c2,c3,c4], kpis):
        col.markdown(
            f"<div class='kpi'><div class='kpi-val'>{val}</div>"
            f"<div class='kpi-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Total return indexed to 100
    tr = daily[["JEPI_tr","SPY_tr","AGG_tr"]].dropna(subset=["JEPI_tr","SPY_tr"])
    b_j, b_s, b_a = tr["JEPI_tr"].iloc[0], tr["SPY_tr"].iloc[0], tr["AGG_tr"].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tr.index, y=tr["JEPI_tr"]/b_j*100, name="JEPI",
        line=dict(color=C_JEPI, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,48,135,0.09)",
    ))
    fig.add_trace(go.Scatter(
        x=tr.index, y=tr["SPY_tr"]/b_s*100, name="SPY",
        line=dict(color=C_SPY, width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=tr.index, y=tr["AGG_tr"]/b_a*100, name="AGG",
        line=dict(color=C_AGG, width=1.5, dash="dash"), opacity=0.7,
    ))
    lo = base_layout(height=360, title="Gesamtrenditeentwicklung seit Auflage (Mai 2020 = 100)")
    lo["xaxis"]["title"] = "Datum"
    lo["yaxis"]["title"] = "Indexiert (Basis = 100)"
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True)
    nav()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — Einleitung & Datenbasis                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def page_intro():
    st.markdown(f"<p class='tag'>Teil 1 · {SPEAKERS[0]}</p>", unsafe_allow_html=True)
    st.markdown("## Einleitung & Datenbasis")

    col_l, col_r = st.columns([1.05, 0.95])

    with col_l:
        st.markdown("### Was ist JEPI?")
        st.markdown("""
<div class="card">
Der <b>JPMorgan Equity Premium Income ETF (JEPI)</b> kombiniert zwei Renditequellen:
<ol>
  <li><b>Defensives Aktienportfolio</b> — eine S&amp;P-500-nahe Auswahl mit niedrigerem Beta</li>
  <li><b>Covered-Call-Overlay</b> — monatlicher Verkauf von Kaufoptionen auf den S&amp;P 500,
      dessen Prämien als monatliche Ausschüttung weitergeleitet werden</li>
</ol>
Mit ~45,6 Mrd. USD AUM und ~8,3 % Ausschüttungsrendite wurde JEPI zum
<em>schnellstwachsenden aktiven ETF aller Zeiten</em>.
</div>
        """, unsafe_allow_html=True)

        st.markdown("### Forschungsfrage")
        st.info(
            "**Ist die ~8,3 %-Ausschüttungsrendite ökonomisch echtes Einkommen — oder eine "
            "systematische Umwandlung von Kursrendite in laufende Auszahlungen, finanziert "
            "durch den Verkauf der Aufwärtsbeteiligung?**"
        )

        st.markdown("""
<div class="card">
<b>Drei Teilfragen:</b>
<ol>
  <li>Lässt sich die Ausschüttung durch Black-Scholes-Preise der verkauften Calls erklären?</li>
  <li>Welches asymmetrische Renditeprofil resultiert empirisch?</li>
  <li>Wie verändert sich die optimale Markowitz-Allokation mit der Volatilitätsrisikoprämie?</li>
</ol>
<b>Daten:</b> JEPI, SPY, AGG, VIX · 1-Monats-Treasury (FRED) · Mai 2020 – Mai 2026 · 72 Monate
</div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("### Wie funktioniert ein Covered Call?")
        xs   = np.linspace(80, 125, 300)
        S0, K, prem = 100.0, 102.0, 1.5
        pnl_stock = xs - S0
        pnl_call  = -np.maximum(xs - K, 0) + prem
        pnl_cc    = pnl_stock + pnl_call

        fig_cc = go.Figure()
        fig_cc.add_trace(go.Scatter(x=xs, y=pnl_stock, name="Long Aktie",
            line=dict(color=C_SPY, width=1.5, dash="dash")))
        fig_cc.add_trace(go.Scatter(x=xs, y=pnl_call, name="Short Call (+ Prämie)",
            line=dict(color=C_CASH, width=1.5, dash="dot")))
        fig_cc.add_trace(go.Scatter(x=xs, y=pnl_cc, name="Covered Call (gesamt)",
            line=dict(color=C_JEPI, width=3),
            fill="tozeroy", fillcolor="rgba(0,48,135,0.10)"))
        fig_cc.add_vline(x=K, line=dict(color=TXT, width=1, dash="dot"),
                         annotation_text=f"Strike K = {K:.0f}",
                         annotation_font=dict(color=TXT, size=10),
                         annotation_position="top right")
        lo_cc = base_layout(height=290, title="Auszahlungsprofil — Covered Call")
        lo_cc["xaxis"]["title"] = "Kurs des S&P 500 bei Verfall"
        lo_cc["yaxis"]["title"] = "Gewinn / Verlust"
        lo_cc["hovermode"] = "x"
        fig_cc.update_layout(**lo_cc)
        st.plotly_chart(fig_cc, use_container_width=True)

        st.markdown("""
<div class="warn">
<b>Kernmechanismus:</b> Der Verkauf des Calls erzeugt sofortige Prämieneinnahmen —
begrenzt jedoch den Kursgewinn oberhalb des Strikes. Die Prämie <em>stammt aus der eigenen
Aufwärtsbeteiligung des Anlegers</em>.
</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Tabelle 1 — Deskriptive Statistik (annualisiert, 72 Monate)")

    t1_df = pd.DataFrame({
        "Kennzahl": [
            "Ann. Rendite (%)", "Ann. Volatilität (%)", "Schiefe (monatl.)",
            "Überschuss-Kurtosis", "Sharpe Ratio", "Max. Drawdown (%)",
        ],
        "JEPI": [T1["JEPI"]["ret"],  T1["JEPI"]["vol"],  T1["JEPI"]["skew"],
                 T1["JEPI"]["kurt"], T1["JEPI"]["sharpe"],T1["JEPI"]["mdd"]],
        "SPY":  [T1["SPY"]["ret"],   T1["SPY"]["vol"],   T1["SPY"]["skew"],
                 T1["SPY"]["kurt"],  T1["SPY"]["sharpe"], T1["SPY"]["mdd"]],
        "AGG":  [T1["AGG"]["ret"],   T1["AGG"]["vol"],   T1["AGG"]["skew"],
                 T1["AGG"]["kurt"],  T1["AGG"]["sharpe"], T1["AGG"]["mdd"]],
    })

    def _hl_t1(df):
        s = pd.DataFrame("", index=df.index, columns=df.columns)
        s.iloc[4, 1:] = "color:#58a6ff;font-weight:700"
        return s

    st.dataframe(
        t1_df.style.apply(_hl_t1, axis=None)
             .format({"JEPI":"{:.2f}","SPY":"{:.2f}","AGG":"{:.2f}"}),
        hide_index=True, use_container_width=True,
    )
    nav()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — Ausschüttungsrekonstruktion                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def page_recon():
    st.markdown(f"<p class='tag'>Teil 2 · {SPEAKERS[1]}</p>", unsafe_allow_html=True)
    st.markdown("## Ausschüttungsrekonstruktion")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("### Methodik")
        st.markdown("""
<div class="card">
<b>Frage:</b> Welcher Anteil der JEPI-Ausschüttung lässt sich durch den Wert der
verkauften Calls rekonstruieren?
<br><br>
<b>Vorgehen je Monat:</b>
<ol>
  <li>Berechnung des theoretischen Call-Preises nach Black-Scholes
      (Standardmodell zur Optionsbewertung) mit VIX als Volatilitätsschätzer</li>
  <li>Skalierung mit dem Overlay-Anteil → <em>Prämienrendite</em></li>
  <li>Addition der laufenden S&amp;P-500-Dividende des Aktienanteils</li>
  <li>Vergleich mit der tatsächlich gezahlten JEPI-Ausschüttung</li>
</ol>
<b>Zentralfall:</b> Moneyness 2 % OTM · Overlay 20 % · Tenor 1 Monat
</div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("### Tabelle 2 — Zentrale Schätzung")
        m1, m2 = st.columns(2)
        m1.metric("Zeitliche Korrelation", f"{T2['corr']:.3f}",
                  help="Pearson-Korrelation Modell vs. tatsächliche Ausschüttung")
        m2.metric("Mittleres Verhältnis", f"{T2['rom']:.1%}",
                  help="Ø rekonstruierte / tatsächliche Ausschüttung")
        m3, m4 = st.columns(2)
        m3.metric("Ø Tatsächliche Rendite", f"{T2['act_pct']:.3f} %")
        m4.metric("Ø Rekonstruierte Rendite", f"{T2['rec_pct']:.3f} %",
                  delta=f"{T2['rec_pct']-T2['act_pct']:.3f} %")
        st.markdown("""
<div class="finding">
<b>Befund 1:</b> Das Modell erklärt <b>~57 % der durchschnittlichen Ausschüttung</b>.
Die Korrelation von 0,51 zeigt: Hochvolatilitätsmonate erzeugen mehr Prämie —
JEPI <em>glättet</em> die Auszahlungen jedoch aktiv.
</div>
        """, unsafe_allow_html=True)

    # Figure 1
    st.markdown("### Abbildung 1 — Tatsächliche vs. rekonstruierte monatliche Rendite")
    recon_df = compute_recon(monthly)

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=recon_df.index, y=recon_df["actual"]*100,
        name="Tatsächlich (JEPI)", marker_color=C_JEPI, opacity=0.75,
        hovertemplate="%{x|%b %Y}: %{y:.3f} %<extra>Tatsächlich</extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=recon_df.index, y=recon_df["reconstructed"]*100,
        name="Rekonstruiert (BS + Div.)", line=dict(color=C_RECON, width=2.5),
        hovertemplate="%{x|%b %Y}: %{y:.3f} %<extra>Rekonstruiert</extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=recon_df.index, y=recon_df["premium"]*100,
        name="Optionsprämie", line=dict(color="#ffa657", width=1.5, dash="dash"), opacity=0.85,
    ))
    fig1.add_trace(go.Scatter(
        x=recon_df.index, y=recon_df["equity_div"]*100,
        name="Aktiendividende", line=dict(color=C_AGG, width=1.5, dash="dot"), opacity=0.85,
    ))
    fig1.add_annotation(
        x=0.02, y=0.97, xref="paper", yref="paper", showarrow=False,
        text=f"Korrelation: {T2['corr']:.3f}  |  Mittleres Verhältnis: {T2['rom']:.3f}",
        font=dict(size=11, color=TXT), bgcolor=BG2, bordercolor=BORDER, borderwidth=1,
    )
    lo1 = base_layout(height=360, title="")
    lo1["xaxis"]["title"] = "Datum"
    lo1["yaxis"]["title"] = "Monatliche Rendite (% des NAV)"
    lo1["yaxis"]["ticksuffix"] = " %"
    fig1.update_layout(**lo1)
    st.plotly_chart(fig1, use_container_width=True)

    # Table 3
    st.markdown("### Tabelle 3 — Sensitivitätsanalyse (Moneyness × Overlay)")
    t3_rows = []
    for m_lbl, ov_lbl, corr, rom, central in T3:
        tag = "  ← Zentralfall" if central else ""
        t3_rows.append({
            "Moneyness": m_lbl + tag,
            "Overlay":   ov_lbl,
            "Korrelation": f"{corr:.3f}",
            "Mittl. Verhältnis": f"{rom:.3f}",
        })
    t3_df = pd.DataFrame(t3_rows)
    central_idx = next(i for i, r in enumerate(T3) if r[4])

    def _hl_t3(df):
        s = pd.DataFrame("", index=df.index, columns=df.columns)
        s.iloc[central_idx] = "background-color:rgba(31,111,235,0.20);font-weight:700"
        return s

    st.dataframe(t3_df.style.apply(_hl_t3, axis=None), hide_index=True, use_container_width=True)
    nav()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — Renditeprofil                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def page_return():
    st.markdown(f"<p class='tag'>Teil 3 · {SPEAKERS[2]}</p>", unsafe_allow_html=True)
    st.markdown("## Renditeprofil & Asymmetrie")

    col_l, col_r = st.columns([1.1, 0.9])

    with col_l:
        # Figure 2 — annual returns
        st.markdown("### Abbildung 2 — Kalenderjahrrenditen")
        ann = compute_annual_returns(monthly)
        years = [str(y) for y in ann.index]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=years, y=ann["JEPI"], name="JEPI", marker_color=C_JEPI,
            text=[f"{v:.1f}%" for v in ann["JEPI"]], textposition="outside",
            textfont=dict(size=9), offsetgroup=0,
        ))
        fig2.add_trace(go.Bar(
            x=years, y=ann["SPY"], name="SPY", marker_color=C_SPY,
            text=[f"{v:.1f}%" for v in ann["SPY"]], textposition="outside",
            textfont=dict(size=9), offsetgroup=1,
        ))
        fig2.add_hline(y=0, line=dict(color=TXT, width=0.8))
        for yr, note in [("2020","*Mai–Dez"),("2026","*Jan–Mai")]:
            if yr in years:
                fig2.add_annotation(
                    x=yr, y=float(min(ann.loc[int(yr)].min(), 0))-7,
                    text=note, showarrow=False, font=dict(size=8, color=MUTED),
                )
        lo2 = base_layout(height=310, title="")
        lo2["xaxis"]["title"] = "Kalenderjahr"
        lo2["yaxis"]["title"] = "Jahresrendite (%)"
        lo2["yaxis"]["ticksuffix"] = " %"
        lo2["barmode"] = "group"
        lo2["hovermode"] = "x unified"
        fig2.update_layout(**lo2)
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        # Scatter: monthly returns
        st.markdown("### Monatliche Renditen: JEPI vs. SPY")
        m = monthly[["JEPI_ret","SPY_ret"]].dropna() * 100
        up = m["SPY_ret"] > 0

        x_up_range = np.linspace(m.loc[up,  "SPY_ret"].min(), m.loc[up,  "SPY_ret"].max(), 50)
        x_dn_range = np.linspace(m.loc[~up, "SPY_ret"].min(), m.loc[~up, "SPY_ret"].max(), 50)

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=m.loc[up, "SPY_ret"], y=m.loc[up, "JEPI_ret"],
            mode="markers", name=f"SPY ↑ (N={int(up.sum())})",
            marker=dict(color="#4CAF50", size=6, opacity=0.7),
        ))
        fig_sc.add_trace(go.Scatter(
            x=m.loc[~up,"SPY_ret"], y=m.loc[~up,"JEPI_ret"],
            mode="markers", name=f"SPY ↓ (N={int((~up).sum())})",
            marker=dict(color=C_SPY, size=6, opacity=0.7),
        ))
        fig_sc.add_trace(go.Scatter(
            x=x_up_range, y=T4["up_beta"]*x_up_range,
            name=f"β↑ = {T4['up_beta']:.3f}", mode="lines",
            line=dict(color="#4CAF50", width=2, dash="dash"),
        ))
        fig_sc.add_trace(go.Scatter(
            x=x_dn_range, y=T4["dn_beta"]*x_dn_range,
            name=f"β↓ = {T4['dn_beta']:.3f}", mode="lines",
            line=dict(color=C_SPY, width=2, dash="dash"),
        ))
        lo_sc = base_layout(height=310, title="")
        lo_sc["xaxis"]["title"] = "SPY Monatsrendite (%)"
        lo_sc["yaxis"]["title"] = "JEPI Monatsrendite (%)"
        lo_sc["hovermode"] = "closest"
        lo_sc["legend"]["font"] = dict(size=9)
        fig_sc.update_layout(**lo_sc)
        st.plotly_chart(fig_sc, use_container_width=True)

    # Table 4
    st.markdown("### Tabelle 4 — Capture Ratios & Beta-Regression")
    ca, cb = st.columns(2)
    with ca:
        st.dataframe(pd.DataFrame({
            "Kennzahl": ["Aufwärts Capture Ratio (geo.)", "Abwärts Capture Ratio (geo.)"],
            "Wert":     [f"{T4['up_cap']:.3f}",           f"{T4['dn_cap']:.3f}"],
            "Bedeutung": [
                "JEPI fängt ~59 % der SPY-Aufwärtsbewegung",
                "JEPI verliert ~59 % der SPY-Abwärtsbewegung",
            ],
        }), hide_index=True, use_container_width=True)
    with cb:
        st.dataframe(pd.DataFrame({
            "Kennzahl": [
                f"Aufwärts-Beta (N={T4['up_n']}, R²={T4['up_r2']:.2f})",
                f"Abwärts-Beta (N={T4['dn_n']}, R²={T4['dn_r2']:.2f})",
                "Asymmetrie (β↓ / β↑)",
            ],
            "Wert": [
                f"{T4['up_beta']:.3f}",
                f"{T4['dn_beta']:.3f}",
                f"{T4['asymmetry']:.2f}×",
            ],
        }), hide_index=True, use_container_width=True)

    st.markdown("""
<div class="warn">
<b>Befund 2:</b> Aufwärts- und Abwärts-Capture liegen beide bei ~59 % — nahezu symmetrisch.
Das Abwärts-Beta (0,556) übersteigt das Aufwärts-Beta (0,474), Asymmetrie = <b>1,17×</b>.
Im Untersuchungszeitraum bietet JEPI keinen statistisch bedeutsamen Schutz in Abwärtsphasen.
</div>
    """, unsafe_allow_html=True)
    nav()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 4 — Markowitz & Fazit                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def page_markowitz():
    st.markdown(f"<p class='tag'>Teil 4 · {SPEAKERS[3]}</p>", unsafe_allow_html=True)
    st.markdown("## Markowitz-Optimierung & Fazit")

    col_l, col_r = st.columns([0.9, 1.1])

    with col_l:
        st.markdown("### Was ist die VRP?")
        st.markdown(f"""
<div class="card">
<b>Volatilitätsrisikoprämie (VRP)</b> = VIX − Realisierte Volatilität
<br><br>
<ul>
  <li><b>Hohe VRP:</b> Optionen sind teuer relativ zur tatsächlichen Schwankung —
      JEPI kassiert attraktive Prämien</li>
  <li><b>Niedrige VRP:</b> Optionen günstig — Prämien kompensieren das Risiko kaum</li>
</ul>
<b>Hypothese:</b> JEPI ist als Portfoliobaustein attraktiver, wenn die VRP hoch ist.
</div>
        """, unsafe_allow_html=True)

        # VRP bar chart
        vrp = monthly["VRP_m"].dropna() * 100
        med = float(vrp.median())
        colors_vrp = [C_JEPI if v >= med else C_SPY for v in vrp.values]

        fig_vrp = go.Figure()
        fig_vrp.add_trace(go.Bar(
            x=vrp.index, y=vrp.values, marker_color=colors_vrp,
            hovertemplate="%{x|%b %Y}: %{y:.1f} %<extra>VRP</extra>",
            showlegend=False,
        ))
        fig_vrp.add_hline(y=med, line=dict(color=TXT, width=1.5, dash="dash"),
                          annotation_text=f"Median: {med:.1f} %",
                          annotation_font=dict(size=10, color=TXT))
        lo_vrp = base_layout(height=220, title="VRP im Zeitverlauf (VIX − realis. Vola)")
        lo_vrp["xaxis"]["title"] = "Datum"
        lo_vrp["yaxis"]["title"] = "VRP (%)"
        lo_vrp["hovermode"] = "closest"
        fig_vrp.update_layout(**lo_vrp)
        st.plotly_chart(fig_vrp, use_container_width=True)

    with col_r:
        st.markdown("### Tabelle 5 — Max-Sharpe-Gewichte nach VRP-Regime")
        t5_rows = [
            {"Regime": lbl, "N": n,
             "SPY (%)": spy, "JEPI (%)": jepi, "AGG (%)": agg,
             "Cash (%)": cash, "Sharpe": sr}
            for lbl, n, spy, jepi, agg, cash, sr in T5
        ]
        t5_df = pd.DataFrame(t5_rows)

        def _hl_t5(df):
            s = pd.DataFrame("", index=df.index, columns=df.columns)
            for i, row in t5_df.iterrows():
                if row["JEPI (%)"] > 0:
                    s.at[i, "JEPI (%)"] = (
                        "background-color:rgba(0,48,135,0.35);font-weight:700"
                    )
            return s

        st.dataframe(
            t5_df.style.apply(_hl_t5, axis=None).format({
                "SPY (%)":"{:.1f}","JEPI (%)":"{:.1f}",
                "AGG (%)":"{:.1f}","Cash (%)":"{:.1f}","Sharpe":"{:.2f}",
            }),
            hide_index=True, use_container_width=True,
        )

        # Stacked bar: weights by regime
        reg_labels = [r[0] for r in T5]
        fig_w = go.Figure()
        for asset, col_c, vals in [
            ("SPY",  C_SPY,  [r[2] for r in T5]),
            ("JEPI", C_JEPI, [r[3] for r in T5]),
            ("Cash", C_CASH, [r[5] for r in T5]),
        ]:
            fig_w.add_trace(go.Bar(name=asset, x=reg_labels, y=vals, marker_color=col_c))
        lo_w = base_layout(height=210, title="Optimale Allokation je Regime (%)")
        lo_w["barmode"]  = "stack"
        lo_w["xaxis"]["tickfont"] = dict(size=8)
        lo_w["yaxis"]["title"] = "Gewichtung (%)"
        lo_w["yaxis"]["ticksuffix"] = " %"
        lo_w["hovermode"] = "x unified"
        fig_w.update_layout(**lo_w)
        st.plotly_chart(fig_w, use_container_width=True)

        st.markdown("""
<div class="finding">
<b>Befund 3:</b> Gesamtstichprobe: <b>0 % JEPI</b>-Allokation.
Im Niedrig-VRP-Terzil steigt die Gewichtung auf <b>46,4 %</b> (Sharpe: 1,99).
JEPI ist ein konditionaler Diversifikationsbaustein — kein universelles Einkommensinstrument.
</div>
        """, unsafe_allow_html=True)

    # Figure 3 — Efficient frontier
    st.markdown("### Abbildung 3 — Effizienzlinie (Long-only, Gesamtstichprobe)")
    with st.spinner("Berechne Effizienzlinie …"):
        mu, vols_ind, rf, w_tang, sr_tang, fv, fr, v_tang, ret_tang = compute_frontier(monthly)

    fig3 = go.Figure()
    if len(fv) > 1:
        fig3.add_trace(go.Scatter(
            x=np.array(fv)*100, y=np.array(fr)*100, mode="lines",
            name="Effizienzlinie (Long-only)",
            line=dict(color=ACCENT, width=2.5),
        ))
    if v_tang > 0:
        cal_v = np.linspace(0, v_tang*1.8, 60)
        cal_r = rf + (ret_tang - rf)/v_tang * cal_v
        fig3.add_trace(go.Scatter(
            x=cal_v*100, y=cal_r*100, mode="lines",
            name="Capital Allocation Line",
            line=dict(color=MUTED, width=1.5, dash="dash"), opacity=0.8,
        ))

    for name, av, ar, ac in zip(
        ["JEPI","SPY","AGG"],
        vols_ind*100, mu*100,
        [C_JEPI, C_SPY, C_AGG],
    ):
        fig3.add_trace(go.Scatter(
            x=[av], y=[ar], mode="markers+text", name=name,
            text=[name], textposition="top center",
            marker=dict(color=ac, size=10),
        ))

    fig3.add_trace(go.Scatter(
        x=[v_tang*100], y=[ret_tang*100], mode="markers",
        name=f"Tangenzportfolio (Sharpe ≈ {sr_tang:.2f})",
        marker=dict(color="gold", size=16, symbol="star",
                    line=dict(color="black", width=1.5)),
    ))
    fig3.add_trace(go.Scatter(
        x=[0], y=[rf*100], mode="markers+text",
        text=[f"RF ({rf*100:.1f}%)"], textposition="bottom right",
        marker=dict(color=MUTED, size=8, symbol="diamond"), showlegend=False,
    ))

    lo3 = base_layout(height=420, title="")
    lo3["xaxis"]["title"] = "Ann. Volatilität (%)"
    lo3["xaxis"]["ticksuffix"] = " %"
    lo3["yaxis"]["title"] = "Ann. Erwartungsrendite (%)"
    lo3["yaxis"]["ticksuffix"] = " %"
    lo3["hovermode"] = "closest"
    lo3["legend"]["font"] = dict(size=9)
    fig3.update_layout(**lo3)
    st.plotly_chart(fig3, use_container_width=True)

    # Summary conclusions
    st.markdown("---")
    st.markdown("### Gesamtfazit")
    c1, c2, c3 = st.columns(3)
    conclusions = [
        ("Befund 1", "Ausschüttungsrekonstruktion",
         f"Das Black-Scholes-Modell erklärt ~57 % der durchschnittlichen Ausschüttung "
         f"(Korrelation 0,51). Die Prämie ist ökonomisch rekonstruierbar — "
         f"JEPI glättet sie aber aktiv über Volatilitätsphasen hinweg."),
        ("Befund 2", "Renditeprofil",
         f"Capture Ratios ~59 % symmetrisch. Abwärts-Beta (0,556) > Aufwärts-Beta (0,474), "
         f"Asymmetrie 1,17×. Kein bedeutsamer Kapitalschutz im Untersuchungszeitraum."),
        ("Befund 3", "Markowitz-Allokation",
         f"Gesamtstichprobe: 0 % JEPI. Niedrig-VRP-Terzil: 46,4 % JEPI, Sharpe 1,99. "
         f"JEPI ist ein konditionaler Baustein — kein universelles Einkommensinstrument."),
    ]
    for col, (num, title, text) in zip([c1,c2,c3], conclusions):
        col.markdown(
            f"<div class='card' style='border-color:{ACCENT};min-height:170px;'>"
            f"<div style='font-size:0.70rem;color:{ACCENT};text-transform:uppercase;"
            f"letter-spacing:2px;font-weight:700;margin-bottom:4px;'>{num}</div>"
            f"<div style='font-weight:700;margin-bottom:8px;'>{title}</div>"
            f"<div style='font-size:0.84rem;color:{TXT};'>{text}</div></div>",
            unsafe_allow_html=True,
        )
    nav()


# ── Router ────────────────────────────────────────────────────────────────────
{
    0: page_title,
    1: page_intro,
    2: page_recon,
    3: page_return,
    4: page_markowitz,
}[st.session_state.page]()
