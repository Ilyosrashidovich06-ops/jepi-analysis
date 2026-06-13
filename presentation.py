"""
presentation.py — "Die 45-Milliarden-Dollar-Frage"
Portfoliomanagement SS 2026 · Frankfurt UAS · Dozent: Benedikt Grimus
Run: streamlit run presentation.py
"""
import os, base64, warnings
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

st.markdown("""
<style>
  section[data-testid="stSidebar"]        { display:none !important; }
  button[data-testid="collapsedControl"]  { display:none !important; }
  html,body,[class*="css"]                { font-size:17px !important; }
  h1  { font-size:2.7rem  !important; color:#79c0ff !important; font-weight:800 !important; }
  h2  { font-size:2.0rem  !important; color:#a5d6ff !important; font-weight:700 !important;
        margin-top:2.2rem !important; }
  h3  { font-size:1.45rem !important; color:#e6edf3 !important; font-weight:600 !important; }
  p,li{ font-size:1.05rem !important; line-height:1.75 !important; }

  .sec-rule { border:none; border-top:2px solid #1f6feb; margin:2rem 0 1.5rem 0; }

  .kpi { background:linear-gradient(135deg,#1a2744 0%,#0d1117 100%);
         border:1px solid #1f6feb; border-radius:12px;
         padding:22px 16px; text-align:center; }
  .kpi-val { font-size:2.3rem; font-weight:800; color:#58a6ff; }
  .kpi-lbl { font-size:0.85rem; color:#8b949e; margin-top:6px; }

  .kpi-box { background:#161b22; border:1px solid #30363d; border-radius:10px;
             padding:18px 16px; text-align:center; margin-bottom:8px; }
  .kpi-box .kpi-label { font-size:0.82rem; color:#8b949e; text-transform:uppercase;
                         letter-spacing:0.05em; margin-bottom:6px; }
  .kpi-box .kpi-value { font-size:1.6rem; font-weight:800; color:#58a6ff; margin-bottom:4px; }
  .kpi-box .kpi-sub   { font-size:0.85rem; color:#aff1b6; }

  .find   { background:#0d2a1a; border-left:5px solid #3fb950; border-radius:0 8px 8px 0;
            padding:14px 18px; margin:12px 0; font-size:1.05rem; color:#aff1b6; }
  .note   { background:#2a1a0d; border-left:5px solid #f0883e; border-radius:0 8px 8px 0;
            padding:14px 18px; margin:12px 0; font-size:1.05rem; color:#ffa657; }
  .info   { background:#0d1f3a; border-left:5px solid #58a6ff; border-radius:0 8px 8px 0;
            padding:14px 18px; margin:12px 0; font-size:1.05rem; color:#bfdbfe; }
  .answer { background:#1a0d2a; border-left:5px solid #d2a8ff; border-radius:0 8px 8px 0;
            padding:16px 20px; margin:18px 0; font-size:1.05rem; color:#e2c8ff; }

  .fq-box { background:linear-gradient(135deg,#0d1f3a 0%,#161b22 100%);
            border:2px solid #1f6feb; border-radius:14px; padding:24px 28px; margin:8px 0; }
  .fq-q   { font-size:1.6rem; font-weight:700; color:#79c0ff;
            margin-bottom:18px; line-height:1.4; }
  .fq-sub { font-size:1.0rem; color:#c9d1d9; margin:8px 0; }
  .fq-num { color:#58a6ff; font-weight:700; margin-right:8px; }

  .conc      { background:#161b22; border:2px solid #1f6feb; border-radius:14px;
               padding:22px; min-height:180px; }
  .conc-num  { font-size:0.73rem; color:#58a6ff; text-transform:uppercase;
               letter-spacing:3px; font-weight:700; margin-bottom:6px; }
  .conc-head { font-size:1.2rem; font-weight:700; color:#e6edf3; margin-bottom:10px; }
  .conc-body { font-size:1.0rem; color:#c9d1d9; line-height:1.65; }

  .fit-box  { background:#161b22; border:1px solid #30363d; border-radius:12px;
              padding:18px; margin:6px 0; min-height:200px; }
  .fit-head { font-size:1.05rem; font-weight:700; margin-bottom:10px; }
  .fit-body { font-size:0.95rem; color:#c9d1d9; line-height:1.65; }

  .logo-box { background:white; padding:8px 10px; border-radius:8px;
              display:inline-block; line-height:0; }
</style>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RESULTS — alle Zahlen aus dem Paper; KEINE Änderungen vornehmen           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

SPEAKERS = ["Leon Ye", "Georgios Pelekanos", "Thomas Palmer", "Ilyos Umurzakov"]

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
T4 = dict(up_cap=0.592, dn_cap=0.594,
          up_beta=0.474, up_r2=0.38, up_n=48,
          dn_beta=0.556, dn_r2=0.50, dn_n=24, asymmetry=1.17)
T5 = [
    ("Gesamtstichprobe",         72, 100.0,  0.0, 0.0,  0.0, 0.95),
    ("Niedr. VRP (≤ Median)",    36,  74.2, 13.6, 0.0, 12.2, 1.77),
    ("Hohe VRP (> Median)",      36,  49.8,  0.0, 0.0, 50.2, 0.05),
    ("Niedr. VRP (Terzil 1)",    24,  53.6, 46.4, 0.0,  0.0, 1.99),
    ("Mittl. VRP (Terzil 2)",    24,   0.0, 96.6, 0.0,  3.4, 0.36),
    ("Hohe VRP (Terzil 3)",      24,  60.8,  0.0, 0.0, 39.2, 0.51),
]

CJ="#003087"; CS="#C8102E"; CA="#228B22"; CC="#888888"; CR="#E8621A"
BG="#0d1117";  BG2="#161b22"; BD="#30363d"; TXT="#e6edf3"; MUTED="#8b949e"

def lo(h=480, title=""):
    d = dict(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color=TXT, size=13),
        xaxis=dict(gridcolor=BD, linecolor="#484f58", linewidth=1, color=TXT),
        yaxis=dict(gridcolor=BD, linecolor="#484f58", linewidth=1, color=TXT),
        legend=dict(bgcolor="rgba(22,27,34,0.92)", bordercolor=BD,
                    borderwidth=1, font=dict(size=12, color=TXT)),
        margin=dict(l=65, r=25, t=50 if title else 30, b=65),
        hovermode="x unified", height=h,
    )
    if title:
        d["title"] = dict(text=title, font=dict(size=15, color="#79c0ff"))
    return d

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
        raw = yf.download(["JEPI","SPY"], start="2020-05-20",
                           auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            return None, "Keine Daten"
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        return close[["JEPI","SPY"]].dropna(), None
    except Exception as e:
        return None, str(e)

monthly = load_monthly()
daily   = load_daily()

def _bs(S, K, r, q, sig, T=1/12):
    if sig <= 0 or T <= 0: return max(S-K, 0.0)
    d1 = (np.log(S/K)+(r-q+0.5*sig**2)*T)/(sig*np.sqrt(T)); d2=d1-sig*np.sqrt(T)
    return float(S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2))

@st.cache_data
def recon_df(_m, moneyness=0.02, overlay=0.20):
    rows = []
    for dt, row in _m.iterrows():
        sig = row["VIX_ms"] if (not pd.isna(row["VIX_ms"]) and row["VIX_ms"]>0) else 0.20
        r   = row["RF_m"]   if not pd.isna(row["RF_m"]) else 0.03
        q   = 0.015 if pd.isna(row.get("SPY_div_yield", np.nan)) else row["SPY_div_yield"]
        prem = _bs(1.0, 1.0*(1+moneyness), r, q, sig) * overlay
        rows.append({"date": dt, "actual": row["JEPI_dist_yield"],
                     "premium": prem, "equity_div": q/12,
                     "reconstructed": prem + q/12})
    df = pd.DataFrame(rows).set_index("date")
    return df.dropna(subset=["actual", "reconstructed"])

@st.cache_data
def annual_returns(_m):
    ann = ((1+_m[["JEPI_ret","SPY_ret"]]).groupby(_m.index.year).prod()-1)*100
    ann.columns = ["JEPI","SPY"]; return ann

@st.cache_data
def drawdown_data(_d):
    dd = {}
    for name, col in [("JEPI","JEPI_tr"),("SPY","SPY_tr"),("AGG","AGG_tr")]:
        s = _d[col].dropna()
        dd[name] = (s / s.cummax() - 1) * 100
    return pd.DataFrame(dd)

@st.cache_data
def compute_10k(_d, initial=100.0):
    d = _d[["JEPI_px","JEPI_divs","SPY_tr"]].dropna().copy()
    shares_j = initial / d["JEPI_px"].iloc[0]
    j_price  = shares_j * d["JEPI_px"]
    j_divs   = shares_j * d["JEPI_divs"].cumsum()
    spy_tot  = d["SPY_tr"] / d["SPY_tr"].iloc[0] * initial
    return d.index, j_price, j_divs, spy_tot

@st.cache_data
def frontier(_m):
    cols = ["JEPI_ret","SPY_ret","AGG_ret"]; data = _m[cols].dropna()
    mu = data.mean().values*12; Sig = data.cov().values*12
    rf = float(_m["RF_m"].reindex(data.index).ffill().mean())
    n = len(mu)  # 3 risky assets only — Cash is CAL anchor, not a holdable asset
    def neg_sr(w):
        vol = max(float(w@Sig@w)**0.5, 1e-10)
        return -(float(w@mu)-rf)/vol
    best_sr, best_w = -np.inf, None
    rng = np.random.default_rng(42)
    for _ in range(30):
        w0  = rng.dirichlet(np.ones(n))
        res = minimize(neg_sr, w0, method="SLSQP", bounds=[(0,1)]*n,
                       constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                       options={"ftol":1e-12,"maxiter":2000})
        if res.success and -res.fun > best_sr:
            best_sr = -res.fun; best_w = res.x
    # Efficient frontier: start at min-variance portfolio (classic banana shape)
    res_mv = minimize(lambda w: float(w@Sig@w)**0.5, x0=np.ones(n)/n,
                      method="SLSQP", bounds=[(0,1)]*n,
                      constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                      options={"ftol":1e-12,"maxiter":2000})
    mu_start = float(res_mv.x@mu) if res_mv.success else float(min(mu))
    fv, fr = [], []
    for tgt in np.linspace(mu_start, float(max(mu))*0.999, 150):
        res = minimize(lambda w: float(w@Sig@w)**0.5,
                       x0=np.ones(n)/n, method="SLSQP", bounds=[(0,1)]*n,
                       constraints=[{"type":"eq",  "fun": lambda w: w.sum()-1},
                                    {"type":"ineq","fun": lambda w, t=tgt: float(w@mu)-t}],
                       options={"ftol":1e-12,"maxiter":2000})
        if res.success: fv.append(float(res.fun)); fr.append(float(res.x@mu))
    vt = float(np.sqrt(float(best_w@Sig@best_w))); rt = float(best_w@mu)
    return mu, np.sqrt(np.diag(Sig)), rf, best_w, best_sr, np.array(fv), np.array(fr), vt, rt

@st.cache_data
def regime_frontier(_m):
    cols = ["JEPI_ret","SPY_ret","AGG_ret"]; data = _m[cols].dropna()
    rf_s = _m["RF_m"].reindex(data.index).ffill()
    vrp  = _m["VRP_m"].reindex(data.index).ffill()
    med  = vrp.median(); t1, t2 = vrp.quantile(1/3), vrp.quantile(2/3)
    masks = [
        ("Gesamtstichprobe",      None),
        ("Niedr. VRP (≤ Median)", vrp <= med),
        ("Hohe VRP (> Median)",   vrp > med),
        ("Niedr. VRP (Terzil 1)", vrp <= t1),
        ("Mittl. VRP (Terzil 2)", (vrp > t1) & (vrp <= t2)),
        ("Hohe VRP (Terzil 3)",   vrp > t2),
    ]
    rng = np.random.default_rng(42); out = []
    for label, mask in masks:
        sub  = data if mask is None else data[mask]
        rf_r = float(rf_s.mean() if mask is None else rf_s[mask].mean())
        n_obs = len(sub)
        if n_obs < 5:
            out.append((label, n_obs, 0.0, 0.0, 0.0, 0.0)); continue
        mu_r = sub.mean().values*12; Sig_r = sub.cov().values*12; n = len(mu_r)
        def neg_sr(w, m=mu_r, S=Sig_r, r=rf_r):
            vol = max(float(w@S@w)**0.5, 1e-10); return -(float(w@m)-r)/vol
        bs, bw = -np.inf, np.ones(n)/n
        for _ in range(30):
            w0  = rng.dirichlet(np.ones(n))
            res = minimize(neg_sr, w0, method="SLSQP", bounds=[(0,1)]*n,
                          constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                          options={"ftol":1e-12,"maxiter":2000})
            if res.success and -res.fun > bs: bs = -res.fun; bw = res.x
        out.append((label, n_obs,
                    round(bw[1]*100,1),   # SPY  (index 1)
                    round(bw[0]*100,1),   # JEPI (index 0)
                    round(bw[2]*100,1),   # AGG  (index 2)
                    round(bs,2)))
    return out

def logo_html(path, width=190):
    if not os.path.exists(path): return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f'<div class="logo-box">'
            f'<img src="data:image/png;base64,{b64}" width="{width}"></div>')

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LAYOUT                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── HEADER ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.markdown(logo_html(os.path.join(_HERE, "fra_logo.png")), unsafe_allow_html=True)
with col_title:
    st.markdown(
        "<h1 style='margin-bottom:0.1rem;'>Die 45-Milliarden-Dollar-Frage</h1>"
        "<p style='font-size:1.2rem;color:#a5d6ff;margin:0 0 0.4rem 0;'>"
        "Generiert JEPI echtes Einkommen — oder verpackt er Aktienrendite neu?</p>"
        f"<p style='color:{MUTED};font-size:0.95rem;margin:0;'>"
        "Portfoliomanagement SS 2026 &nbsp;·&nbsp; Frankfurt UAS &nbsp;·&nbsp;"
        " Dozent: Benedikt Grimus<br>"
        + " &nbsp;·&nbsp; ".join(SPEAKERS) + "</p>",
        unsafe_allow_html=True,
    )

st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
for col, (val, lbl) in zip([k1,k2,k3,k4], [
    ("$45,6 Mrd.",  "JEPI Verwaltetes Vermögen"),
    ("~8,3 %",      "JEPI Ausschüttungsrendite p.a."),
    ("~1,3 %",      "SPY Ausschüttungsrendite p.a."),
    ("~6× höher",   "JEPI- vs. SPY-Rendite"),
]):
    col.markdown(f"<div class='kpi'><div class='kpi-val'>{val}</div>"
                 f"<div class='kpi-lbl'>{lbl}</div></div>",
                 unsafe_allow_html=True)

# ── FORSCHUNGSFRAGE ───────────────────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)

st.markdown("""
<div class="fq-box">
  <div class="fq-q">
    Generiert JEPI Einkommen —<br>oder verpackt er Aktienrendite neu?
  </div>
  <p style="font-size:1.1rem;color:#c9d1d9;margin:0;line-height:1.75;">
    Der JPMorgan Equity Premium Income ETF verwaltet $45,6 Mrd. und schüttet monatlich
    rund 8,3 % p.a. aus — rund sechsmal mehr als der S&amp;P 500.
    Diese Arbeit untersucht, ob diese Ausschüttung ökonomisch echtes Einkommen darstellt
    oder ob sie systematisch aus der Kursrendite der Anleger finanziert wird —
    als Nebenprodukt des monatlichen Covered-Call-Overlays.
  </p>
</div>
""", unsafe_allow_html=True)

# ── WAS IST JEPI? ─────────────────────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Was ist JEPI?")

col_txt, col_cc = st.columns([1, 1])
with col_txt:
    st.markdown("""
**JPMorgan Equity Premium Income ETF** — aktiv verwalteter ETF mit zwei Ertragsquellen:

1. **Defensives S&P-500-Aktienportfolio**
2. **Monatlicher Covered-Call-Overlay** — Verkauf von Kaufoptionen auf den S&P 500;
   die Prämien werden als monatliche Ausschüttung weitergeleitet

Mit **$45,6 Mrd. AUM** und **~8,3 % Ausschüttungsrendite** — der
schnellstwachsende aktive ETF aller Zeiten.
    """)
    st.markdown("""
<div class="info">
<strong>Kernmechanismus:</strong> Der Verkauf des Calls erzeugt sofortige Prämieneinnahmen —
begrenzt aber den Kursgewinn oberhalb des Strikes.
Die Prämie <em>stammt aus der eigenen Aufwärtsbeteiligung des Anlegers</em>.
</div>
    """, unsafe_allow_html=True)
    st.markdown("""
<div class="note">
<strong>Zentrale Spannung:</strong> Klingt die ~8,3 %-Rendite nach echtem Einkommen —
oder ist sie nur Kursrendite in einem anderen Gewand?
</div>
    """, unsafe_allow_html=True)

with col_cc:
    st.markdown("**Auszahlungsprofil eines Covered Call**")
    xs = np.linspace(80, 125, 300)
    S0, K, prem = 100.0, 102.0, 1.5
    pnl_stock = xs - S0
    pnl_call  = -np.maximum(xs-K, 0) + prem
    pnl_cc    = pnl_stock + pnl_call

    fig_cc = go.Figure()
    fig_cc.add_trace(go.Scatter(x=xs, y=pnl_stock, name="Long Aktie",
        line=dict(color=CS, width=1.8, dash="dash")))
    fig_cc.add_trace(go.Scatter(x=xs, y=pnl_call, name="Short Call + Prämie",
        line=dict(color="#888888", width=1.8, dash="dot")))
    fig_cc.add_trace(go.Scatter(x=xs, y=pnl_cc, name="Covered Call (gesamt)",
        line=dict(color=CJ, width=3.5),
        fill="tozeroy", fillcolor="rgba(0,80,200,0.12)"))
    fig_cc.add_vline(x=K, line_color=TXT, line_width=1.2, line_dash="dot",
                     annotation_text=f"Strike K={int(K)}", annotation_font_color=TXT,
                     annotation_font_size=12, annotation_position="top right")
    fig_cc.update_layout(**lo(h=420))
    fig_cc.update_layout(xaxis_title="S&P-500-Kurs bei Verfall",
                         yaxis_title="Gewinn / Verlust", hovermode="x")
    st.plotly_chart(fig_cc, use_container_width=True)

# ── LIVE KURSVERLAUF ──────────────────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Live Kursverlauf — JEPI & SPY (seit Auflage Mai 2020)")

live_data, live_err = load_live()

if live_data is not None and not live_data.empty:
    jepi_now = float(live_data["JEPI"].iloc[-1])
    spy_now  = float(live_data["SPY"].iloc[-1])
    jepi_chg = float((live_data["JEPI"].iloc[-1]/live_data["JEPI"].iloc[-2]-1)*100)
    spy_chg  = float((live_data["SPY"].iloc[-1]/live_data["SPY"].iloc[-2]-1)*100)

    mc1, mc2, mc3 = st.columns([1,1,3])
    mc1.metric("JEPI", f"${jepi_now:.2f}", f"{jepi_chg:+.2f} % (1T)")
    mc2.metric("SPY",  f"${spy_now:.2f}",  f"{spy_chg:+.2f} % (1T)")
    mc3.markdown(
        f"<p style='color:{MUTED};font-size:0.9rem;padding-top:20px;'>"
        f"Letzter Datenpunkt: {live_data.index[-1].strftime('%d.%m.%Y')} · "
        "Live-Daten via Yahoo Finance · Aktualisierung alle 5 Minuten</p>",
        unsafe_allow_html=True,
    )
    bj = live_data["JEPI"].iloc[0]; bs_ = live_data["SPY"].iloc[0]
    fig_live = go.Figure()
    fig_live.add_trace(go.Scatter(
        x=live_data.index, y=live_data["JEPI"]/bj*100, name="JEPI",
        line=dict(color="#58a6ff", width=2.8),
        fill="tozeroy", fillcolor="rgba(31,111,235,0.10)"))
    fig_live.add_trace(go.Scatter(
        x=live_data.index, y=live_data["SPY"]/bs_*100, name="SPY",
        line=dict(color=CS, width=2.2, dash="dot")))
    fig_live.update_layout(**lo(h=440, title="Gesamtrendite indexiert (Basis = 100, seit Mai 2020)"))
    fig_live.update_layout(xaxis_title="Datum", yaxis_title="Indexiert (Basis = 100)")
    st.plotly_chart(fig_live, use_container_width=True)
else:
    st.warning(f"Live-Daten nicht verfügbar ({live_err}). Offline-Daten werden angezeigt.")
    tr = daily[["JEPI_tr","SPY_tr"]].dropna()
    bj = tr["JEPI_tr"].iloc[0]; bs_ = tr["SPY_tr"].iloc[0]
    fig_off = go.Figure()
    fig_off.add_trace(go.Scatter(x=tr.index, y=tr["JEPI_tr"]/bj*100, name="JEPI",
        line=dict(color="#58a6ff", width=2.8)))
    fig_off.add_trace(go.Scatter(x=tr.index, y=tr["SPY_tr"]/bs_*100, name="SPY",
        line=dict(color=CS, width=2, dash="dot")))
    fig_off.update_layout(**lo(h=440, title="Gesamtrendite indexiert (Basis = 100, CSV-Daten)"))
    fig_off.update_layout(xaxis_title="Datum", yaxis_title="Indexiert (Basis = 100)")
    st.plotly_chart(fig_off, use_container_width=True)

# ── DATENBASIS ────────────────────────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Datenbasis & Deskriptive Statistik")

t1_df = pd.DataFrame({
    "Kennzahl": ["Ann. Rendite (%)", "Ann. Volatilität (%)", "Schiefe (monatl.)",
                 "Überschuss-Kurtosis", "Sharpe Ratio", "Max. Drawdown (%)"],
    "JEPI": [T1["JEPI"]["ret"],  T1["JEPI"]["vol"],  T1["JEPI"]["skew"],
             T1["JEPI"]["kurt"], T1["JEPI"]["sharpe"],T1["JEPI"]["mdd"]],
    "SPY":  [T1["SPY"]["ret"],   T1["SPY"]["vol"],   T1["SPY"]["skew"],
             T1["SPY"]["kurt"],  T1["SPY"]["sharpe"], T1["SPY"]["mdd"]],
    "AGG":  [T1["AGG"]["ret"],   T1["AGG"]["vol"],   T1["AGG"]["skew"],
             T1["AGG"]["kurt"],  T1["AGG"]["sharpe"], T1["AGG"]["mdd"]],
})

def _hl_t1(df):
    s = pd.DataFrame("", index=df.index, columns=df.columns)
    s.iloc[4, 1:] = "background-color:#1a3a6e;color:#93c5fd;font-weight:700"
    return s

st.dataframe(
    t1_df.style.apply(_hl_t1, axis=None)
         .format({"JEPI":"{:.2f}","SPY":"{:.2f}","AGG":"{:.2f}"}),
    hide_index=True, use_container_width=True,
)
st.markdown("""
<div class="find">
JEPI bietet ca. <strong>65 % der SPY-Rendite bei nur 66 % der Volatilität</strong>
— aber SPY bleibt nach Sharpe Ratio der stärkere Baustein (0,95 vs. 0,74).
</div>
""", unsafe_allow_html=True)

# ── ABBILDUNG 1 — AUSSCHÜTTUNGSREKONSTRUKTION ─────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Abbildung 1 — Ausschüttungsrekonstruktion")

m1a, m1b, m1c = st.columns(3)
m1a.metric("Zeitliche Korrelation",  f"{T2['corr']:.3f}")
m1b.metric("Mittleres Verhältnis",   f"{T2['rom']:.1%}")
m1c.metric("Erklärte Ausschüttung", "~57 %")

rc = recon_df(monthly)
fig1 = go.Figure()
fig1.add_trace(go.Bar(x=rc.index, y=rc["actual"]*100, name="Tatsächlich (JEPI)",
    marker_color=CJ, opacity=0.80,
    hovertemplate="%{x|%b %Y}: %{y:.3f} %<extra>Tatsächlich</extra>"))
fig1.add_trace(go.Scatter(x=rc.index, y=rc["reconstructed"]*100,
    name="Rekonstruiert (BS + Div.)", line=dict(color=CR, width=2.8),
    hovertemplate="%{x|%b %Y}: %{y:.3f} %<extra>Rekonstruiert</extra>"))
fig1.add_trace(go.Scatter(x=rc.index, y=rc["premium"]*100,
    name="Optionsprämie", line=dict(color="#f59e0b", width=1.6, dash="dash"), opacity=0.9))
fig1.add_trace(go.Scatter(x=rc.index, y=rc["equity_div"]*100,
    name="Aktiendividende", line=dict(color=CA, width=1.6, dash="dot"), opacity=0.9))
fig1.add_annotation(x=0.01, y=0.97, xref="paper", yref="paper", showarrow=False,
    text=f"Korrelation: {T2['corr']:.3f}  |  Verhältnis: {T2['rom']:.3f}",
    font=dict(size=13, color=TXT), bgcolor=BG2, bordercolor=BD, borderwidth=1)
fig1.update_layout(**lo(h=460, title="JEPI Tatsächliche vs. Rekonstruierte Ausschüttungsrendite"))
fig1.update_layout(xaxis_title="Datum", yaxis_title="Monatliche Rendite (% des NAV)")
fig1.update_yaxes(ticksuffix=" %")
st.plotly_chart(fig1, use_container_width=True)

col_f1a, col_f1b = st.columns(2)
with col_f1a:
    st.markdown("""
<div class="find">
<strong>Befund 1:</strong> Black-Scholes erklärt ~57 % der durchschnittlichen Ausschüttung
(Korrelation 0,51). Hohe Volatilität → höhere Prämien → höhere Ausschüttung.
JEPI <em>glättet</em> aktiv.
</div>
    """, unsafe_allow_html=True)
with col_f1b:
    cix = next(i for i,r in enumerate(T3) if r[4])
    t3_df = pd.DataFrame([{"Moneyness": m+(" ★" if c else ""), "Overlay": ov,
                            "Korrelation": f"{co:.3f}", "Verhältnis": f"{rm:.3f}"}
                           for m,ov,co,rm,c in T3])
    def _hl3(df):
        s = pd.DataFrame("", index=df.index, columns=df.columns)
        s.iloc[cix] = "background-color:#1a3a6e;font-weight:700"; return s
    st.caption("Tabelle 3 — Sensitivitätsgitter (★ = Zentralfall)")
    st.dataframe(t3_df.style.apply(_hl3, axis=None), hide_index=True, use_container_width=True)

# ── VRP VISUALISIERUNG ────────────────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Volatilitätsrisikoprämie (VRP) — Das Herzstück von JEPIs Strategie")

vrp_m = monthly[["VIX_ms","VRP_m"]].dropna().copy()
vrp_m["VIX_pct"] = vrp_m["VIX_ms"] * 100
vrp_m["RV_pct"]  = (vrp_m["VIX_ms"] - vrp_m["VRP_m"]) * 100

v_col1, v_col2 = st.columns([2.2, 1])
with v_col1:
    fig_vrp = go.Figure()
    fig_vrp.add_trace(go.Scatter(
        x=vrp_m.index, y=vrp_m["RV_pct"],
        name="Realisierte Volatilität", line=dict(color="#58a6ff", width=2, dash="dash")))
    fig_vrp.add_trace(go.Scatter(
        x=vrp_m.index, y=vrp_m["VIX_pct"],
        name="VIX (implizite Volatilität)",
        fill="tonexty", fillcolor="rgba(200,16,46,0.20)",
        line=dict(color=CS, width=2.5)))
    avg_vrp = float(vrp_m["VRP_m"].mean() * 100)
    fig_vrp.add_annotation(
        x=0.02, y=0.95, xref="paper", yref="paper",
        text=f"Ø VRP: {avg_vrp:.1f} Pp. — schraffierte Fläche = vereinnahmte Prämie",
        font=dict(size=12, color=CS), bgcolor=BG2, bordercolor=BD, borderwidth=1,
        showarrow=False, xanchor="left")
    fig_vrp.update_layout(**lo(h=380, title="VIX vs. Realisierte Volatilität (monatlich, ann.)"))
    fig_vrp.update_layout(xaxis_title="Datum", yaxis_title="Volatilität (ann., %)")
    fig_vrp.update_yaxes(ticksuffix=" %")
    st.plotly_chart(fig_vrp, use_container_width=True)

with v_col2:
    st.markdown("""
<div class="info">
<strong>VRP = VIX − Realisierte Vola</strong><br><br>
Die schraffierte Fläche zwischen VIX und realisierter Vola ist die
<em>Prämie</em>, die JEPI durch Call-Verkäufe vereinnahmt.<br><br>
&bull;&nbsp; Ø VIX&nbsp;  ≈ 20,2 %<br>
&bull;&nbsp; Ø RV &nbsp;&nbsp; ≈ 15,8 %<br>
&bull;&nbsp; Ø VRP ≈ &nbsp;4,4 Pp.<br><br>
<strong>Verbindung zur Optionsprämie:</strong><br>
Black-Scholes bewertet JEPIs Calls mit σ&nbsp;=&nbsp;VIX.
Weil VIX&nbsp;&gt;&nbsp;σ<sub>real</sub>, sind die Calls
<em>systematisch überbewertet</em> — JEPI kassiert diese
Überprämie (VRP&nbsp;&gt;&nbsp;0).<br><br>
<span style="font-family:monospace;font-size:0.95em;">
C(σ=VIX) &gt; C(σ=σ<sub>real</sub>)<br>
⟹ Prämie ∝ VRP
</span>
</div>
    """, unsafe_allow_html=True)

# ── ABBILDUNG 2 — JAHRESRENDITEN ──────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Abbildung 2 — Jahresrenditen: JEPI vs. SPY")

ann = annual_returns(monthly)
years = [str(y) for y in ann.index]
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=years, y=ann["JEPI"], name="JEPI", marker_color=CJ, offsetgroup=0,
    text=[f"{v:.1f}%" for v in ann["JEPI"]], textposition="outside",
    textfont=dict(size=12, color="#93c5fd")))
fig2.add_trace(go.Bar(x=years, y=ann["SPY"],  name="SPY",  marker_color=CS, offsetgroup=1,
    text=[f"{v:.1f}%" for v in ann["SPY"]],  textposition="outside",
    textfont=dict(size=12, color="#fca5a5")))
fig2.add_hline(y=0, line_color=TXT, line_width=0.8)
for yr, note in [("2020","* Mai–Dez."),("2026","* Jan.–Mai")]:
    if yr in years:
        fig2.add_annotation(x=yr, y=float(min(ann.loc[int(yr)].min(),0))-9,
            text=note, showarrow=False, font=dict(size=10, color=MUTED))
fig2.update_layout(**lo(h=460, title="Kalenderjahrrenditen (2020* und 2026* sind Teilperioden)"))
fig2.update_layout(xaxis_title="Kalenderjahr", yaxis_title="Jahresrendite (%)", barmode="group")
fig2.update_yaxes(ticksuffix=" %")
st.plotly_chart(fig2, use_container_width=True)

cc1, cc2, cc3, cc4 = st.columns(4)
cc1.metric("Aufwärts Capture",  f"{T4['up_cap']:.1%}", help=f"N={T4['up_n']} Monate")
cc2.metric("Abwärts Capture",   f"{T4['dn_cap']:.1%}", help=f"N={T4['dn_n']} Monate")
cc3.metric("Beta Aufwärts",     f"{T4['up_beta']:.3f}")
cc4.metric("Beta Abwärts",      f"{T4['dn_beta']:.3f}", delta=f"Asymmetrie {T4['asymmetry']:.2f}×")
st.markdown("""
<div class="note">
<strong>Befund 2:</strong> Capture Ratios ~59 % — nahezu symmetrisch auf- und abwärts.
Abwärts-Beta (0,556) > Aufwärts-Beta (0,474), Asymmetrie 1,17×.
Kein statistisch bedeutsamer Kapitalschutz in Abwärtsmärkten.
</div>
""", unsafe_allow_html=True)

# ── DRAWDOWN ─────────────────────────────────────────────────────────────────
st.markdown("**Kumulativer Drawdown — JEPI, SPY, AGG**")
dd = drawdown_data(daily)
fig_dd = go.Figure()
for name, color, fill in [
    ("JEPI", "#58a6ff", "rgba(88,166,255,0.12)"),
    ("SPY",  CS,        "rgba(200,16,46,0.10)"),
    ("AGG",  CA,        "rgba(34,139,34,0.10)"),
]:
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd[name], name=name,
        fill="tozeroy", fillcolor=fill,
        line=dict(color=color, width=2.2)))
for name, val, color in [("JEPI", T1["JEPI"]["mdd"],"#58a6ff"),
                          ("SPY",  T1["SPY"]["mdd"],  CS),
                          ("AGG",  T1["AGG"]["mdd"],  CA)]:
    fig_dd.add_annotation(x=0.99, y=val, xref="paper", yref="y",
        text=f"{val:.1f} %", font=dict(size=11, color=color),
        showarrow=False, xanchor="right", bgcolor=BG2, bordercolor=BD)
fig_dd.add_hline(y=0, line_color=MUTED, line_width=0.8)
fig_dd.update_layout(**lo(h=420))
fig_dd.update_layout(xaxis_title="Datum", yaxis_title="Drawdown (%)")
fig_dd.update_yaxes(ticksuffix=" %")
st.plotly_chart(fig_dd, use_container_width=True)

# ── ABBILDUNG 3 — EFFIZIENZLINIE & VRP-REGIMES ───────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Abbildung 3 — Markowitz-Effizienzlinie & VRP-Regimes")

with st.spinner("Berechne Effizienzlinie …"):
    mu, vols_i, rf, bw, bsr, fv, fr, vt, rt = frontier(monthly)

# Min-Var-Portfolio bei JEPI's Rendite-Niveau
_jepi_ret = mu[0]
_res_mv = minimize(
    lambda w: float(w @ (monthly[["JEPI_ret","SPY_ret","AGG_ret"]].dropna().cov().values*12) @ w)**0.5,
    x0=np.array([0.33,0.33,0.34]), method="SLSQP", bounds=[(0,1)]*3,
    constraints=[{"type":"eq","fun":lambda w:w.sum()-1},
                 {"type":"ineq","fun":lambda w:float(w@mu)-_jepi_ret}],
    options={"ftol":1e-14,"maxiter":5000})
_mv_vol = _res_mv.fun*100 if _res_mv.success else vols_i[0]*100
_mv_w   = _res_mv.x if _res_mv.success else np.array([1,0,0])
_jepi_sharpe = (mu[0]-rf)/vols_i[0]
_spy_sharpe  = (mu[1]-rf)/vols_i[1]

fig3 = go.Figure()
if len(fv) > 1:
    fig3.add_trace(go.Scatter(x=np.array(fv)*100, y=np.array(fr)*100,
        mode="lines", name="Effizienzlinie (Long-only)",
        line=dict(color="#58a6ff", width=2.8)))
if vt > 0:
    cal_v = np.linspace(0, vt*1.8, 60)
    fig3.add_trace(go.Scatter(x=cal_v*100, y=(rf+(rt-rf)/vt*cal_v)*100,
        mode="lines", name="Capital Allocation Line",
        line=dict(color="#9b59b6", width=1.8, dash="dash"), opacity=0.85))
for nm, av, ar, ac in zip(["JEPI","SPY","AGG"], vols_i*100, mu*100, [CJ,CS,CA]):
    fig3.add_trace(go.Scatter(x=[av], y=[ar], mode="markers+text", name=nm,
        text=[f"  {nm}"], textposition="middle right",
        marker=dict(color=ac, size=14, line=dict(color="white",width=1.5)),
        textfont=dict(size=13)))
fig3.add_trace(go.Scatter(x=[vt*100], y=[rt*100], mode="markers",
    name=f"Tangenzportfolio (Sharpe ≈ {bsr:.2f})",
    marker=dict(color="gold", size=20, symbol="star",
                line=dict(color="#374151",width=1.5))))
fig3.add_trace(go.Scatter(x=[0], y=[rf*100], mode="markers+text",
    text=[f"  RF ({rf*100:.1f}%)"], textposition="middle right",
    marker=dict(color=CC, size=10, symbol="diamond"),
    textfont=dict(size=11, color=MUTED), showlegend=False))
fig3.update_layout(**lo(h=500, title="Effizienzlinie: SPY, JEPI, AGG (Long-only)"))
fig3.update_layout(xaxis_title="Ann. Volatilität (%)", yaxis_title="Ann. Erwartungsrendite (%)",
                   hovermode="closest", legend=dict(font=dict(size=11)))
fig3.update_xaxes(ticksuffix=" %"); fig3.update_yaxes(ticksuffix=" %")
st.plotly_chart(fig3, use_container_width=True)

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(f"""<div class="kpi-box">
        <div class="kpi-label">Tangenzportfolio</div>
        <div class="kpi-value">100 % SPY</div>
        <div class="kpi-sub">Sharpe {_spy_sharpe:.2f} — höchste risikoadjustierte Rendite</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-box">
        <div class="kpi-label">JEPI — Rendite &amp; Risiko</div>
        <div class="kpi-value">{mu[0]*100:.1f} % | {vols_i[0]*100:.1f} %</div>
        <div class="kpi-sub">Rendite | Vola &nbsp;·&nbsp; Frontier bei gleicher Rendite: σ = {_mv_vol:.1f} %</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-box">
        <div class="kpi-label">Min-Var bei JEPI-Rendite</div>
        <div class="kpi-value">σ = {_mv_vol:.1f} %</div>
        <div class="kpi-sub">{_mv_w[0]*100:.0f} % JEPI · {_mv_w[1]*100:.0f} % SPY · {_mv_w[2]*100:.0f} % AGG</div>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="find">
<strong>Befund 3:</strong> Tangenzportfolio = <strong>100 % SPY</strong> (Sharpe {_spy_sharpe:.2f}).
JEPI (Sharpe {_jepi_sharpe:.2f}) liegt knapp unterhalb der Effizienzlinie —
bei gleicher Rendite ({mu[0]*100:.1f} %) wäre eine Volatilität von {_mv_vol:.1f} % erreichbar (JEPI: {vols_i[0]*100:.1f} %).
Im Gesamtzeitraum bietet JEPI <strong>keinen mean-variance Mehrwert</strong> gegenüber SPY.
</div>
""", unsafe_allow_html=True)

# ── ANLEGER-PERSPEKTIVE ───────────────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Anleger-Perspektive: Was wäre aus Ihrer Investition geworden?")

amt_col, _ = st.columns([1, 2])
with amt_col:
    invest_amt = st.number_input(
        "Anfangsinvestition in USD (Mai 2020) — Betrag frei wählbar",
        min_value=1,
        max_value=10_000_000,
        value=100,
        step=100,
        format="%d",
        help="Alle Zahlen — Kapitalwert, Ausschüttungen und SPY-Vergleich — aktualisieren sich automatisch.",
    )

idx_10k, j_px_val, j_div_val, spy_val = compute_10k(daily, initial=float(invest_amt))
final_j_price = float(j_px_val.iloc[-1])
final_j_divs  = float(j_div_val.iloc[-1])
final_j_total = final_j_price + final_j_divs
final_spy     = float(spy_val.iloc[-1])
gap           = final_spy - final_j_total

fig_10k = go.Figure()
fig_10k.add_trace(go.Scatter(
    x=idx_10k, y=j_px_val,
    name="JEPI — Kapitalwert (Kursanteil)",
    fill="tozeroy", fillcolor="rgba(0,48,135,0.45)",
    line=dict(color=CJ, width=2.0)))
fig_10k.add_trace(go.Scatter(
    x=idx_10k, y=j_px_val + j_div_val,
    name="JEPI — Gesamt inkl. kumulierter Ausschüttungen",
    fill="tonexty", fillcolor="rgba(88,166,255,0.28)",
    line=dict(color="#58a6ff", width=3.0)))
fig_10k.add_trace(go.Scatter(
    x=idx_10k, y=spy_val,
    name="SPY — Total Return (Dividenden reinvestiert)",
    line=dict(color=CS, width=3.2, dash="dot")))
fig_10k.add_hline(y=float(invest_amt), line_color=MUTED, line_width=1, line_dash="dash")
fig_10k.add_annotation(x=0.01, y=float(invest_amt), xref="paper", yref="y",
    text=f"Anfangsinvestition ${invest_amt:,}", font=dict(size=10, color=MUTED),
    showarrow=False, yanchor="bottom", xanchor="left")
fig_10k.add_annotation(
    x=idx_10k[-1], y=final_j_total + max(final_j_total*0.03, 1),
    text=f"JEPI gesamt: ${final_j_total:,.0f}",
    showarrow=True, arrowhead=2, ax=-150, ay=-40,
    font=dict(size=12, color="#58a6ff"), bgcolor=BG2, bordercolor="#58a6ff", borderwidth=1)
fig_10k.add_annotation(
    x=idx_10k[-1], y=final_spy,
    text=f"SPY: ${final_spy:,.0f}",
    showarrow=True, arrowhead=2, ax=-110, ay=50,
    font=dict(size=12, color=CS), bgcolor=BG2, bordercolor=CS, borderwidth=1)
fig_10k.update_layout(**lo(h=510,
    title=f"${invest_amt:,} Anfangsinvestition Mai 2020 — Kapitalwert + Ausschüttungen vs. SPY"))
fig_10k.update_layout(xaxis_title="Datum", yaxis_title="Wert (USD)")
fig_10k.update_yaxes(tickprefix="$", tickformat=",.0f")
st.plotly_chart(fig_10k, use_container_width=True)

inv1, inv2, inv3 = st.columns(3)
inv1.metric("JEPI Kapitalwert (Kurs)", f"${final_j_price:,.0f}",
            delta=f"{final_j_price/invest_amt-1:+.1%}")
inv2.metric("JEPI kumulierte Ausschüttungen", f"${final_j_divs:,.0f}",
            help="Bare Auszahlungen, nicht reinvestiert")
inv3.metric("SPY Total Return", f"${final_spy:,.0f}",
            delta=f"{final_spy/invest_amt-1:+.1%}")


# ── GESAMTFAZIT ───────────────────────────────────────────────────────────────
st.markdown('<hr class="sec-rule">', unsafe_allow_html=True)
st.markdown("## Gesamtfazit")

c1, c2, c3 = st.columns(3)
for col, (num, head, body) in zip([c1,c2,c3], [
    ("Befund 1", "Ausschüttungsrekonstruktion",
     "Black-Scholes erklärt ~57 % der durchschnittlichen Ausschüttung (Korrelation 0,51). "
     "Die Prämie ist ökonomisch rekonstruierbar — JEPI glättet sie jedoch aktiv."),
    ("Befund 2", "Renditeprofil & Asymmetrie",
     "Capture Ratios ~59 % — symmetrisch. Abwärts-Beta (0,556) > Aufwärts-Beta (0,474), "
     "Asymmetrie 1,17×. Kein bedeutsamer Kapitalschutz beobachtbar."),
    ("Befund 3", "Markowitz-Effizienzlinie",
     "Tangenzportfolio = 100 % SPY (Sharpe 0,95). JEPI (Sharpe 0,74) liegt knapp "
     "unterhalb der Effizienzlinie. Im Gesamtzeitraum kein mean-variance Mehrwert gegenüber SPY."),
]):
    col.markdown(
        f"<div class='conc'><div class='conc-num'>{num}</div>"
        f"<div class='conc-head'>{head}</div>"
        f"<div class='conc-body'>{body}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("""
<div class="answer" style="margin-top:24px;">
  JEPI generiert kein ökonomisch <em>neues</em> Einkommen — er verpackt Aktienrendite um.
  Die monatlichen Ausschüttungen werden strukturell durch den Verzicht auf Kursgewinne
  oberhalb des Optionsstrikes finanziert. Wer alle Erträge summiert, liegt mit JEPI
  systematisch hinter dem SPY — das Tangenzportfolio allokiert 0 % in JEPI.
  Als <em>konditionaler</em> Baustein für Anleger mit explizitem Ausschüttungsbedarf
  (Stiftungen, Entnahmephasen) kann JEPI jedoch gezielt eingesetzt werden —
  sofern die Volatilitätsrisikoprämie (VRP) dies rechtfertigt.
</div>
""", unsafe_allow_html=True)

st.markdown(
    f"<p style='text-align:center;color:{MUTED};font-size:0.85rem;margin-top:3rem;'>"
    "Frankfurt University of Applied Sciences · Portfoliomanagement SS 2026 · "
    "Dozent: Benedikt Grimus</p>",
    unsafe_allow_html=True,
)
