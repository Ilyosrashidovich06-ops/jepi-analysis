# JEPI Analyse — FRA-UAS Portfoliomanagement SoSe 2026

**Deutsch:** Streamlit-App zur Analyse des JPMorgan Equity Premium Income ETF (JEPI)
als Grundlage einer 15-minütigen Bachelorpräsentation an der Frankfurt University of
Applied Sciences (FRA-UAS) im Sommersemester 2026.

**English:** A full Streamlit web app analyzing JEPI's covered-call yield strategy —
built for a Bachelor's portfolio management presentation at FRA-UAS, SoSe 2026.

---

## Forschungsfrage

> Stellt die Ausschüttungsrendite eines Covered-Call-ETF ökonomisch tatsächlich
> generiertes Einkommen dar — oder lässt sie sich vollständig als systematische
> Umwandlung der Aktienrendite des Anlegers in laufende Auszahlungen rekonstruieren,
> finanziert durch den Verkauf der Aufwärtsbeteiligung?

---

## Seitenstruktur

| # | Seite | Inhalt |
|---|-------|--------|
| 1 | Die 45-Milliarden-Dollar-Frage | Eröffnung, KPIs, Hero-Chart, Forschungsfrage |
| 2 | Funktionsweise | Sankey-Diagramm, Vergleichstabelle, interaktiver Payoff |
| 3 | Die Mathematik | Black-Scholes, VRP, BS-Prämie vs. JEPI-Ausschüttung |
| 4 | Empirische Realität | 7 Charts: Gesamtrendite, Vol, Capture Ratio, Drawdown, Dekomposition |
| 5 | Portfolio-Optimierung | Effiziente Grenze, Max-Sharpe, Risikoaversions-Slider, Backtest |
| 6 | Das Verdikt | Claim vs. Reality, Allokationsmatrix, Schlussfolgerung |

---

## Installation & Start

```bash
# 1. Repository klonen
git clone https://github.com/DEIN-USERNAME/jepi-analysis.git
cd jepi-analysis

# 2. Abhängigkeiten installieren
pip install -r requirements.txt

# 3. App starten
streamlit run app.py
```

Die App läuft dann unter `http://localhost:8501`.

---

## Deployment auf Streamlit Community Cloud

1. Repository auf GitHub pushen (Public oder Private mit Streamlit-Zugriff)
2. https://share.streamlit.io → "New app"
3. Repository auswählen, Branch `main`, Datei `app.py`
4. "Deploy!" klicken

**Live-App:** `https://share.streamlit.io/DEIN-USERNAME/jepi-analysis/main/app.py`
*(Platzhalter — nach Deployment aktualisieren)*

---

## Screenshots

| Seite | Screenshot |
|-------|------------|
| Page 1 — Die Frage | *[Screenshot einfügen]* |
| Page 3 — Mathematik | *[Screenshot einfügen]* |
| Page 4 — Daten | *[Screenshot einfügen]* |
| Page 5 — Optimierung | *[Screenshot einfügen]* |

---

## Team

- [Name 1] — [Beitrag]
- [Name 2] — [Beitrag]
- Ilyos Rashidovich — App-Entwicklung, Datenanalyse

---

## Referenzen

- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Hull, J. C. (2021). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.
- JPMorgan Asset Management. JEPI Factsheet (2024). https://am.jpmorgan.com/us/en/asset-management/adv/products/jpmorgan-equity-premium-income-etf-46641q332
