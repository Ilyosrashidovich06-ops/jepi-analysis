import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "snapshots")
START_DATE = "2020-05-20"


def _snapshot_path(ticker: str) -> str:
    return os.path.join(SNAPSHOT_DIR, f"{ticker.replace('^', '')}.csv")


def _save_snapshot(df: pd.DataFrame, ticker: str) -> None:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    df.to_csv(_snapshot_path(ticker))


def _load_snapshot(ticker: str) -> pd.DataFrame | None:
    path = _snapshot_path(ticker)
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    return None


@st.cache_data(ttl=3600)
def load_prices(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    try:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("Empty response")
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        _save_snapshot(df, f"{ticker}_prices")
        return df
    except Exception as e:
        snap = _load_snapshot(f"{ticker}_prices")
        if snap is not None:
            st.warning(f"Offline-Modus: Nutze gespeicherte Daten für {ticker} ({e})")
            return snap
        st.error(f"Konnte {ticker} nicht laden und kein Snapshot vorhanden.")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_dividends(ticker: str) -> pd.Series:
    try:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = yf.Ticker(ticker)
            divs = t.dividends
        if divs.empty:
            raise ValueError("No dividends")
        divs.index = divs.index.tz_localize(None)
        _save_snapshot(divs.to_frame("Dividends"), f"{ticker}_divs")
        return divs
    except Exception as e:
        snap = _load_snapshot(f"{ticker}_divs")
        if snap is not None:
            st.warning(f"Offline-Modus: Dividenden-Snapshot für {ticker}")
            return snap["Dividends"]
        return pd.Series(dtype=float, name="Dividends")


@st.cache_data(ttl=3600)
def load_total_returns(ticker: str, start: str = START_DATE) -> pd.Series:
    """Cumulative total return series (price + reinvested dividends), base=1."""
    prices = load_prices(ticker, start)
    if prices.empty:
        return pd.Series(dtype=float)

    close = prices["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)

    divs = load_dividends(ticker)
    divs = divs[divs.index >= close.index[0]]

    # Daily price returns
    price_ret = close.pct_change().fillna(0)

    # Add dividend yield on ex-div days
    div_yield = pd.Series(0.0, index=close.index)
    for date, div in divs.items():
        if date in close.index:
            div_yield[date] = div / close.shift(1).get(date, close[date])

    total_ret = price_ret + div_yield
    cumulative = (1 + total_ret).cumprod()
    cumulative = cumulative / cumulative.iloc[0]
    cumulative.name = ticker
    return cumulative


@st.cache_data(ttl=3600)
def load_vix(start: str = START_DATE) -> pd.Series:
    try:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download("^VIX", start=start, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("Empty VIX")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        _save_snapshot(s.to_frame("VIX"), "VIX")
        return s
    except Exception as e:
        snap = _load_snapshot("VIX")
        if snap is not None:
            st.warning(f"Offline-Modus: VIX-Snapshot ({e})")
            return snap["VIX"]
        return pd.Series(dtype=float, name="VIX")


@st.cache_data(ttl=3600)
def load_risk_free(start: str = START_DATE) -> pd.Series:
    """1-month Treasury yield via ^IRX (annualized %)."""
    try:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download("^IRX", start=start, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("Empty IRX")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"] / 100
        s.index = pd.to_datetime(s.index).tz_localize(None)
        _save_snapshot(s.to_frame("IRX"), "IRX")
        return s
    except Exception as e:
        snap = _load_snapshot("IRX")
        if snap is not None:
            st.warning(f"Offline-Modus: IRX-Snapshot ({e})")
            return snap["IRX"]
        return pd.Series(dtype=float, name="IRX")


@st.cache_data(ttl=3600)
def load_monthly_returns(ticker: str, start: str = START_DATE) -> pd.Series:
    """Monthly total returns for a ticker."""
    tr = load_total_returns(ticker, start)
    if tr.empty:
        return pd.Series(dtype=float)
    monthly = tr.resample("ME").last()
    returns = monthly.pct_change().dropna()
    returns.name = ticker
    return returns


def get_latest_price(ticker: str) -> float:
    prices = load_prices(ticker)
    if prices.empty:
        return float("nan")
    return float(prices["Close"].iloc[-1])


def get_latest_vix() -> float:
    vix = load_vix()
    if vix.empty:
        return 20.0
    return float(vix.iloc[-1])


def get_latest_rf() -> float:
    rf = load_risk_free()
    if rf.empty:
        return 0.05
    return float(rf.iloc[-1])
