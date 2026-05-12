import numpy as np
import pandas as pd
from scipy.optimize import minimize


def realized_vol(returns: pd.Series, window: int = 30) -> pd.Series:
    """Annualized rolling realized volatility (std of daily returns)."""
    return returns.rolling(window).std() * np.sqrt(252)


def sharpe_ratio(returns: pd.Series, rf: float) -> float:
    """Annualized Sharpe ratio. rf is annualized."""
    excess = returns - rf / 252
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def sortino_ratio(returns: pd.Series, rf: float) -> float:
    """Annualized Sortino ratio. rf is annualized."""
    excess = returns - rf / 252
    downside = excess[excess < 0].std()
    if downside == 0:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(252))


def max_drawdown(prices: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return float(drawdown.min())


def drawdown_series(prices: pd.Series) -> pd.Series:
    rolling_max = prices.cummax()
    return (prices - rolling_max) / rolling_max


def calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
    ann_return = float(returns.mean() * 252)
    mdd = abs(max_drawdown(prices))
    if mdd == 0:
        return 0.0
    return ann_return / mdd


def capture_ratios(asset_returns: pd.Series, benchmark_returns: pd.Series):
    """Up/down capture ratios."""
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    a, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    up_mask = b > 0
    down_mask = b < 0
    up_cap = (a[up_mask].mean() / b[up_mask].mean()) if up_mask.sum() > 0 else float("nan")
    down_cap = (a[down_mask].mean() / b[down_mask].mean()) if down_mask.sum() > 0 else float("nan")
    return float(up_cap), float(down_cap)


def summary_stats(prices: pd.Series, rf_annual: float = 0.05) -> dict:
    returns = prices.pct_change().dropna()
    return {
        "Ann. Rendite": returns.mean() * 252,
        "Ann. Volatilität": returns.std() * np.sqrt(252),
        "Sharpe Ratio": sharpe_ratio(returns, rf_annual),
        "Sortino Ratio": sortino_ratio(returns, rf_annual),
        "Max Drawdown": max_drawdown(prices),
        "Calmar Ratio": calmar_ratio(returns, prices),
    }


# ── Mean-Variance Optimization ──────────────────────────────────────────────

def _portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, cov: np.ndarray):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(weights @ cov @ weights)
    return port_return, port_vol


def efficient_frontier(returns: pd.DataFrame, n_points: int = 80) -> pd.DataFrame:
    """Compute efficient frontier points. Returns DataFrame with columns: vol, ret, weights."""
    mean_ret = returns.mean().values * 252
    cov = returns.cov().values * 252
    n = len(mean_ret)
    bounds = [(0, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    min_ret = mean_ret.min()
    max_ret = mean_ret.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    results = []
    for target in target_returns:
        cons = constraints + [{"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_ret) - t}]
        w0 = np.ones(n) / n
        res = minimize(
            lambda w: np.sqrt(w @ cov @ w),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if res.success:
            port_ret, port_vol = _portfolio_stats(res.x, mean_ret, cov)
            results.append({"vol": port_vol, "ret": port_ret, "weights": res.x})

    return pd.DataFrame(results)


def max_sharpe_weights(returns: pd.DataFrame, rf: float = 0.0, allow_short: bool = False) -> np.ndarray:
    mean_ret = returns.mean().values * 252
    cov = returns.cov().values * 252
    n = len(mean_ret)
    bounds = [(-1, 1) if allow_short else (0, 1)] * n

    def neg_sharpe(w):
        ret, vol = _portfolio_stats(w, mean_ret, cov)
        return -(ret - rf) / (vol + 1e-10)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w0 = np.ones(n) / n
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    return res.x if res.success else np.ones(n) / n


def min_variance_weights(returns: pd.DataFrame, allow_short: bool = False) -> np.ndarray:
    cov = returns.cov().values * 252
    n = cov.shape[0]
    bounds = [(-1, 1) if allow_short else (0, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w0 = np.ones(n) / n
    res = minimize(
        lambda w: w @ cov @ w,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    return res.x if res.success else np.ones(n) / n


def optimal_weights_given_delta(returns: pd.DataFrame, delta: float = 3.0) -> np.ndarray:
    """Mean-variance utility maximization: max w'μ - (δ/2) w'Σw."""
    mean_ret = returns.mean().values * 252
    cov = returns.cov().values * 252
    n = len(mean_ret)
    bounds = [(0, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def neg_utility(w):
        ret = np.dot(w, mean_ret)
        var = w @ cov @ w
        return -(ret - (delta / 2) * var)

    w0 = np.ones(n) / n
    res = minimize(neg_utility, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    return res.x if res.success else np.ones(n) / n


def simulate_portfolio(weights: np.ndarray, monthly_returns: pd.DataFrame, initial: float = 10_000) -> pd.Series:
    """Simulate a rebalanced portfolio given monthly returns."""
    aligned = monthly_returns.dropna()
    port_returns = aligned @ weights
    cum = (1 + port_returns).cumprod() * initial
    cum.name = "Portfolio"
    return cum
