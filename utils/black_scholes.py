import numpy as np
from scipy.stats import norm


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return float("inf") if S >= K else float("-inf")
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    d1 = _d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        return (term1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    return (term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% move in vol


def bs_implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """Newton-Raphson implied volatility solver."""
    sigma = 0.2
    for _ in range(max_iter):
        if option_type == "call":
            price = bs_call_price(S, K, T, r, sigma)
        else:
            price = bs_put_price(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma) * 100  # undo the /100 in vega
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        if abs(vega) < 1e-10:
            break
        sigma -= diff / vega
        sigma = max(sigma, 1e-6)
    return sigma


# Unit tests — run with: python -m utils.black_scholes
if __name__ == "__main__":
    p = bs_call_price(100, 100, 1, 0.05, 0.2)
    assert abs(p - 10.4506) < 0.001, f"BS call price wrong: {p}"
    p2 = bs_put_price(100, 100, 1, 0.05, 0.2)
    # Put-call parity: C - P = S - K*e^(-rT)
    pcp = p - p2 - (100 - 100 * np.exp(-0.05 * 1))
    assert abs(pcp) < 0.001, f"Put-call parity violated: {pcp}"
    print("All BS unit tests passed.")
