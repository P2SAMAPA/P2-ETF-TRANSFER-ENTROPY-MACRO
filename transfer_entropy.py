import numpy as np
from scipy.stats import entropy

def discretise(series, n_bins=5):
    """Discretise series into n_bins equally spaced bins."""
    min_val, max_val = series.min(), series.max()
    if max_val - min_val < 1e-12:
        return np.zeros_like(series, dtype=int)
    bins = np.linspace(min_val, max_val, n_bins+1)
    return np.digitize(series, bins[1:-1])

def transfer_entropy(X, Y, lag=1, n_bins=5):
    """
    Transfer entropy from X to Y: TE(X -> Y) = I(Y_{t+lag}; X_t | Y_t)
    """
    if len(X) != len(Y):
        min_len = min(len(X), len(Y))
        X = X[:min_len]
        Y = Y[:min_len]
    if len(X) <= lag + 2:
        return 0.0
    # Discretise
    Xd = discretise(X, n_bins)
    Yd = discretise(Y, n_bins)
    # Create tuples for joint states
    n = len(Yd) - lag
    y_future = Yd[lag:]
    y_present = Yd[:-lag]
    x_present = Xd[:-lag]
    # Count joint occurrences
    def joint_counts(x, y, z=None):
        if z is None:
            return { (xi, yi): 0 for xi, yi in zip(x, y) }
        else:
            return { (xi, yi, zi): 0 for xi, yi, zi in zip(x, y, z) }
    # TE = H(Y_future | Y_present) - H(Y_future | X_present, Y_present)
    # First term: H(Y_future | Y_present)
    counts_yy = {}
    for yf, yp in zip(y_future, y_present):
        counts_yy[(yf, yp)] = counts_yy.get((yf, yp), 0) + 1
    total = len(y_future)
    H_cond1 = 0.0
    for (yf, yp), cnt in counts_yy.items():
        pyf_yp = cnt / sum(1 for yp_ in y_present if yp_ == yp)
        pyf_yp = max(pyf_yp, 1e-12)
        H_cond1 += (cnt / total) * (-np.log2(pyf_yp))
    # Second term: H(Y_future | X_present, Y_present)
    counts_xyy = {}
    for yf, xp, yp in zip(y_future, x_present, y_present):
        counts_xyy[(yf, xp, yp)] = counts_xyy.get((yf, xp, yp), 0) + 1
    H_cond2 = 0.0
    for (yf, xp, yp), cnt in counts_xyy.items():
        denom = sum(1 for (_, xp_, yp_) in counts_xyy.keys() if xp_ == xp and yp_ == yp)
        p_yf_xp_yp = cnt / denom if denom > 0 else 0
        p_yf_xp_yp = max(p_yf_xp_yp, 1e-12)
        H_cond2 += (cnt / total) * (-np.log2(p_yf_xp_yp))
    TE = max(H_cond1 - H_cond2, 0.0)
    return TE

def transfer_entropy_score(returns, macro_df, lag=1, n_bins=5):
    """
    For each ETF, compute the average transfer entropy from all macro variables.
    Score = mean TE(macro -> ETF) over selected macros.
    """
    scores = {}
    for ticker in returns.columns:
        te_list = []
        for macro_col in macro_df.columns:
            macro_series = macro_df[macro_col].values
            if len(macro_series) < len(returns):
                # Trim to same length
                min_len = min(len(macro_series), len(returns))
                te = transfer_entropy(macro_series[:min_len], returns[ticker].values[:min_len], lag, n_bins)
            else:
                te = transfer_entropy(macro_series, returns[ticker].values, lag, n_bins)
            te_list.append(te)
        scores[ticker] = np.mean(te_list) if te_list else 0.0
    return scores
