"""ScoreForge 基线生成模型：Gaussian / GMM(BIC) / KDE。

集大成：优先复用 scikit-learn（Tier-0 顶级开源），缺失时回退纯 NumPy 实现。
所有基线语义统一：fit(data) → sample(n, rng)。
"""

from __future__ import annotations

import numpy as np

_KRANGE_DEFAULT = range(2, 13)


def available_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


class _SkGMM:
    name = "gmm"

    def __init__(self, model) -> None:
        self.m = model

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # sklearn GaussianMixture.sample 内部使用 self.random_state（构造时固定的整数 seed），
        # 每次调用重建 RandomState(seed) → 同 seed 逐位一致，无需外部 rng。
        return np.asarray(self.m.sample(n)[0], dtype=np.float64)


class _NumpyGMM:
    """对角协方差 GMM + EM + BIC 选 k（纯 NumPy 离线兜底）。"""

    name = "gmm"

    def __init__(self, weights, means, vars_, n_iter) -> None:
        self.weights = weights
        self.means = means
        self.vars = vars_
        self.n_iter = n_iter

    def _logpdf(self, X: np.ndarray) -> np.ndarray:
        # X:[N,D] -> [N,K]
        K = self.means.shape[0]
        out = np.empty((X.shape[0], K))
        for k in range(K):
            d = X - self.means[k]
            logdet = np.sum(np.log(np.sqrt(2 * np.pi * self.vars[k])))
            mah = np.sum(d**2 / self.vars[k], axis=1)
            out[:, k] = np.log(self.weights[k] + 1e-300) - 0.5 * (logdet + mah)
        return out

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        K = self.means.shape[0]
        k = rng.choice(K, size=n, p=self.weights)
        return self.means[k] + rng.standard_normal((n, self.means.shape[1])) * np.sqrt(self.vars[k])


def fit_gmm(X: np.ndarray, seed: int, k_range=None):
    X = np.asarray(X, dtype=np.float64)
    k_range = k_range or _KRANGE_DEFAULT
    if available_sklearn():
        return _fit_gmm_sklearn(X, seed, k_range)
    return _fit_gmm_numpy(X, seed, k_range)


def _fit_gmm_sklearn(X: np.ndarray, seed: int, k_range) -> _SkGMM:
    from sklearn.mixture import GaussianMixture

    best, best_bic = None, np.inf
    for k in k_range:
        g = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=1,
            max_iter=200,
            random_state=seed,
            reg_covar=1e-6,
            tol=1e-4,
        )
        g.fit(X)
        b = g.bic(X)
        if b < best_bic:
            best_bic, best = b, g
    return _SkGMM(best)


def _fit_gmm_numpy(X: np.ndarray, seed: int, k_range) -> _NumpyGMM:
    rng = np.random.default_rng(seed)
    best = None
    best_bic = np.inf
    N, D = X.shape
    for k in k_range:
        w, mu, var = _em_diag(X, k, rng)
        # BIC
        logresp = _logresp(X, w, mu, var)
        loglik = float(np.sum(_logsumexp(logresp, axis=1)))
        n_params = k * (2 * D + 1) - 1
        bic = -2 * loglik + n_params * np.log(N)
        if bic < best_bic:
            best_bic, best = bic, (w, mu, var)
    w, mu, var = best
    return _NumpyGMM(w, mu, var, 100)


def _em_diag(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_iter: int = 100,
    tol: float = 1e-4,
):
    N, _ = X.shape
    # kmeans++ 风格初始化
    mu = X[rng.choice(N, k, replace=False)].copy()
    var = np.tile(np.var(X, axis=0, keepdims=True) + 1e-6, (k, 1))
    w = np.full(k, 1.0 / k)
    loglik_old = -np.inf
    for _ in range(n_iter):
        logresp = _logresp(X, w, mu, var)
        loglik = float(np.sum(_logsumexp(logresp, axis=1)))
        if abs(loglik - loglik_old) < tol:
            break
        loglik_old = loglik
        resp = np.exp(logresp - _logsumexp(logresp, axis=1, keepdims=True))
        Nk = resp.sum(0) + 1e-8
        w = Nk / N
        mu = (resp.T @ X) / Nk[:, None]
        d = X[:, None, :] - mu[None, :, :]
        var = (resp.T @ (d**2).sum(2)) / Nk[:, None] + 1e-6
    return w, mu, var


def _logresp(X: np.ndarray, w: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    K = mu.shape[0]
    out = np.empty((X.shape[0], K))
    for k in range(K):
        d = X - mu[k]
        logdet = np.sum(np.log(np.sqrt(2 * np.pi * var[k])))
        mah = np.sum(d**2 / var[k], axis=1)
        out[:, k] = np.log(w[k] + 1e-300) - 0.5 * (logdet + mah)
    return out


def _logsumexp(a: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    if not keepdims:
        out = np.squeeze(out, axis=(axis if axis is not None else 0))
    return out


class _NumpyKDE:
    name = "kde"

    def __init__(self, data, h) -> None:
        self.data = data
        self.h = h

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.integers(0, self.data.shape[0], n)
        return self.data[idx] + rng.standard_normal((n, self.data.shape[1])) * self.h


class _SkKDE:
    name = "kde"

    def __init__(self, data: np.ndarray, bandwidth) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.bw = np.asarray(bandwidth, dtype=np.float64)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # 自实现采样，完全等价于 sklearn KernelDensity.sample 的内部分支，但用传入的
        # rng 控制随机性（sklearn 默认 random_state=None 会走未播种的 numpy 全局态 → 非确定）。
        idx = rng.integers(0, self.data.shape[0], n)
        noise = rng.standard_normal((n, self.data.shape[1])) * self.bw
        return self.data[idx] + noise


def fit_kde(X: np.ndarray, seed: int = 0, bandwidth=None):
    X = np.asarray(X, dtype=np.float64)
    if available_sklearn():
        from sklearn.neighbors import KernelDensity

        bw = bandwidth if bandwidth is not None else "scott"
        kd = KernelDensity(kernel="gaussian", bandwidth=bw)
        kd.fit(X)
        # kd.bandwidth_ 为 fit 后解析出的带宽（标量或 per-dim 数组），采样严格等价。
        return _SkKDE(X, kd.bandwidth_)
    # 纯 NumPy 兜底：Silverman 带宽
    n, d = X.shape
    std = X.std(0)
    h = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4)) * np.median(std) + 1e-3
    return _NumpyKDE(X, h)


class _Gaussian:
    name = "gaussian"

    def __init__(self, mean, cov) -> None:
        self.mean = mean
        self.cov = cov

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.cov, size=n)


def fit_gaussian(X: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(0)
    cov = np.cov(X.T)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    return _Gaussian(mean, cov)
