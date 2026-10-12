"""ScoreForge 评测指标：MMD / 2-Wasserstein / 模式覆盖度。

纯 NumPy 实现主路径；2-Wasserstein 在有 scipy 时用精确指派，否则回退切片 Wasserstein。
"""

from __future__ import annotations

import numpy as np


def available_scipy() -> bool:
    try:
        import scipy  # noqa: F401

        return True
    except Exception:
        return False


def _median_gamma(x: np.ndarray, y: np.ndarray) -> float:
    z = np.vstack([x, y])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(z.shape[0], 1)
    med = float(np.median(d2[iu]))
    return 1.0 / med if med > 0 else 1.0


def _rbf_mean(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    def kmat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a2 = (a**2).sum(1)
        b2 = (b**2).sum(1)
        d2 = a2[:, None] + b2[None, :] - 2.0 * a @ b.T
        return np.exp(-gamma * np.clip(d2, 0.0, None))

    kxx = kmat(x, x)
    kyy = kmat(y, y)
    kxy = kmat(x, y)
    # 去偏估计（移除对角线）
    n = kxx.shape[0]
    m = kyy.shape[0]
    kxx_off = (kxx.sum() - n) / max(n * (n - 1), 1)
    kyy_off = (kyy.sum() - m) / max(m * (m - 1), 1)
    kxy_m = kxy.mean()
    return kxx_off + kyy_off - 2.0 * kxy_m


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """最大均值差异（RBF，中位启发式带宽，去偏估计）。越小越好。"""
    if gamma is None:
        gamma = _median_gamma(x, y)
    return float(_rbf_mean(x, y, gamma))


def _sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_proj: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    dim = x.shape[1]
    total = 0.0
    for _ in range(n_proj):
        theta = rng.standard_normal(dim)
        theta /= np.linalg.norm(theta) + 1e-12
        px = np.sort(x @ theta)
        py = np.sort(y @ theta)
        total += float(np.mean((px - py) ** 2))
    return float(np.sqrt(total / n_proj))


def wasserstein2(x: np.ndarray, y: np.ndarray, max_n: int = 1000) -> float:
    """2-Wasserstein 距离（经验/子采样，仅供参考）。

    注：经验 2-Wasserstein 对有限样本数 n 敏感（n 越小，样本计数失配越易抬高数值），
    故仅作辅助参考，不作为性能门禁。有 scipy 用精确指派，否则切片 Wasserstein。
    """
    if x.shape[0] > max_n or y.shape[0] > max_n:
        x = x[:max_n]
        y = y[:max_n]
    if available_scipy():
        from scipy.optimize import linear_sum_assignment

        d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
        row, col = linear_sum_assignment(d2)
        return float(np.sqrt(float(d2[row, col].mean())))
    return _sliced_wasserstein(x, y)


def _min_dist_to(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d2 = ((points[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
    return np.min(d2, axis=1)


def coverage(
    generated: np.ndarray,
    true: np.ndarray,
    mode_centers: np.ndarray | None = None,
    mode_radius: float = 0.4,
) -> float:
    """模式覆盖度：给定真实 mode 中心时，被生成样本覆盖（≤ radius）的比例。"""
    if mode_centers is None:
        return float("nan")
    if generated.size == 0:
        return 0.0
    covered = 0
    for c in mode_centers:
        d = float(np.min(np.linalg.norm(generated - c, axis=1)))
        if d <= mode_radius:
            covered += 1
    return covered / len(mode_centers)


def evaluate(
    generated: np.ndarray,
    true: np.ndarray,
    mode_centers: np.ndarray | None = None,
    mode_radius: float = 0.4,
) -> dict:
    """返回某次生成的评测指标字典。"""
    return {
        "mmd": mmd_rbf(generated, true),
        "wasserstein": wasserstein2(generated, true),
        "coverage": coverage(generated, true, mode_centers, mode_radius),
    }
