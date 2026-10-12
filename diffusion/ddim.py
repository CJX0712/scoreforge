"""DDIM 确定性/随机采样（Song et al. 2020），支持子步数加速。"""

from __future__ import annotations

import numpy as np

from .schedule import NoiseSchedule


def _ddim_indices(T: int, ddim_steps: int) -> list[int]:
    """生成自 T-1 到 0（含）的降序去重整数索引。"""
    if ddim_steps >= T:
        return list(range(T - 1, -1, -1))
    idx = np.linspace(0, T - 1, ddim_steps).round().astype(int)
    indices = sorted(set(idx.tolist()), reverse=True)
    if indices[0] != T - 1:  # 保证从最噪步开始
        indices = [T - 1, *indices]
    return indices


def sample_ddim(
    net,
    schedule: NoiseSchedule,
    n: int,
    rng: np.random.Generator,
    data_dim: int,
    ddim_steps: int = 50,
    eta: float = 0.0,
    clip: float = 8.0,
) -> np.ndarray:
    """DDIM 采样：eta=0 为确定性，eta>0 引入随机性。clip 防止末端反演爆炸。"""
    T = schedule.T
    indices = _ddim_indices(T, ddim_steps)
    x = rng.standard_normal((n, data_dim))
    for step, i in enumerate(indices):
        t_norm = np.full(n, i / T)
        eps = net.predict_eps(x, t_norm)
        alpha_cum = schedule.alpha_cum[i]
        alpha_cum_prev = schedule.alpha_cum[i - 1] if i > 0 else 1.0
        x0_pred = np.clip(schedule.predict_x0_from_eps(x, i, eps), -clip, clip)
        ratio = (1.0 - alpha_cum_prev) / (1.0 - alpha_cum)
        sigma = eta * np.sqrt(ratio * (1.0 - alpha_cum / alpha_cum_prev))
        dir_term = np.sqrt(np.clip(1.0 - alpha_cum_prev - sigma**2, 0.0, None)) * eps
        x = np.sqrt(alpha_cum_prev) * x0_pred + dir_term
        if eta > 0.0 and step < len(indices) - 1:
            x = x + sigma * rng.standard_normal((n, data_dim))
    return x
