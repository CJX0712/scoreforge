"""DDPM 祖先采样（Ho et al. 2020）。"""

from __future__ import annotations

import numpy as np

from .schedule import NoiseSchedule


def sample_ddpm(
    net,
    schedule: NoiseSchedule,
    n: int,
    rng: np.random.Generator,
    data_dim: int,
    clip: float = 8.0,
) -> np.ndarray:
    """从 x_T ~ N(0,I) 逐步去噪到 x_0。clip 防止 cosine 调度末端 alpha_cum→0 反演爆炸。"""
    T = schedule.T
    x = rng.standard_normal((n, data_dim))
    for i in range(T - 1, -1, -1):
        t_norm = np.full(n, i / T)
        eps = net.predict_eps(x, t_norm)
        alpha_cum = schedule.alpha_cum[i]
        alpha = schedule.alphas[i]
        alpha_cum_prev = schedule.alpha_cum[i - 1] if i > 0 else 1.0
        beta = schedule.betas[i]
        beta_tilde = beta * (1.0 - alpha_cum_prev) / (1.0 - alpha_cum)
        x0_pred = np.clip(schedule.predict_x0_from_eps(x, i, eps), -clip, clip)
        mean = (np.sqrt(alpha_cum_prev) * beta / (1.0 - alpha_cum)) * x0_pred + (
            np.sqrt(alpha) * (1.0 - alpha_cum_prev) / (1.0 - alpha_cum)
        ) * x
        if i > 0:
            noise = rng.standard_normal((n, data_dim))
            x = mean + np.sqrt(beta_tilde) * noise
        else:
            x = mean
    return x
