"""噪声调度（noise schedule）与 DDPM 前向过程。

复用：
  - Ho et al. 2020 线性 beta 调度（DDPM）
  - Nichol & Dhariwal 2021 余弦调度（Improved DDPM）
索引约定：betas 为长度 T 的数组，betas[i] 对应时间步 i+1（0-based）。
"""

from __future__ import annotations

import numpy as np


def cosine_betas(T: int, s: float = 0.008) -> np.ndarray:
    """Nichol & Dhariwal 2021 余弦调度。"""
    x = np.linspace(0.0, float(T), T + 1) / T
    f = np.cos((x + s) / (1.0 + s) * np.pi / 2.0) ** 2
    f = f / f[0]
    betas = 1.0 - f[1:] / f[:-1]
    return np.clip(betas, 1e-4, 0.999)


def linear_betas(T: int, beta_0: float = 1e-4, beta_T: float = 0.02) -> np.ndarray:
    """Ho et al. 2020 线性调度。"""
    return np.linspace(beta_0, beta_T, T)


class NoiseSchedule:
    """预计算扩散过程系数，提供 q_sample 与前向 x0 反演。"""

    def __init__(self, T: int, betas: np.ndarray) -> None:
        self.T = T
        self.betas = np.asarray(betas, dtype=np.float64).reshape(-1)
        if self.betas.shape[0] != T:
            raise ValueError("betas 长度必须等于 T")
        self.alphas = 1.0 - self.betas
        self.alpha_cum = np.cumprod(self.alphas)
        self.sqrt_alpha_cum = np.sqrt(self.alpha_cum)
        self.sqrt_1m_alpha_cum = np.sqrt(1.0 - self.alpha_cum)

    def q_sample(self, x0: np.ndarray, t_idx: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """x_t = sqrt(barα_t) x0 + sqrt(1-barα_t) ε。t_idx ∈ [0, T-1] 或标量。"""
        t = np.atleast_1d(np.asarray(t_idx, dtype=np.int64)).reshape(-1, 1)
        return self.sqrt_alpha_cum[t] * x0 + self.sqrt_1m_alpha_cum[t] * noise

    def predict_x0_from_eps(self, xt: np.ndarray, t_idx: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """由噪声预测反演 x0。t_idx ∈ [0, T-1] 或标量。"""
        t = np.atleast_1d(np.asarray(t_idx, dtype=np.int64)).reshape(-1, 1)
        return (xt - self.sqrt_1m_alpha_cum[t] * eps) / self.sqrt_alpha_cum[t]


def make_schedule(kind: str, T: int) -> NoiseSchedule:
    if kind == "cosine":
        return NoiseSchedule(T, cosine_betas(T))
    if kind == "linear":
        return NoiseSchedule(T, linear_betas(T))
    raise ValueError(f"未知 schedule: {kind}")
