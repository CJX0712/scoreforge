"""ScoreForge 合成 2D 数据生成器（DGP）。

所有生成器固定 seed 可复现；同时提供「真实采样」用于评测与基线的 fit 数据。
域覆盖：连续流形（moons / swiss_roll）、不相交多峰（ring8 / grid25）、不连续（checkerboard）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DGP:
    """一个数据生成过程。"""

    name: str
    dim: int = 2
    # 真实 mode 中心（用于 coverage / mode 计数）；无 mode 结构的 DGP 留空。
    mode_centers: np.ndarray | None = None
    mode_radius: float = 0.4


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rng_or(rng, seed):
    return rng if rng is not None else _rng(seed)


# ----------------------------- 生成器实现 -----------------------------


def moons(
    n: int, rng: np.random.Generator | None = None, seed: int = 0, noise: float = 0.06
) -> np.ndarray:
    """两个交错半圆（half-moons）。"""
    rng = _rng_or(rng, seed)
    n1 = n // 2
    n2 = n - n1
    t = np.linspace(0, np.pi, n1)[:, None]
    x1 = np.concatenate([np.cos(t) - 0.5, np.sin(t)], axis=1)
    t2 = np.linspace(0, np.pi, n2)[:, None]
    x2 = np.concatenate([np.cos(t2) + 0.5, -np.sin(t2) + 1.0], axis=1)
    x = np.concatenate([x1, x2], axis=0)
    x += rng.standard_normal(x.shape) * noise
    return x


def ring8(
    n: int,
    rng: np.random.Generator | None = None,
    seed: int = 0,
    R: float = 2.0,
    sigma: float = 0.08,
) -> np.ndarray:
    """圆周上等距 8 个窄高斯（不相交多峰）。"""
    rng = _rng_or(rng, seed)
    centers = np.array(
        [[R * np.cos(2 * np.pi * k / 8), R * np.sin(2 * np.pi * k / 8)] for k in range(8)]
    )
    idx = rng.integers(0, 8, n)
    x = centers[idx] + rng.standard_normal((n, 2)) * sigma
    return x


def grid25(
    n: int,
    rng: np.random.Generator | None = None,
    seed: int = 0,
    span: float = 2.4,
    sigma: float = 0.08,
) -> np.ndarray:
    """5×5 网格上的 25 个窄高斯（不相交多峰）。"""
    rng = _rng_or(rng, seed)
    coords = np.linspace(-span, span, 5)
    centers = np.array([[a, b] for a in coords for b in coords])
    idx = rng.integers(0, 25, n)
    x = centers[idx] + rng.standard_normal((n, 2)) * sigma
    return x


def checkerboard(
    n: int,
    rng: np.random.Generator | None = None,
    seed: int = 0,
    grid: int = 4,
    half: float = 2.0,
) -> np.ndarray:
    """[-half, half]^2 内交替着色的方格（不连续均匀密度）。"""
    rng = _rng_or(rng, seed)
    cell = (2 * half) / grid
    # 仅选取 (i+j) 为偶的格子
    cells = [(i, j) for i in range(grid) for j in range(grid) if (i + j) % 2 == 0]
    idx = rng.integers(0, len(cells), n)
    x = np.empty((n, 2))
    for k, (i, j) in enumerate(cells):
        m = (idx == k).sum()
        if m == 0:
            continue
        low = np.array([-half + i * cell, -half + j * cell])
        x[idx == k] = low + rng.random((m, 2)) * cell
    return x


def swiss_roll(
    n: int, rng: np.random.Generator | None = None, seed: int = 0, noise: float = 0.05
) -> np.ndarray:
    """二维投影的瑞士卷（连续弯曲流形）。"""
    rng = _rng_or(rng, seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    x = t[:, None] * np.array([np.cos(t), np.sin(t)]).T / 6.0
    x += rng.standard_normal((n, 2)) * noise
    return x


# ----------------------------- DGP 登记表 -----------------------------


def get_dgp(name: str) -> DGP:
    if name == "moons":
        return DGP(name="moons", mode_centers=None)
    if name == "ring8":
        centers = np.array(
            [[2.0 * np.cos(2 * np.pi * k / 8), 2.0 * np.sin(2 * np.pi * k / 8)] for k in range(8)]
        )
        return DGP(name="ring8", mode_centers=centers, mode_radius=0.45)
    if name == "grid25":
        coords = np.linspace(-2.4, 2.4, 5)
        centers = np.array([[a, b] for a in coords for b in coords])
        return DGP(name="grid25", mode_centers=centers, mode_radius=0.4)
    if name == "checkerboard":
        return DGP(name="checkerboard", mode_centers=None)
    if name == "swiss_roll":
        return DGP(name="swiss_roll", mode_centers=None)
    raise ValueError(f"未知 DGP: {name}")


def sample_true(
    name: str, n: int, rng: np.random.Generator | None = None, seed: int = 0
) -> np.ndarray:
    if name == "moons":
        return moons(n, rng, seed)
    if name == "ring8":
        return ring8(n, rng, seed)
    if name == "grid25":
        return grid25(n, rng, seed)
    if name == "checkerboard":
        return checkerboard(n, rng, seed)
    if name == "swiss_roll":
        return swiss_roll(n, rng, seed)
    raise ValueError(f"未知 DGP: {name}")
