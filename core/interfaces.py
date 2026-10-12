"""生成模型与基线统一接口（Protocol）。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GenerativeModel(Protocol):
    """任何生成模型都实现 fit / sample。"""

    name: str

    def fit(self, data: np.ndarray, cfg: object) -> GenerativeModel:
        """在 data（[N, D]）上拟合；返回 self 以便链式。"""
        ...

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """生成 n 个样本，形状 [n, D]。"""
        ...
