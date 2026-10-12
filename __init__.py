"""ScoreForge — 世界顶级 score-based 生成模型（DDPM / DDIM）实现。

复用顶级算法（非自研 SOTA）：
  - Ho et al. 2020, Denoising Diffusion Probabilistic Models (DDPM)
  - Song et al. 2020, Denoising Diffusion Implicit Models (DDIM)
  - Nichol & Dhariwal 2021, Improved DDPM（cosine noise schedule）
  - Song & Ermon 2019/2021, Score-Based Generative Modeling（统一框架）

作者：晨星
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星"

from .core.seed import set_all

__all__ = ["__author__", "__version__", "set_all"]
