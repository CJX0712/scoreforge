"""ScoreFuse 旗舰：DDPM 训练 + cosine 调度 + DDIM 加速采样。

集大成：复用 DDPM（Ho 2020）/ DDIM（Song 2020）/ cosine 调度（Nichol 2021）。
默认配置即为本系统最强组合。
"""

from __future__ import annotations

import numpy as np

from ..core.types import TrainConfig
from .ddim import sample_ddim
from .ddpm import sample_ddpm
from .schedule import make_schedule
from .scorenet import Adam, ScoreNet


class ScoreFuse:
    """score-based 生成旗舰模型。"""

    name = "scorefuse"

    def __init__(self, cfg: TrainConfig | None = None) -> None:
        self.cfg = cfg or TrainConfig()
        self.net: ScoreNet | None = None
        self.schedule = None

    def fit(self, data: np.ndarray, cfg: TrainConfig | None = None) -> ScoreFuse:
        cfg = cfg or self.cfg
        data = np.asarray(data, dtype=np.float64)
        data_dim = data.shape[1]
        T = cfg.T
        schedule = make_schedule(cfg.schedule, T)
        self.schedule = schedule
        rng = np.random.default_rng(cfg.seed)
        net = ScoreNet(data_dim, cfg.time_emb_dim, cfg.net_dims, seed=cfg.seed)
        opt = Adam(net.parameters(), cfg.lr)
        for _step in range(cfg.steps):
            B = cfg.batch
            idx = rng.integers(0, data.shape[0], B)
            x0 = data[idx]
            t_idx = rng.integers(0, T, B)
            net.train_step(opt, x0, t_idx, T, schedule, rng)
        self.net = net
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.net is None or self.schedule is None:
            raise RuntimeError("请先 fit()")
        if self.cfg.sampler == "ddim":
            return sample_ddim(
                self.net,
                self.schedule,
                n,
                rng,
                self.net.data_dim,
                ddim_steps=self.cfg.sampler_steps,
                eta=self.cfg.eta,
            )
        return sample_ddpm(self.net, self.schedule, n, rng, self.net.data_dim)


# 兼容 GenerativeModel Protocol 的工厂
def make_scorefuse(cfg: TrainConfig) -> ScoreFuse:
    return ScoreFuse(cfg)
