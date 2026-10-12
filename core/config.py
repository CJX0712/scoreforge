"""全局配置：ENV_SCOREFORGE_* 覆盖 + 校验。"""

from __future__ import annotations

import os

from .errors import ConfigError

_ENV_PREFIX = "SCOREFORGE_"


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(_ENV_PREFIX + name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError as exc:
        raise ConfigError(f"{_ENV_PREFIX}{name} 非法整数: {v!r}") from exc


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(_ENV_PREFIX + name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError as exc:
        raise ConfigError(f"{_ENV_PREFIX}{name} 非法浮点: {v!r}") from exc


def _env_str(name: str, default: str) -> str:
    return os.environ.get(_ENV_PREFIX + name, default)


class Config:
    """Benchmark / pipeline 顶层配置，可由环境变量覆盖。"""

    def __init__(
        self,
        seeds: tuple = (7, 42, 123),
        dgps: tuple = ("moons", "ring8", "grid25", "checkerboard"),
        n_train: int = 2000,
        n_sample: int = 2000,
        steps: int = 3000,
        batch: int = 256,
        lr: float = 2.0e-3,
        schedule: str = "cosine",
        T: int = 1000,
        sampler: str = "ddim",
        sampler_steps: int = 50,
        eta: float = 0.0,
        out: str = "benchmark.json",
        net_dims: tuple = (64, 64, 2),
        time_emb_dim: int = 16,
    ) -> None:
        self.seeds = seeds
        self.dgps = dgps
        self.n_train = n_train
        self.n_sample = n_sample
        self.steps = steps
        self.batch = batch
        self.lr = lr
        self.schedule = schedule
        self.T = T
        self.sampler = sampler
        self.sampler_steps = sampler_steps
        self.eta = eta
        self.out = out
        self.net_dims = net_dims
        self.time_emb_dim = time_emb_dim

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            seeds=tuple(_env_int("SEEDS", 7)),
            dgps=tuple(_env_str("DGPS", "moons,ring8,grid25,checkerboard").split(",")),
            n_train=_env_int("N_TRAIN", 2000),
            n_sample=_env_int("N_SAMPLE", 2000),
            steps=_env_int("STEPS", 3000),
            batch=_env_int("BATCH", 256),
            lr=_env_float("LR", 2.0e-3),
            schedule=_env_str("SCHEDULE", "cosine"),
            T=_env_int("T", 1000),
            sampler=_env_str("SAMPLER", "ddim"),
            sampler_steps=_env_int("SAMPLER_STEPS", 50),
            eta=_env_float("ETA", 0.0),
            out=_env_str("OUT", "benchmark.json"),
        )

    def validate(self) -> None:
        if self.steps <= 0:
            raise ConfigError("steps 必须 > 0")
        if self.batch <= 0:
            raise ConfigError("batch 必须 > 0")
        if self.T <= 0:
            raise ConfigError("T 必须 > 0")
        if self.sampler not in ("ddim", "ddpm"):
            raise ConfigError(f"sampler 非法: {self.sampler}")
        if self.schedule not in ("cosine", "linear"):
            raise ConfigError(f"schedule 非法: {self.schedule}")
        if self.sampler_steps <= 0 or self.sampler_steps > self.T:
            raise ConfigError("sampler_steps 必须在 (0, T] 区间")
        if not self.dgps:
            raise ConfigError("dgps 不能为空")
        if not self.seeds:
            raise ConfigError("seeds 不能为空")
