"""ScoreForge 核心数据类型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """扩散模型训练 / 采样配置（语义统一：分数越大越异常 不适用，此处为生成模型）。"""

    seed: int = 7
    steps: int = 3000
    batch: int = 256
    lr: float = 2.0e-3
    net_dims: tuple = (64, 64, 2)  # 隐藏层 + 输出层（输出维度 = 数据维度）
    time_emb_dim: int = 16
    schedule: str = "cosine"  # cosine | linear
    T: int = 1000
    sampler: str = "ddim"  # ddim | ddpm
    sampler_steps: int = 50
    eta: float = 0.0  # DDIM 随机性；0 = 确定性
    n_train: int = 2000

    def input_dim(self, data_dim: int) -> int:
        return data_dim + self.time_emb_dim


@dataclass
class EvalResult:
    """单条评测记录（一个方法 × 一个 DGP × 一个 seed）。"""

    method: str
    dgp: str
    seed: int
    mmd: float
    wasserstein: float
    coverage: float
    elapsed_sec: float


@dataclass
class MethodReport:
    """某方法在某 DGP 上的聚合报告（≥3 seeds mean±std）。"""

    method: str
    dgp: str
    mmd_mean: float = 0.0
    mmd_std: float = 0.0
    wasserstein_mean: float = 0.0
    wasserstein_std: float = 0.0
    coverage_mean: float = 0.0
    coverage_std: float = 0.0
    n_seeds: int = 0
