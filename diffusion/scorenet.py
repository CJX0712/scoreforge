"""ScoreNet：小型 MLP 噪声预测网络 + 正弦时间嵌入 + 手写反向传播 + Adam。

纯 NumPy 实现（零第三方运行时依赖），反向传播用解析梯度，并有独立梯度检验。
复用了顶级算法范式：DDPM 的 eps-prediction 参数化（Ho et al. 2020）。
"""

from __future__ import annotations

import numpy as np


def time_embedding(t_norm: np.ndarray, dim: int) -> np.ndarray:
    """正弦时间嵌入（Vaswani 2017 风格），t_norm ∈ [0,1]，输出 [B, dim]。"""
    t_norm = np.asarray(t_norm, dtype=np.float64).reshape(-1)
    half = dim // 2
    freqs = np.exp(np.linspace(np.log(1.0), np.log(1000.0), half))
    args = t_norm[:, None] * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=1)


class MLP:
    """小型全连接 MLP（tanh 隐藏 + 线性输出），前向/后向手写。"""

    def __init__(self, input_dim: int, net_dims: tuple, seed: int) -> None:
        self.dims = [int(input_dim)] + [int(d) for d in net_dims]
        rng = np.random.default_rng(seed)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(self.dims) - 1):
            scale = np.sqrt(2.0 / self.dims[i])
            self.W.append(rng.standard_normal((self.dims[i], self.dims[i + 1])) * scale)
            self.b.append(np.zeros(self.dims[i + 1]))
        self.h: list[np.ndarray] = []
        self.out: np.ndarray | None = None

    def params(self) -> list[np.ndarray]:
        p: list[np.ndarray] = []
        for w, b in zip(self.W, self.b):
            p.append(w)
            p.append(b)
        return p

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.h = [np.asarray(x, dtype=np.float64)]
        a = self.h[0]
        self.out = None
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                a = np.tanh(z)
                self.h.append(a)
            else:
                self.out = z
        return self.out

    def backward(self, dout: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        grads_W = [np.zeros_like(w) for w in self.W]
        grads_b = [np.zeros_like(b) for b in self.b]
        dz = np.asarray(dout, dtype=np.float64)
        for i in range(len(self.W) - 1, -1, -1):
            a_prev = self.h[i]
            grads_W[i] = a_prev.T @ dz
            grads_b[i] = dz.sum(axis=0)
            if i > 0:
                da = dz @ self.W[i].T
                dh = da * (1.0 - self.h[i] ** 2)
                dz = dh
        return grads_W, grads_b


class Adam:
    """最小 Adam 优化器（Kingma & Ba 2015）。"""

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, g in enumerate(grads):
            if g is None:
                continue
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1**self.t)
            vhat = self.v[i] / (1 - self.b2**self.t)
            self.params[i] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class ScoreNet:
    """噪声预测网络：输入 = [x_t, time_embedding(t)]，输出 = ε 预测。"""

    def __init__(self, data_dim: int, time_emb_dim: int, net_dims: tuple, seed: int) -> None:
        self.data_dim = data_dim
        self.time_emb_dim = time_emb_dim
        self.input_dim = data_dim + time_emb_dim
        self.mlp = MLP(self.input_dim, net_dims, seed)

    def _input(self, x: np.ndarray, t_norm: np.ndarray) -> np.ndarray:
        emb = time_embedding(np.asarray(t_norm, dtype=np.float64), self.time_emb_dim)
        return np.concatenate([x, emb], axis=1)

    def predict_eps(self, x_t: np.ndarray, t_norm: np.ndarray) -> np.ndarray:
        return self.mlp.forward(self._input(x_t, t_norm))

    def parameters(self) -> list[np.ndarray]:
        return self.mlp.params()

    def train_step(
        self, optimizer: Adam, x0: np.ndarray, t_idx: np.ndarray, T: int, schedule, rng
    ) -> float:
        """单步训练：MSE(ε_pred, ε_true)，返回标量 loss。noise 用传入 Generator，保证确定性。"""
        B = x0.shape[0]
        t_idx = np.asarray(t_idx, dtype=np.int64)
        noise = rng.standard_normal(x0.shape)
        x_t = schedule.q_sample(x0, t_idx, noise)
        t_norm = t_idx / T
        eps_pred = self.predict_eps(x_t, t_norm)
        loss = float(np.mean((eps_pred - noise) ** 2))
        dout = 2.0 * (eps_pred - noise) / B
        grads_W, grads_b = self.mlp.backward(dout)
        grads = []
        for gw, gb in zip(grads_W, grads_b):
            grads.append(gw)
            grads.append(gb)
        optimizer.step(grads)
        return loss


def gradient_check(
    score_dim: int = 4, time_emb_dim: int = 8, seed: int = 0, eps: float = 1e-5
) -> float:
    """数值 vs 解析梯度最大相对误差（用于 CI 不变量门禁）。"""
    net = MLP(score_dim + time_emb_dim, (8, 6, score_dim), seed)
    rng = np.random.default_rng(123)
    x = rng.standard_normal((5, score_dim + time_emb_dim))
    # 构造一个随机标量目标：out 经随机矩阵求和
    Wt = rng.standard_normal((score_dim,))
    net.forward(x)
    dout = np.tile(Wt, (5, 1))
    ga_W, _ga_b = net.backward(dout)
    analytic = ga_W[0].ravel()
    # 数值梯度对 W[0]
    num = np.zeros_like(net.W[0])
    W0 = net.W[0]
    for idx in np.ndindex(W0.shape):
        orig = W0[idx]
        W0[idx] = orig + eps
        fp = (net.forward(x) @ Wt).sum()
        W0[idx] = orig - eps
        fm = (net.forward(x) @ Wt).sum()
        W0[idx] = orig
        num[idx] = (fp - fm) / (2 * eps)
    denom = np.abs(analytic) + np.abs(np.asarray(num).ravel()) + 1e-12
    return float(np.max(np.abs(analytic - np.asarray(num).ravel()) / denom))
