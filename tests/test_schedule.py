"""调度前向不变量：q_sample 加噪后，用 predict_x0_from_eps 应能还原 x0。"""

import numpy as np
from scoreforge.diffusion.schedule import make_schedule


def test_predict_x0_recovers_x0():
    for sched_name in ("cosine", "linear"):
        sch = make_schedule(sched_name, T=1000)
        rng = np.random.default_rng(0)
        x0 = rng.standard_normal((128, 2))
        t = rng.integers(0, 1000, 128)
        noise = rng.standard_normal(x0.shape)
        x_t = sch.q_sample(x0, t, noise)
        x0_hat = sch.predict_x0_from_eps(x_t, t, noise)
        # 在 t 较大的高噪步，还原误差应仍很小（解析关系精确）
        assert np.max(np.abs(x0_hat - x0)) < 1e-6, f"{sched_name} 还原误差过大"


def test_alpha_cum_monotonic_nonincreasing():
    sch = make_schedule("cosine", T=1000)
    # alpha_cum 应随 t 增大（噪声更多）而单调不增
    assert np.all(np.diff(sch.alpha_cum) <= 1e-9)


def test_cosine_alpha_cum_terminal():
    # Nichol cosine 在 t=T 处 alpha_cum→0，采样端 clip 已处理；此处仅验证为正且递减
    sch = make_schedule("cosine", T=1000)
    assert sch.alpha_cum[-1] >= 0.0
    assert sch.alpha_cum[0] > sch.alpha_cum[-1]
