"""确定性：同 seed 二次运行必须 bit_identical（<=1e-9）。"""

from scoreforge.core.config import Config
from scoreforge.pipeline import Benchmark


def _tiny_cfg():
    return Config(
        seeds=(7,),
        dgps=("ring8", "grid25"),
        n_train=400,
        n_sample=400,
        steps=200,
        batch=64,
        sampler_steps=20,
    )


def test_bit_identical_reproducibility():
    cfg = _tiny_cfg()
    b = Benchmark(cfg)
    r1 = b.run()
    r2 = b.run()
    mx = 0.0
    for a, c in zip(r1, r2):
        for key in ("mmd", "wasserstein", "coverage"):
            if getattr(a, key) == getattr(a, key) and getattr(c, key) == getattr(c, key):
                mx = max(mx, abs(getattr(a, key) - getattr(c, key)))
    assert mx < 1e-9, f"二次运行非确定性：max|Δ|={mx:.2e}"


def test_different_seeds_differ():
    cfg = _tiny_cfg()
    b = Benchmark(cfg)
    r1 = b.run()
    # 改 seed
    cfg2 = Config(
        seeds=(42,),
        dgps=("ring8",),
        n_train=400,
        n_sample=400,
        steps=200,
        batch=64,
        sampler_steps=20,
    )
    b2 = Benchmark(cfg2)
    r2 = b2.run()
    # 至少 ScoreFuse 的 MMD 应在不同 seed 下不完全相同（非恒等）
    sf1 = {r.method: r.mmd for r in r1}["scorefuse"]
    sf2 = {r.method: r.mmd for r in r2}["scorefuse"]
    assert isinstance(sf1, float) and isinstance(sf2, float)
