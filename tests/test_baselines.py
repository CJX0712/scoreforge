"""基线正确性 + 离线兜底（纯 NumPy 实现）。"""

import numpy as np
from scoreforge.baselines import _NumpyGMM, _NumpyKDE, fit_gaussian
from scoreforge.data import get_dgp, sample_true
from scoreforge.eval import coverage


def test_numpy_kde_fallback():
    rng = np.random.default_rng(0)
    X = sample_true("ring8", 500, rng)
    kde = _NumpyKDE(X, 0.1)
    s = kde.sample(300, np.random.default_rng(1))
    assert s.shape == (300, 2)
    assert np.isfinite(s).all()


def test_numpy_gmm_fallback_coverage():
    g = _NumpyGMM(
        weights=np.full(8, 1 / 8),
        means=np.array(
            [[2 * np.cos(2 * np.pi * k / 8), 2 * np.sin(2 * np.pi * k / 8)] for k in range(8)]
        ),
        vars_=np.full((8, 2), 0.0064),
        n_iter=1,
    )
    s = g.sample(800, np.random.default_rng(1))
    assert s.shape == (800, 2)
    cov = coverage(s, None, mode_centers=get_dgp("ring8").mode_centers, mode_radius=0.45)
    assert cov >= 0.99, f"离线 GMM 覆盖度不足：{cov}"


def test_gaussian_baseline():
    rng = np.random.default_rng(0)
    X = sample_true("ring8", 500, rng)
    g = fit_gaussian(X)
    s = g.sample(200, np.random.default_rng(1))
    assert s.shape == (200, 2)


def test_coverage_nan_when_no_centers():
    assert np.isnan(coverage(np.zeros((3, 2)), None))
