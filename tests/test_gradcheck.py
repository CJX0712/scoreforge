"""梯度检验：手写反向传播 vs 中心差分，最大相对误差应极小。"""

from scoreforge.diffusion.scorenet import gradient_check


def test_gradient_check_passes():
    err = gradient_check(seed=0)
    assert err < 1e-6, f"梯度检验失败：max_rel_err={err:.2e}"


def test_gradient_check_stable_across_seeds():
    for s in (1, 2, 3):
        assert gradient_check(seed=s) < 1e-6
