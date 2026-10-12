"""全局确定性：统一随机种子入口 + BLAS 线程钉死（保证同 seed 逐位一致）。

必须在 import numpy 之前设置线程环境变量，否则 BLAS 多线程会引入非确定性。
run_demo.py / cli.py 已在最顶部设置；此处再做一次保险。
"""

from __future__ import annotations

import os
import random

_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def pin_threads(threads: int = 1) -> None:
    """钉死所有 BLAS/线程库为单线程，消除 numpy 矩阵乘的非确定性。"""
    for k in _THREAD_KEYS:
        os.environ[k] = str(threads)


def set_all(seed: int) -> int:
    """一次性设齐 Python / numpy / hash 种子，并钉死线程。返回归一化后的 seed。"""
    pin_threads(1)
    seed = int(seed) & 0x7FFFFFFF
    random.seed(seed)
    # numpy 在 set_all 之前若已 import，矩阵乘已绑定线程数；线程钉死仍可降低抖动。
    try:
        import numpy as _np  # 延迟 import 以兼容调用顺序（set_all 可能早于 numpy 被调用）

        _np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy 应始终可用
        pass
    os.environ["PYTHONHASHSEED"] = str(seed % (2**32))
    return seed


def make_rng(*salt: int) -> object:
    """由主 seed 派生子 rng（default_rng，确定性且独立于全局 numpy 状态）。"""
    base = int(salt[0]) if salt else 0
    for s in salt[1:]:
        base = (base * 2654435761 + int(s)) & 0xFFFFFFFF
    try:
        import numpy as _np

        return _np.random.default_rng(base)
    except Exception:  # pragma: no cover
        import random as _random

        return _random.Random(base)
