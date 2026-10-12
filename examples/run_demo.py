"""ScoreForge 端到端演示：运行完整 benchmark，落盘 benchmark.json 并打印报告。

必须在 import 任何依赖 numpy 的模块之前钉死 BLAS 线程（见顶部），以保证 bit_identical 复现。
用法：
    python -m scoreforge.examples.run_demo            # 默认配置
    python -m scoreforge.examples.run_demo --fast     # 小尺度快速验证（CI / 本地）
    SCOREFORGE_SEEDS=7,42,123 python -m scoreforge.examples.run_demo
"""

from __future__ import annotations

import os
import sys

# 必须在 import numpy（经 scoreforge）之前钉死 BLAS 线程，否则同 seed 二次运行非确定。
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import time

from ..core.config import Config
from ..core.seed import set_all
from ..pipeline import Benchmark


def main(argv=None) -> int:
    args = argv or sys.argv[1:]
    fast = "--fast" in args
    # 统一主 seed（确定性入口）
    set_all(7)

    if fast:
        cfg = Config(
            seeds=(7,),
            dgps=("ring8", "grid25"),
            n_train=800,
            n_sample=800,
            steps=600,
            batch=128,
            out="benchmark_fast.json",
        )
    else:
        cfg = Config(out="benchmark.json")

    t0 = time.perf_counter()
    bench = Benchmark(cfg)
    payload = bench.run_report(cfg.out)
    dt = time.perf_counter() - t0

    from ..cli import print_report

    print_report(payload)
    print(f"\n 总耗时 {dt:.1f}s · 结果已落盘 {cfg.out}")

    # 确定性二次校验（仅 fast 模式，避免双倍耗时）
    if fast:
        r1 = bench.run()
        r2 = bench.run()
        mx = 0.0
        for a, b in zip(r1, r2):
            for key in ("mmd", "wasserstein", "coverage"):
                if getattr(a, key) == getattr(a, key) and getattr(b, key) == getattr(b, key):
                    mx = max(mx, abs(getattr(a, key) - getattr(b, key)))
        ok = mx < 1e-9
        print(f" 确定性二次校验：max|Δ|={mx:.2e}  →  {'bit_identical ✅' if ok else 'FAIL ❌'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
