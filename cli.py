"""ScoreForge 命令行入口：运行 benchmark 并打印 / 落盘报告。"""

from __future__ import annotations

# 关键：必须在 import numpy（经 .core.config / .pipeline）之前钉死 BLAS 线程。
# numpy 在 import 时即绑定多线程策略，之后再用环境变量设置无效，会导致 matmul
# 非确定 → 同 seed 二次运行 MMD 抖动。故线程钉死必须最先执行。
from .core.seed import pin_threads

pin_threads(1)

import argparse
import math

from .core.config import Config
from .core.seed import set_all
from .pipeline import Benchmark


def build_parser() -> argparse.ArgumentParser:
    _argument = argparse.ArgumentParser(
        description="ScoreForge — score-based 生成模型基准（DDPM / DDIM）"
    )
    _argument.add_argument("--dgps", default=",".join(Config().dgps))
    _argument.add_argument("--seeds", default="7,42,123")
    _argument.add_argument("--n-train", type=int, default=2000)
    _argument.add_argument("--n-sample", type=int, default=2000)
    _argument.add_argument("--steps", type=int, default=3000)
    _argument.add_argument("--batch", type=int, default=256)
    _argument.add_argument("--lr", type=float, default=2.0e-3)
    _argument.add_argument("--schedule", default="cosine")
    _argument.add_argument("--T", type=int, default=1000)
    _argument.add_argument("--sampler", default="ddim")
    _argument.add_argument("--sampler-steps", type=int, default=50)
    _argument.add_argument("--eta", type=float, default=0.0)
    _argument.add_argument("--out", default="benchmark.json")
    _argument.add_argument("--check-determinism", action="store_true", help="二次运行校验逐位一致")
    return _argument


def _parse_seeds(s: str) -> tuple:
    return tuple(int(x) for x in s.split(",") if x.strip() != "")


def print_report(payload: dict) -> None:
    cfg = payload["config"]
    print("=" * 78)
    print(
        f" ScoreForge benchmark  ·  author 晨星  ·  "
        f"flags: {cfg['schedule']}/{cfg['sampler']} T={cfg['T']} steps={cfg['steps']}"
    )
    print("=" * 78)
    methods = sorted({r["method"] for r in payload["reports"]})
    dgps = cfg["dgps"]
    # MMD 表
    print("\n MMD（越小越好，mean±std over seeds）")
    header = f" {'method':12s}" + "".join(f"{d:>14s}" for d in dgps)
    print(header)
    for m in methods:
        row = f" {m:12s}"
        for d in dgps:
            rep = next(
                (r for r in payload["reports"] if r["method"] == m and r["dgp"] == d),
                None,
            )
            if rep:
                row += f"{rep['mmd_mean']:12.4f}±{rep['mmd_std']:.4f}"
            else:
                row += f"{'':>14s}"
        print(row)
    # headline
    h = payload["headline"]
    cov = h["multimodal_mode_coverage_mean"]
    cov_s = f"{cov:.3f}" if not math.isnan(cov) else "n/a"
    print("\n 性能门禁（多峰 DGP · 扩散核心能力：模式覆盖度；Wasserstein 仅参考见主表）")
    print(
        f"   多峰模式覆盖度 mean = {cov_s}  (门槛 ≥ {h['coverage_gate_threshold']:.2f})"
        f"  →  {'PASS ✅' if h['gate_pass'] else '未达 ⚠️'}"
    )
    for d, v in h["per_dgp_coverage"].items():
        c = v["scorefuse_cov"]
        cs = f"{c:.3f}" if not math.isnan(c) else "n/a"
        print(f"   {d:12s} scorefuse_cov={cs}")
    print("=" * 78)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    set_all(_parse_seeds(args.seeds)[0] if args.seeds else 7)
    cfg = Config(
        seeds=_parse_seeds(args.seeds),
        dgps=tuple(s for s in args.dgps.split(",") if s),
        n_train=args.n_train,
        n_sample=args.n_sample,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        schedule=args.schedule,
        T=args.T,
        sampler=args.sampler,
        sampler_steps=args.sampler_steps,
        eta=args.eta,
        out=args.out,
    )
    bench = Benchmark(cfg)
    payload = bench.run_report(args.out)
    print_report(payload)

    if args.check_determinism:
        r1 = bench.run()
        r2 = bench.run()
        max_diff = 0.0
        for a, b in zip(r1, r2):
            for key in ("mmd", "wasserstein", "coverage"):
                if getattr(a, key) == getattr(a, key) and getattr(b, key) == getattr(b, key):
                    max_diff = max(max_diff, abs(getattr(a, key) - getattr(b, key)))
        ok = max_diff < 1e-9
        print(
            f" 确定性二次校验：max|Δ|={max_diff:.2e}  →  {'bit_identical ✅' if ok else 'FAIL ❌'}"
        )
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
