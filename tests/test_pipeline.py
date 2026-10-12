"""Pipeline 聚合 / 门禁 / 端到端小尺度运行。"""

from scoreforge.core.config import Config
from scoreforge.pipeline import Benchmark


def test_run_produces_results():
    cfg = Config(
        seeds=(7,),
        dgps=("ring8", "grid25"),
        n_train=400,
        n_sample=400,
        steps=150,
        batch=64,
        sampler_steps=20,
    )
    b = Benchmark(cfg)
    res = b.run()
    # 2 dgps × 1 seed × 4 methods = 8
    assert len(res) == 8
    for r in res:
        assert r.mmd == r.mmd  # 非 nan
        assert r.coverage == r.coverage


def test_headline_coverage_gate():
    cfg = Config(
        seeds=(7,),
        dgps=("ring8", "grid25"),
        n_train=800,
        n_sample=800,
        steps=600,
        batch=128,
    )
    b = Benchmark(cfg)
    res = b.run()
    h = b.headline(res)
    # 小尺度下 ScoreFuse 应能捕获多峰 → 覆盖度门禁应通过
    assert h["gate_pass"] is True, f"覆盖度门禁未通过：{h['multimodal_mode_coverage_mean']}"


def test_run_report_writes_json(tmp_path):
    out = tmp_path / "bench.json"
    cfg = Config(
        seeds=(7,),
        dgps=("ring8",),
        n_train=400,
        n_sample=400,
        steps=150,
        batch=64,
        sampler_steps=20,
        out=str(out),
    )
    b = Benchmark(cfg)
    payload = b.run_report(str(out))
    assert out.exists()
    assert "headline" in payload and "results" in payload
