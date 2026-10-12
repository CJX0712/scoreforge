"""CLI 冒烟：端到端运行并验证退出码与门禁输出。"""

from scoreforge.cli import main


def test_cli_small_run(capsys):
    rc = main(
        [
            "--dgps",
            "ring8",
            "--seeds",
            "7",
            "--steps",
            "150",
            "--n-train",
            "400",
            "--n-sample",
            "400",
            "--out",
            "benchmark_cli_test.json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out or "未达" in out  # 门禁结论已打印


def test_cli_determinism_flag(capsys):
    rc = main(
        [
            "--dgps",
            "ring8",
            "--seeds",
            "7",
            "--steps",
            "150",
            "--n-train",
            "400",
            "--n-sample",
            "400",
            "--check-determinism",
            "--out",
            "benchmark_cli_det.json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "bit_identical" in out
