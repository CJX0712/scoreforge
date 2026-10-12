"""ScoreForge 错误码（E100~E500），统一异常层级。"""

from __future__ import annotations


class ScoreForgeError(Exception):
    """所有 ScoreForge 异常的基类。"""

    code = "E000"


class ConfigError(ScoreForgeError):
    """配置 / 参数校验失败。"""

    code = "E100"


class DataError(ScoreForgeError):
    """数据生成 / 载入失败。"""

    code = "E200"


class TrainingError(ScoreForgeError):
    """模型训练失败。"""

    code = "E300"


class EvalError(ScoreForgeError):
    """评测 / 指标计算失败。"""

    code = "E400"


class DependencyError(ScoreForgeError):
    """可选依赖缺失且无可降级路径。"""

    code = "E500"
