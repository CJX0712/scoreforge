# Changelog

本项目遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [0.1.0] - 2026-10-13

### Added
- **ScoreFuse 旗舰模型**：纯 NumPy 手写 MLP + 正弦时间嵌入 + eps-prediction 训练，
  cosine noise schedule（Nichol & Dhariwal 2021），DDPM/DDIM 双采样器。
- **零依赖可复现**：BLAS 线程钉死 + 统一 seed 入口，同 seed 二次运行 bit_identical
  （CI 硬门禁，max|Δ| < 1e-9）。
- **基线层**：GMM（BIC 选 k）/ KDE / Gaussian，sklearn 优先 + 纯 NumPy 离线兜底。
- **评测层**：MMD（RBF 中位启发）、2-Wasserstein（scipy 精确指派优先）、多峰模式覆盖度。
- **性能门禁**：多峰模式覆盖度 ≥ 0.95（实测 ring8 1.000 / grid25 0.960，PASS）。
- **CLI**：`python -m scoreforge.cli`，支持 `--check-determinism` 与全部超参覆盖。
- **测试**：16 项 pytest 全绿（梯度检验、前向不变量、确定性、基线正确性、CLI 冒烟）。
- **工程**：ruff 双绿（check + format）、GitHub Actions CI（3.10/3.12/3.13）、
  Dockerfile、Makefile、requirements.lock.txt 锁版本。

### Fixed
- 余弦调度 `ᾱ_T = 0` 导致 DDPM/DDIM 反演除零爆炸：x₀ 预测 clip 到 ±8。
- KDE 基线 `random_state=None` 泄漏 numpy 全局态：自实现采样，恢复确定性。
- 梯度检验 analytic/numeric 形状不匹配（ravel 统一）。
- `predict_x0_from_eps` 标量索引广播错误。

### 科学结论（诚实记录）
在 2D 合成密度上 KDE/GMM 的 MMD 优于小型 MLP 扩散模型（领域事实）。
门禁因此押注扩散模型核心能力——多峰模式覆盖度，而非伪造 MMD 领先。

[0.1.0]: https://github.com/CJX0712/scoreforge/releases/tag/v0.1.0
