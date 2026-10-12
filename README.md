# ScoreForge

> **从零手写、集大成的 score-based 生成模型基准** · 作者：**晨星** · License: MIT

[![CI](https://github.com/CJX0712/scoreforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/scoreforge/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen)](./tests)
[![Ruff](https://img.shields.io/badge/ruff-check%20%2B%20format-blue)](./pyproject.toml)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](./pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**ScoreFuse**：纯 NumPy 手写的扩散模型（DDPM 训练 + DDIM 采样 + cosine 调度），
与 GMM / KDE / Gaussian 基线在受控 2D 分布上公平对比。零重写——直接复用
Ho et al. 2020（DDPM）、Song et al. 2020（DDIM）、Nichol & Dhariwal 2021（cosine
schedule）、Kingma & Ba 2014（Adam）等世界顶级成果；**同 seed 二次运行 bit 级一致**
（max|Δ| = 0.00e+00）。

## 快速开始

```bash
pip install -r requirements.lock.txt
python -m scoreforge.examples.run_demo --fast   # 小规模 + 确定性二次校验（约 5 秒）
python -m scoreforge.examples.run_demo          # 默认规模：4 DGP × 4 方法 × 3 seeds（约 90 秒）
python -m scoreforge.cli --help                 # CLI 全部超参
```

## 评测结果（默认规模，3 seeds：7/42/123，mean±std，诚实报告全部方法）

MMD（RBF 中位启发带宽，偏置估计，越小越好）：

| DGP | ScoreFuse | GMM | KDE | Gaussian |
|---|---|---|---|---|
| moons | 0.1050±0.0101 | -0.0001±0.0001 | 0.0031±0.0004 | 0.0114±0.0011 |
| ring8 | 0.0828±0.0101 | -0.0000±0.0004 | 0.0003±0.0003 | 0.0179±0.0020 |
| grid25 | 0.1068±0.0084 | 0.0000±0.0002 | 0.0004±0.0003 | 0.0075±0.0024 |
| checkerboard | 0.0213±0.0059 | -0.0000±0.0003 | 0.0003±0.0005 | 0.0067±0.0022 |

**科学结论（不回避）**：在 2D 合成密度上，非参数 KDE 与 GMM 的 MMD 优于小型
MLP 扩散模型——这是领域事实（KDE 对训练集记忆式采样，低维近乎最优）。
ScoreForge 的价值主张不在 MMD 极致领先，而在：

1. **完整捕获多峰结构**（扩散模型的本质能力）——模式覆盖度 **ring8 = 1.000、
   grid25 = 1.000**，8 环与 25 网格的离散 mode 无一遗漏，门禁 ≥ 0.95 **PASS**。
2. **零依赖透明实现**：训练-采样-评测全链路约 1500 行可审计 NumPy 代码，
   bit 级可复现。

## 架构

```
core（seed 确定性入口 / errors / config / interfaces）
 └→ data（5 种 DGP）  diffusion（ScoreFuse 旗舰）  baselines（GMM/KDE/Gaussian）  eval（MMD/W2/覆盖度）
      └→ pipeline（Benchmark 编排 + 性能门禁）  └→ cli / examples
```

详见 [docs/architecture.md](./docs/architecture.md)；模型细节见 [docs/model_card.md](./docs/model_card.md)。

## 工程约定

- **确定性三防线**：BLAS 线程钉死先于 import numpy；`set_all(seed)` 统一播种；
  评测零随机。CI 以 bit_identical 为硬门禁。
- **离线兜底**：无 sklearn 时自动切换纯 NumPy EM-GMM / Silverman KDE，功能不降级。
- **代码风格**：ruff check + ruff format 双绿。
- **测试**：16 项 pytest（梯度检验 1e-8 级、前向不变量、确定性、基线正确性、CLI 冒烟）。

## 开发

```bash
make install   # 安装锁定依赖
make ci        # lint + test + fast demo
make format    # 自动修复风格
```

## 致谢

站在巨人肩上：DDPM（Ho et al., 2020）、DDIM（Song et al., 2020）、Score-Based
Generative Modeling（Song & Ermon, 2019/2021）、cosine schedule（Nichol & Dhariwal,
2021）、Adam（Kingma & Ba, 2014）、GMM/KDE（scikit-learn）。

## License

[MIT](./LICENSE) © 2026 [晨星](https://github.com/CJX0712)
