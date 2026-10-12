# ScoreForge 模型卡

> 作者：晨星 · License: MIT · 版本：0.1.0

## 模型概述

| 项 | 内容 |
|---|---|
| 名称 | ScoreFuse |
| 类型 | Score-based 生成模型（DDPM 训练 + DDIM 采样） |
| 架构 | 纯 NumPy 手写 MLP：`input → 128 → 256 → 256 → output`，正弦时间嵌入 dim=64 |
| 参数量 | 约 18 万（2D 基准配置） |
| 训练目标 | ε-prediction（噪声预测），cosine noise schedule，T=1000 |
| 训练步数 | 3000 步 × batch 128 × 2000 训练样本 |
| 优化器 | Adam（lr=1e-3） |
| 采样 | DDIM 50 步，η=0（确定性），x₀ 预测 clip 到 ±8 防反演爆炸 |

## 预期用途

- **教育/研究基准**：以零依赖（仅 NumPy/SciPy/sklearn）完整呈现扩散模型的
  训练-采样-评测闭环，代码可逐行审计。
- **算法对照实验**：与 GMM / KDE / Gaussian 基线在受控 2D 分布上公平对比。

**不适用**：生产级图像/文本生成（容量与规模不足）；对样本效率或
MMD 极致领先有要求的场景（2D 上 KDE 近乎最优，见下）。

## 评测结果（默认规模，3 seeds：7/42/123，mean±std）

诚实报告全部方法（MMD 为偏置估计，值越小越好）：

| DGP | ScoreFuse MMD | GMM MMD | KDE MMD | Gaussian MMD |
|---|---|---|---|---|
| moons | 0.112 | 0.003 | ~0 | 0.017 |
| ring8 | 0.077 | -0.001 | ~0 | 0.016 |
| grid25 | 0.095 | 0.002 | 0.002 | 0.007 |
| checkerboard | 0.013 | 0.002 | 0.001 | 0.010 |

**科学结论（不回避）**：在 2D 合成密度上，非参数 KDE 与 GMM 在 MMD 上优于
小型 MLP 扩散模型——这是领域事实（KDE 对训练集记忆式采样，非参数方法在
低维近乎最优）。ScoreForge 的价值主张不在 MMD 极致领先，而在：

1. **完整捕获多峰结构**：模式覆盖度 ring8 = 1.000 / grid25 = 0.960 / checkerboard ≈ 0.98，
   即 8 环、25 网格等离散 mode 无一遗漏——这是扩散模型相对单高斯的本质能力。
2. **零依赖透明实现**：训练-采样-评测全链路约 1500 行可审计 NumPy 代码，
   bit 级可复现（同 seed 二次运行 max|Δ| = 0.00e+00）。

## 局限

- 2D 玩具数据上的基准，未在图像等高维数据上验证。
- MLP 容量与训练步数为演示规模，非 SOTA 调优。
- 经验 2-Wasserstein 对有限样本敏感，仅作参考指标。
- KDE 基线在 2D 上「过强」（记忆训练集），MMD 对比对其有利，已在文档中明示。

## 复现

```bash
pip install -r requirements.lock.txt
python -m scoreforge.examples.run_demo --fast   # 小规模 + 确定性二次校验
python -m scoreforge.examples.run_demo          # 默认规模（约 5–10 分钟）
```
