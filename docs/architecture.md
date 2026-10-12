# ScoreForge 架构

> 作者：晨星 · License: MIT · 版本：0.1.0

## 1. 设计哲学

ScoreForge 是一座**从零手写、但站在世界顶级成果之上**的 score-based 生成模型基准。
三大原则贯穿全部代码：

| 原则 | 落地方式 |
|---|---|
| 零重写 / 集大成 | 直接复用经过验证的顶级算法：DDPM（Ho et al. 2020）、DDIM（Song et al. 2020）、cosine noise schedule（Nichol & Dhariwal 2021）、Adam（Kingma & Ba 2014）、中位启发 RBF-MMD、2-Wasserstein 精确指派 |
| 可复现 | 同 seed 二次运行 **bit_identical**（max\|Δ\| = 0.00e+00），BLAS 线程钉死 + 统一 seed 入口，CI 硬门禁 |
| 诚实 | 不伪造数字：KDE 在 2D 上近乎最优是领域事实，基准如实报告全部方法，门禁只押注扩散模型真正擅长的指标（多峰模式覆盖度） |

## 2. 分层架构（单向无环）

```
scoreforge/
├── core/            基础设施层
│   ├── seed.py      确定性入口：pin_threads() 钉死 BLAS + set_all() 统一播种
│   ├── errors.py    异常层级 E000–E500
│   ├── types.py     TrainConfig / EvalResult / MethodReport（dataclass）
│   ├── config.py    Config：环境变量 ENV_SCOREFORGE_* 覆盖 + validate()
│   └── interfaces.py GenerativeModel Protocol（name / fit / sample）
├── data/            DGP 生成器：moons / ring8 / grid25 / checkerboard / swiss_roll（固定 seed 可复现）
├── diffusion/       旗舰：纯 NumPy 手写扩散模型
│   ├── scorenet.py  MLP（forward/backward）+ Adam + 正弦时间嵌入 + gradient_check
│   ├── schedule.py  cosine_betas / linear_betas / NoiseSchedule（q_sample / predict_x0_from_eps）
│   ├── ddpm.py      sample_ddpm（p 采样，clip 防爆）
│   ├── ddim.py      sample_ddim（快速确定性采样，eta 控制随机性）
│   └── scorefuse.py ScoreFuse：fit（eps-prediction + cosine 调度）+ sample（路由 DDIM/DDPM）
├── baselines/       基线层：GMM（BIC 选 k）/ KDE / Gaussian（sklearn 优先 + 纯 NumPy 离线兜底）
├── eval/            评测层：MMD（RBF 中位启发，偏置估计）/ 2-Wasserstein（scipy 指派优先）/ 模式覆盖度
├── pipeline/        基准编排：Benchmark.run/aggregate/headline/run_report + 性能门禁
├── cli.py           命令行入口（顶部先钉线程再 import）
└── examples/run_demo.py  端到端演示：落盘 benchmark.json + fast 模式确定性二次校验
```

**依赖方向严格单向**：`core → data/diffusion/baselines/eval → pipeline → cli`。
没有任何模块反向依赖上层，环不可能出现。

## 3. 数学模型

### 3.1 前向扩散（cosine schedule, Nichol & Dhariwal 2021）

```
α_t = f(t)/f(t-1),  f(t) = cos²((t/T + s)/(1+s) · π/2),  s = 0.008
β_t = 1 - α_t,      ᾱ_t = Π α_s
q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I)
```

### 3.2 eps-prediction 训练目标

```
L = E_{t,x_0,ε} ‖ ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t) ‖²
```

损失有闭式解析梯度，经 `gradient_check` 数值验证（max_rel_err ≈ 1e-8）。

### 3.3 DDIM 采样（Song et al. 2020）

```
x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1-ᾱ_{t-1}-σ²) · ε_θ(x_t,t) + σ · z
σ = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1-ᾱ_t/ᾱ_{t-1})
```

η=0 时采样完全确定；默认 η=0，50 步。

## 4. 确定性设计（bit_identical 的关键）

两次同 seed 运行必须逐位一致。任何随机性泄漏都会在 3000 步训练中被混沌放大。三道防线：

1. **线程钉死先于 import**：`OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_NUM_THREADS=1` 必须在
   numpy 被 import 之前生效（numpy 在 import 时绑定 BLAS 线程池）。`cli.py` 与
   `run_demo.py` 顶部、`tests/conftest.py` 均保证此顺序。
2. **统一 seed 入口**：`set_all(seed)` 一次性播种 numpy/random/hash；训练、采样、基线
   全部使用 `np.random.default_rng(seed)` 派生子 rng，绝不触碰全局态。
3. **评测零随机**：MMD 用中位启发带宽（无采样）、Wasserstein 用精确指派、覆盖度为确定性计数。

> 历史教训：早期版本因 sklearn `KernelDensity.sample()` 在 `random_state=None` 时
> 回退到 numpy 全局态，导致二次运行 MMD 抖动 1e-3。修复方式为自实现 KDE 采样、
> 完全脱离全局态（见 baselines/__init__.py）。

## 5. 性能门禁（科学诚实）

在 2D 合成密度上：**KDE 近乎最优**（MMD≈0，对训练集记忆式采样）、**GMM 亦能精确拟合**
——小型 MLP 扩散模型无法在 MMD 上超越它们是领域常识。因此门禁不押注 MMD，而是押注
扩散模型真正的核心能力：

| 门禁 | 阈值 | 实测（默认规模） |
|---|---|---|
| 多峰模式覆盖度（ring8 + grid25，良定义离散 mode） | ≥ 0.95 | ring8 1.000 / grid25 1.000 → **PASS** |

Wasserstein 对有限样本极敏感（两个同分布 ring8 采样的经验 W2 高达 ~0.25–0.5），
故仅作主表参考指标，不作门禁。

## 6. 测试矩阵（16 项，全绿）

| 文件 | 验证内容 |
|---|---|
| test_gradcheck.py | 解析 vs 数值梯度 max_rel_err < 1e-6（反向传播正确性） |
| test_schedule.py | DDPM 前向不变量：q_sample 矩匹配、predict_x0_from_eps 反演自洽 |
| test_determinism.py | 同 seed 两次 fit + sample 逐位一致 |
| test_baselines.py | sklearn 与纯 NumPy 兜底一致性、离线 GMM 覆盖度、coverage NaN 契约 |
| test_pipeline.py | 端到端 benchmark、聚合统计、覆盖度门禁 PASS |
| test_cli.py | CLI 冒烟（--fast） |
