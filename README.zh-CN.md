# CITES 贸易优先级

一套可复现的排序流程，用于**固定复核名额下野生来源 CITES 贸易流**的优先级判断。

场景：复核团队每年只能细看少数“出口国—物种”流，而描述这些流的年度报告往往迟到、
还会补报。流程只使用决策时点可得的信息，排出“下一年最可能出现激增”的名单。

仓库只含分析代码与图件，**不包含贸易记录**。

```
src/       面板构建、标签、特征、实验、出图
figures/   6 张图件（600 dpi PNG）
docs/      数据字典、方法说明、图件清单
output/    运行生成的表（git 忽略）
```

---

## 一、排的是什么

一条观测是**出口国 × 物种 × 年**。标签年 `T` 的候选，是 `T-3 … T-1` 至少有一年
野生来源记录的单元。2016–2024 年测试窗口中，**每年候选 3,129–4,136 个**。

标签是“下一年是否激增”的二值变量，对应 CITES 重大贸易审查里的两条 sharp increase
判据：

- **本国激增**：`T` 年野生量超过前五年均值的 4 倍，且均值不低于 100；
- **全球激增**：该物种全球出口量超过其前五年均值的 4 倍，且本国占全球量不低于 10%。

阈值在建模前固定；3 倍与 5 倍作为敏感性检验。

## 二、为什么比较不是走过场

拿到年度汇总后，复核团队本来就能做三件事：照搬上年名单、按上年涨幅排序、按上年出口量
排序。这三条就是事后基线。问题是：只用更早信息的模型，能否在同样 100 个名额上同时
赢过三者。

能，但优势不大，且随年份变化。九年里模型前 100 名命中 **5–26** 个真激增；最强的
事后基线命中 **2–7** 个。前 100 只占候选的 2.4–3.2%，却覆盖当年激增的 21–48%。

更有意思的是构成：295 个正例中 **89%** 不在上年正例名单上。只会照搬上年的名单够不到
它们。新案检验把上年正例从测试池中剔除再排序，模型仍能找到相当数量。

## 三、数据

数据为 **CITES Trade Database**（v2026.1，1975–2024），发布地址
<https://trade.cites.org/>。本仓库使用的快照取于 2026-10-16，1975–2024 年共
29,158,995 条记录。

仓库不重新分发任何记录。代码读取本地 DuckDB 中名为 `trade` 的表；列名与一行导入命令
见 [docs/data.zh-CN.md](docs/data.zh-CN.md)。

分析保留出口方报告、直接出口、计件单位的记录。标签与候选池用野生来源量
（`Source = W`）；特征用全部来源的直接出口面板，并显式加入各来源占比，使模型能区分
野生与人工繁殖贸易，而不是把所有记录当成同一种东西。

## 四、快速开始

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 先按 docs/data.zh-CN.md 建好 data/cites.duckdb
CITES_DB=data/cites.duckdb python src/run_all.py
```

运行后生成：

| 文件 | 内容 |
|---|---|
| `output/panel.parquet` | 出口国 × 物种 × 年面板 |
| `output/labels.parquet` | 二值激增标签 |
| `output/feat_T*.parquet` | 截至各年的 60 个特征 |
| `output/main_table.csv` | 逐年 AP、precision、recall、hits |
| `output/baselines.csv` | 三条事后基线 |
| `output/new_case_table.csv` | 剔除上年正例后的结果 |
| `output/overlap.json` | 名单重叠与延续率 |
| `output/permutation.csv` | 标签打乱对照 |
| `output/shap_features.csv` | 各特征的平均绝对 SHAP |
| `output/diag_*.png` | 简版诊断图 |

全链只用 CPU。一次完整运行训练 27 个模型（9 年 × 3 种子）外加对照拟合；4 个线程足够。

## 五、方法概要

- **60 个特征**，全部截至 `T-1`：15 个贸易史、18 个网络位置、27 个形态与分类。
  完整定义见 [docs/methods.zh-CN.md](docs/methods.zh-CN.md)。
- **模型**：LightGBM 二分类，300 轮，学习率 0.05，31 叶，叶最小样本 50，
  特征与行采样 0.8，类别权重 = 负样本数 / 正样本数。
- **协议**：滚动起点。每个测试年用 2005 … `T-1` 训练，预测 `T`，每年重训；
  不在测试年上调参。
- **指标**：AP、precision@100、recall@100、hits@100。
- **检查**：新案通道、前 100 重叠、标签打乱、SHAP 分组。

## 六、图件

| 文件 | 面板 |
|---|---|
| [fig01_overview.png](figures/fig01_overview.png) | 一页交代数据、决策点与结果 |
| [fig02_new_case_channel.png](figures/fig02_new_case_channel.png) | 剔除上年正例后的命中；名单重叠 |
| [fig03_what_the_model_uses.png](figures/fig03_what_the_model_uses.png) | SHAP 分布与组占比 |
| [fig04_case_series.png](figures/fig04_case_series.png) | 三个示例时间序列 |
| [fig05_year_by_year.png](figures/fig05_year_by_year.png) | 逐年 AP 与命中对最强基线 |
| [fig06_robustness_and_taxa.png](figures/fig06_robustness_and_taxa.png) | 八年复核、阈值、区间、类群 |

每张图看什么、读哪些数，见 [docs/figures.zh-CN.md](docs/figures.zh-CN.md)。

## 七、范围说明

- 2025 年不纳入：快照时只有约 11,000 条，而正常年份约 100–135 万条。
- 2016 年是异常强的一年（55 个正例、26 个命中）。文中单独展示，并在八年复核中剔除；
  方向不依赖它。
- 排序回答“先看谁”，不回答“多少贸易是可持续的”。它是复核输入，不是非致危性判定。
- 公开代码是整理后的分析链；内部探索分支与一次性检查未纳入。

## 八、作者与许可

作者：**廖彬（Liao Bin）**，2026。

MIT 许可，见 [LICENSE](LICENSE)。
