# 数据

本仓库不存放任何贸易记录。`output/` 下的内容全部由代码生成，且已被 git 忽略。

## 来源

数据为 CITES 秘书处与 UNEP-WCMC 发布的 **CITES Trade Database**：

- 门户：<https://trade.cites.org/>
- 使用版本：**v2026.1**，覆盖 1975–2024 年
- 快照日期：2026-10-16
- 1975–2024 年共 29,158,995 条记录

从门户下载完整贸易数据（CSV）。代码读取一个本地 DuckDB 文件中的 `trade` 表。
一种建库方式：

```bash
duckdb data/cites.duckdb -c \
  "CREATE TABLE trade AS SELECT * FROM read_csv_auto('trade.csv', header=true);"
```

然后运行：

```bash
CITES_DB=data/cites.duckdb python src/run_all.py
```

## 代码读取的列

下表是批量导出中的列名。如果你的导出用了别的列名，请先重命名。

| 列 | 用途 |
|---|---|
| `Year` | 贸易年份 |
| `Exporter` | 出口国 |
| `Importer` | 进口国 |
| `Origin` | 原产国；直接出口为 null 或 `Origin = Exporter` |
| `Taxon` | 学名 |
| `Class`、`Order`、`Family`、`Genus` | 分类学编码 |
| `Appendix` | CITES 附录（I、II、III、N） |
| `Reporter.type` | `E` 出口方报告，`I` 进口方报告 |
| `Quantity` | 报告数量 |
| `Unit` | 数量单位，如 `Number of specimens` |
| `Term` | 产品形态（活体、肉、鳍……） |
| `Purpose` | 用途代码（T、P、H……） |
| `Source` | 来源代码（W 野生、C 人工繁殖、F、R、D、A、U） |

## 固定口径

分析只保留出口方报告、直接出口、计件单位的记录：

```sql
"Reporter.type" = 'E'
AND (Origin IS NULL OR Origin = Exporter)
AND (Unit IS NULL OR Unit = 'Number of specimens')
```

这是任务选择，不是“别的口径不对”：出口端记录对应许可证签发与复核动作发生的时点；
进口方报告更适合事后核对。

## 派生文件（不随仓库发布）

| 文件 | 内容 |
|---|---|
| `output/panel.parquet` | 出口国 × 物种 × 年的汇总面板 |
| `output/labels.parquet` | 次年激增的二值标签 |
| `output/feat_T*.parquet` | 截至第 t 年的 60 个特征 |
| `output/*.csv` | 逐年指标、基线、置换检验、SHAP |

2025 年数据不纳入：快照时只有约 11,000 条，而正常年份约 100–135 万条。
