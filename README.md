# CITES Trade Priority

A reproducible ranking pipeline for **wild-sourced CITES trade flows under a
fixed review budget**.

The setting: a review team can look closely at only a small number of
exporter–species flows each year, while the annual reports that describe those
flows arrive late and are often revised. The pipeline ranks the flows that are
most likely to show a sharp increase in the next year, using only information
available at the decision point.

This repository contains analysis code and a figure set. It does **not**
contain trade records.

```
src/        panel construction, labels, features, experiment, figures
figures/    six figures (600 dpi PNG)
docs/       data dictionary, method notes, figure inventory
output/     generated tables (ignored by git)
```

---

## 1. What is being ranked

One observation is an **exporter × taxon × year** cell. A cell is a candidate
for label year `T` if it has at least one wild-source record in `T-3 … T-1`.
Across the 2016–2024 test window this gives **3,129–4,136 candidates per
year**.

The label is a binary "sharp increase next year" flag. It follows the two
sharp-increase criteria used in the CITES Review of Significant Trade:

- **country surge** — wild volume in `T` exceeds four times the previous
  five-year mean, and that mean is at least 100;
- **global surge** — world exports of the taxon exceed four times their previous
  five-year mean, and the country holds at least 10% of them.

The thresholds are fixed before modelling. Three- and five-fold thresholds are
reported as a sensitivity check.

## 2. Why the comparison is not trivial

A review team that receives the annual summary can already do three things:
copy last year's list, rank by last year's growth, or rank by last year's
volume. These are the post-hoc baselines. The question is whether a model that
uses only older information can beat all three on the same 100 slots.

It can, but the margin is modest and year-dependent. Over nine test years the
model finds **5–26** true surges in its top 100; the strongest post-hoc
baseline finds **2–7**. The top 100 is 2.4–3.2% of the candidate pool but
covers 21–48% of that year's surges.

The more interesting result is compositional: **89%** of the 295 positives were
not on last year's positive list. A list that merely repeats last year cannot
reach them. The new-case check removes last year's positives from the test pool
and asks the model to rank again; it still finds a substantial number.

## 3. Data

The records are the **CITES Trade Database** (v2026.1, 1975–2024), published at
<https://trade.cites.org/>. The snapshot used here was taken on 2026-10-16 and
contains 29,158,995 records for 1975–2024.

No records are redistributed. The code reads a local DuckDB table named
`trade`; see [docs/data.md](docs/data.md) for the expected columns and a
one-line import command.

The analysis keeps exporter-reported, direct-export records in the count unit.
Labels and the candidate pool use wild-source volume (`Source = W`). Features
use the full direct-export panel plus explicit source shares, so the model can
distinguish wild trade from captive-bred trade instead of treating all records
as interchangeable.

## 4. Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# build data/cites.duckdb first (docs/data.md)
CITES_DB=data/cites.duckdb python src/run_all.py
```

A run writes:

| file | content |
|---|---|
| `output/panel.parquet` | exporter × taxon × year panel |
| `output/labels.parquet` | binary surge label |
| `output/feat_T*.parquet` | 60 features as of each year |
| `output/main_table.csv` | yearly AP, precision, recall, hits |
| `output/baselines.csv` | the three post-hoc baselines |
| `output/new_case_table.csv` | results after removing last year's positives |
| `output/overlap.json` | list overlap and carry-over rate |
| `output/permutation.csv` | label-shuffle control |
| `output/shap_features.csv` | mean absolute SHAP by feature |
| `output/diag_*.png` | plain diagnostic panels |

The full chain is CPU-only. A single run trains 27 models (9 years × 3 seeds)
plus the control fits; four threads are enough.

## 5. Method in brief

- **60 features**, all computed as of `T-1`: 15 trade-history, 18
  network-position, 27 trade-form and taxonomy. The full list with definitions
  is in [docs/methods.md](docs/methods.md).
- **Model**: LightGBM binary classifier, 300 rounds, learning rate 0.05, 31
  leaves, minimum 50 per leaf, 0.8 feature and row sampling, class weight equal
  to negatives / positives.
- **Protocol**: rolling origin. For each test year, train on 2005 … `T-1`,
  predict `T`, refit from scratch. No tuning on test years.
- **Metrics**: AP, precision@100, recall@100, hits@100.
- **Checks**: new-case channel, top-100 overlap, label shuffle, SHAP group
  split.

## 6. Figures

| file | panel |
|---|---|
| [fig01_overview.png](figures/fig01_overview.png) | data, decision point and result on one page |
| [fig02_new_case_channel.png](figures/fig02_new_case_channel.png) | hits without last year's positives; list overlap |
| [fig03_what_the_model_uses.png](figures/fig03_what_the_model_uses.png) | SHAP distribution and group shares |
| [fig04_case_series.png](figures/fig04_case_series.png) | three example time series |
| [fig05_year_by_year.png](figures/fig05_year_by_year.png) | yearly AP and hits against the best baseline |
| [fig06_robustness_and_taxa.png](figures/fig06_robustness_and_taxa.png) | eight-year check, thresholds, intervals, taxa |

See [docs/figures.md](docs/figures.md) for what each panel shows and which
numbers to read.

## 7. Notes on scope

- The 2025 annual file is excluded: at the snapshot date it has about 11,000
  records against roughly 1.0–1.35 million in a normal year.
- 2016 is an unusually strong year (55 positives, 26 hits). It is shown
  separately and removed in the eight-year check; the direction does not depend
  on it.
- The ranking answers "who to look at first", not "how much trade is
  sustainable". It is a triage input, not a non-detriment finding.
- The released code is a cleaned version of the analysis pipeline. Internal
  exploratory variants and one-off checks are not included.

## 8. License

MIT License. See [LICENSE](LICENSE).
