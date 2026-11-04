# Figure inventory

The six PNG files in `figures/` are the full versions. They are static
outputs; `src/figures.py` regenerates plain diagnostic versions from
`output/*.csv`.

| file | what it shows | reading |
|---|---|---|
| `fig01_overview.png` | 2022 exporter–species network; yearly candidate, slot and surge counts; the two label channels; yearly model-vs-baseline comparison; coverage of the top 100 | one page that carries the problem, the data and the result |
| `fig02_new_case_channel.png` | hits after removing last year's positives (a); overlap with last year's top 100 and the share of last year's positives that remain positive (b) | the model is not just copying the previous list |
| `fig03_what_the_model_uses.png` | SHAP distribution of the top six features (a); mean absolute SHAP by feature group (b) | recent trade history dominates; raw last-year volume is a minor feature |
| `fig04_case_series.png` | annual wild-source volume for three example units, with the decision point and the surge year marked | how a ranking looks on real records |
| `fig05_year_by_year.png` | yearly AP (a) and hits@100 (b), model vs the strongest post-hoc baseline | the direction is consistent across all nine test years |
| `fig06_robustness_and_taxa.png` | eight-year check without 2016 (a, b); threshold sensitivity (c); bootstrap intervals for 2023/2024 (d); taxon composition and capture rate (e, f) | where the result is stable and where it is thin |

## Notes on the numbers

- The top 100 is 2.4–3.2% of the yearly candidate pool but covers 21–48% of
  that year's surges.
- The model finds 5–26 surges per year; the strongest post-hoc baseline finds
  2–7.
- 89% of the nine-year positives were not on last year's positive list.
- 2016 is a high year (55 positives, 26 hits). It is shown separately and
  excluded in the eight-year check; the direction does not depend on it.
- The 2023 and 2024 bootstrap intervals are wider because those years have only
  11 and 17 positives.

## Style

Figures use a colour-blind-safe palette, vector-equivalent 600 dpi raster
output, and 7–9 pt type at printed width. The source series behind each panel
are in `output/` after a full run.
