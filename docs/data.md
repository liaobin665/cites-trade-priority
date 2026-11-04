# Data

No trade records are stored in this repository. Everything under `output/` is
derived and is also ignored by git.

## Source

The records are the **CITES Trade Database**, published by the CITES
Secretariat and UNEP-WCMC:

- Portal: <https://trade.cites.org/>
- Version used: **v2026.1**, covering 1975–2024
- Snapshot date: 2026-10-16
- 29,158,995 records in 1975–2024

Download the full trade export (CSV) from the portal. The code expects a local
DuckDB file with a table named `trade`. One way to build it:

```bash
duckdb data/cites.duckdb -c \
  "CREATE TABLE trade AS SELECT * FROM read_csv_auto('trade.csv', header=true);"
```

Then run:

```bash
CITES_DB=data/cites.duckdb python src/run_all.py
```

## Expected columns

The pipeline reads the following columns. Names follow the bulk export; if your
export uses different names, rename them before loading.

| column | used as |
|---|---|
| `Year` | trade year |
| `Exporter` | exporting country |
| `Importer` | importing country |
| `Origin` | origin country; direct exports have null or `Origin = Exporter` |
| `Taxon` | scientific name |
| `Class`, `Order`, `Family`, `Genus` | taxonomy codes |
| `Appendix` | CITES appendix (I, II, III, N) |
| `Reporter.type` | `E` exporter report, `I` importer report |
| `Quantity` | reported quantity |
| `Unit` | quantity unit, e.g. `Number of specimens` |
| `Term` | product form (live, meat, fins, …) |
| `Purpose` | trade purpose code (T, P, H, …) |
| `Source` | source code (W wild, C captive-bred, F, R, D, A, U) |

## Fixed caliber

The analysis keeps exporter-reported, direct-export records in the count unit:

```sql
"Reporter.type" = 'E'
AND (Origin IS NULL OR Origin = Exporter)
AND (Unit IS NULL OR Unit = 'Number of specimens')
```

This is a deliberate choice, not a claim that other calibers are wrong:
export-side records match the point at which a permit is issued, while
importer-reported records are better suited to post-hoc reconciliation.

## What is derived, not shipped

| file | content |
|---|---|
| `output/panel.parquet` | exporter × taxon × year aggregates |
| `output/labels.parquet` | binary next-year surge label |
| `output/feat_T*.parquet` | 60 features as of year t |
| `output/*.csv` | yearly metrics, baselines, permutation, SHAP |

The 2025 annual file is excluded. At the snapshot date it contains about
11,000 records, against roughly 1.0–1.35 million in a typical year.
