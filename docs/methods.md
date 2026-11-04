# Method notes

This page records the choices that are easy to lose in a short methods section:
the exact label rule, the candidate pool, the 60 features, and the checks that
separate a real ranking gain from a list that merely copies last year.

## 1. Unit and volume

One observation is an **exporter × taxon × year** cell. Volume is the sum of
`Quantity` over exporter-reported, direct-export records in the count unit.
Records with a null unit are kept; records in other units are not converted.

Two volumes are kept side by side:

- `v_wild` — records with `Source = W`. Labels and the candidate pool use this.
- `v_cnt` — all source codes. Features use this, together with the source
  shares (`src_W` … `src_U`) and `v_wild_share`.

Keeping both is deliberate. A wild-only label answers "which wild flow is about
to surge"; a wild-only feature matrix would throw away the fact that the same
exporter also ships captive-bred specimens, which changes the meaning of a
volume jump. The source shares let the model use that context without treating
all records as the same kind of trade.

## 2. Label

For exporter `c`, taxon `u` and label year `T`:

```
M(c,u,T) = (1/5) * sum_{k=1..5} V_wild(c,u,T-k)
```

A cell is positive when either condition holds:

```
country surge: V_wild(c,u,T) > 4 * M(c,u,T)  and  M(c,u,T) >= 100
global surge : V_wild(u,T)   > 4 * M_world(u,T) and
               M_world(u,T) >= 100 and
               V_wild(c,u,T) / V_wild(u,T) >= 0.10
```

The two conditions correspond to the two "sharp increase" criteria used in the
CITES Review of Significant Trade. The thresholds (4×, mean ≥ 100, 10% share)
are fixed before modelling; the figure set repeats the exercise at 3× and 5×.

Zero-trade and unreported years both enter the mean as zero, with a fixed
denominator of five. This is intentionally permissive: requiring five
consecutive years of trade would remove exactly the newly appearing flows the
list is meant to catch. The cost is that a small, sparse history can produce a
large ratio from a modest recovery.

## 3. Candidate pool

For label year `T`, candidates are cells with at least one wild-source record in
`T-3 … T-1`. The five-year window is used only to compute the mean; it does not
require trade in every year. In the 2016–2024 test window this yields
**3,129–4,136 candidates per year**.

## 4. The 60 features

All features are computed as of year `t = T-1` and use only years ≤ `t`.

### 4.1 Trade history (15)

| feature | definition |
|---|---|
| `v_log1p` | log(1 + current-year volume) |
| `v_lag2_log1p` | log(1 + volume two years earlier) |
| `active5` | active years in `t-4 … t` |
| `recency` | years since the last positive year (25 if never) |
| `cv5` | coefficient of variation over `t-4 … t` |
| `slope5` | OLS slope over `t-4 … t`, normalized by the mean |
| `max_over_mean5` | maximum / mean over `t-4 … t` |
| `zero_years5` | 5 − `active5` |
| `log_records` | log(1 + number of records) |
| `qty_per_record` | volume / number of records |
| `last_share5` | current volume / five-year sum |
| `v_allunit_ratio` | all-unit volume / count-unit volume |
| `v_lag3_log1p` | log(1 + volume three years earlier) |
| `v_sum_lag3_log1p` | log(1 + sum over `t-3 … t-1`) |
| `v_std_log` | log(1 + standard deviation over `t-4 … t`) |

### 4.2 Network position (18)

| feature | definition |
|---|---|
| `deg_exp` | number of species the exporter traded |
| `deg_sp` | number of exporters trading the species |
| `share_global` | unit volume / world volume of the species |
| `hhi_sp` | concentration of the species across exporters |
| `peer_vol_log` | log(1 + species volume excluding this unit) |
| `peer_growth` | peer volume / peer five-year mean |
| `exp_total_log` | log(1 + exporter total volume) |
| `exp_entropy` | entropy of the exporter's species mix |
| `exp_growth` | exporter total / five-year mean |
| `jac_max` | maximum Jaccard overlap with peer exporters |
| `jac_mean` | mean Jaccard overlap |
| `aa_peer` | Adamic–Adar peer score on the country projection |
| `genus_countries` | number of exporters trading the genus |
| `genus_vol_share` | unit volume / genus volume |
| `partner_country_n` | countries sharing at least one species |
| `growth_rank_sp` | percentile rank of growth within the species |
| `deg_exp_chg` | change in exporter degree vs the previous year |
| `sp_growth` | species total / five-year mean |

### 4.3 Trade form and taxonomy (27)

| feature | definition |
|---|---|
| `term_entropy` | entropy of the product-form mix |
| `live_share` | live specimens / volume |
| `top_term_share` | largest product-form share |
| `terms_n` | number of distinct terms |
| `importers_n` | number of distinct importers |
| `top_importer_share` | largest importer share |
| `origin_n` | number of distinct origins |
| `src_W` … `src_U` | share of each source code (W, C, F, R, D, A, U) |
| `v_wild_share` | wild volume / total volume |
| `p_T`, `p_P`, `p_H` | share of purpose codes T, P, H |
| `reexport_share` | re-export volume / (volume + re-export) |
| `appendix_code` | I=3, II=2, III=1, N=0, unknown=−1 |
| `class_code`, `order_code`, `family_code`, `genus_code` | frequency codes of the taxonomy |
| `is_spp` | name contains “spp.” |
| `is_hybrid` | name contains “hybrid”, “cultivar” or “x” |
| `taxon_n_tokens` | tokens in the scientific name |

Rule-state features (the lagged rule flag, the label ratio itself, the five-year
mean level) are deliberately excluded. They would let the model read the answer
off the rule instead of learning a transferable pattern.

## 5. Model and protocol

- LightGBM binary classifier, 300 rounds, learning rate 0.05, 31 leaves,
  minimum 50 observations per leaf, feature and row sampling 0.8.
- Class weight = negatives / positives.
- Three seeds (0/1/2); the yearly table is the seed mean.
- For test year `T`, train on 2005 … `T-1`, predict `T`, refit for every year.
- No hyper-parameter tuning on the test years.

## 6. Metrics and baselines

For the top 100 of each yearly list:

- AP — average precision over the whole ranking;
- precision@100 — positives / 100;
- recall@100 — positives found / all positives that year;
- hits@100 — number of positives found.

The post-hoc baselines use only the previous annual summary:

1. **persistence** — copy last year's positive list;
2. **growth** — rank by `V(T-1) / M(T-1)`;
3. **volume** — rank by `V(T-1)`.

The strongest baseline in a year is the best of the three on the metric being
reported. Ties are broken at random and averaged over 20 draws for the
persistence and volume baselines.

## 7. Checks

- **New-case channel** — remove last year's positives from the test pool and
  rank again. Persistence becomes random by construction; a model that only
  copies last year's list cannot pass.
- **List overlap** — share of this year's top 100 that appeared in last year's
  top 100, and share of last year's positives that are still positive.
- **Label shuffle** — retrain on randomly permuted training labels and evaluate
  on the real labels. The control should fall to the positive rate.
- **SHAP** — mean absolute TreeSHAP value per feature, averaged over the nine
  test years, then summed by feature group.

## 8. Descriptive anchors

These are useful when checking a rerun:

| quantity | value |
|---|---|
| test years | 2016–2024 (9) |
| candidates per year | 3,129–4,136 |
| positives over nine years | 295 |
| positives that continue from last year | 32 |
| new cases | 263 (89%) |
| yearly positive rate | 0.35–1.43% |
| top-100 share of candidates | 2.4–3.2% |
| model hits@100 | 5–26 |
| best post-hoc hits@100 | 2–7 |

A full run writes the yearly metrics to `output/main_table.csv`, the baselines
to `output/baselines.csv`, and the control to `output/permutation.csv`.
