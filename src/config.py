# -*- coding: utf-8 -*-
"""Project-wide settings.

Date   : 2026-11-04

The only input this project needs is a local DuckDB file containing the CITES
trade records in a table named ``trade``.  Point ``CITES_DB`` at that file, or
drop it at ``data/cites.duckdb`` (the default).  No trade records are shipped
with the repository.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output")
FIG_DIR = os.path.join(ROOT, "figures")

CITES_DB = os.environ.get("CITES_DB", os.path.join(DATA_DIR, "cites.duckdb"))
PANEL = os.path.join(OUT_DIR, "panel.parquet")
LABELS = os.path.join(OUT_DIR, "labels.parquet")

# Analysis window.  2025 is deliberately excluded: the annual file is still
# incomplete when the study snapshot is taken.
YEAR_MIN, YEAR_MAX = 2000, 2024
TRAIN_START = 2005
TEST_YEARS = list(range(2016, 2025))

# Label rule (fixed before modelling).
SI_MULT = 4.0          # "more than four times the previous five-year mean"
SI_MIN_MEAN = 100.0    # ignore tiny baselines
GLOBAL_SHARE = 0.10    # a country must hold at least 10% of world exports

# Evaluation.
K = 100
SEEDS = (0, 1, 2)
N_THREADS = 4

SOURCES = ["W", "C", "F", "R", "D", "A", "U"]
LIVE_TERMS = ("live", "eggs (live)")

# The 60 features used by the model.  Rule-state features and the two explicit
# five-year-mean levels are excluded: they either repeat the label rule or are
# replaced by the normalized history features.
VOL = [
    "v_log1p", "v_lag2_log1p", "active5", "recency", "cv5", "slope5",
    "max_over_mean5", "zero_years5", "log_records", "qty_per_record",
    "last_share5", "v_allunit_ratio", "v_lag3_log1p", "v_sum_lag3_log1p",
    "v_std_log",
]
GRAPH = [
    "deg_exp", "deg_sp", "share_global", "hhi_sp", "peer_vol_log",
    "peer_growth", "exp_total_log", "exp_entropy", "exp_growth", "jac_max",
    "jac_mean", "aa_peer", "genus_countries", "genus_vol_share",
    "partner_country_n", "growth_rank_sp", "deg_exp_chg", "sp_growth",
]
FORM = [
    "term_entropy", "live_share", "top_term_share", "terms_n", "importers_n",
    "top_importer_share", "origin_n", "src_W", "src_C", "src_F", "src_R",
    "src_D", "src_A", "src_U", "v_wild_share", "p_T", "p_P", "p_H",
    "reexport_share", "appendix_code", "class_code", "order_code",
    "family_code", "genus_code", "is_spp", "is_hybrid", "taxon_n_tokens",
]
FEATURES = VOL + GRAPH + FORM
assert len(FEATURES) == 60, len(FEATURES)

FEATURE_GROUPS = {
    "trade history": VOL,
    "network position": GRAPH,
    "trade form and taxonomy": FORM,
}
