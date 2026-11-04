# -*- coding: utf-8 -*-
"""Regenerate the core diagnostic panels from the result tables.

Date   : 2026-11-04

The full figures in ``figures/`` were styled separately; this script produces
clean, plain versions of four checks: yearly hits, yearly AP, list overlap, and
the SHAP group split.

Run:  python src/figures.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

BLUE = "#2166ac"
ORANGE = "#d6604d"
GREY = "#8c8c8c"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.set_axisbelow(True)


def main():
    os.makedirs(C.OUT_DIR, exist_ok=True)
    main = pd.read_csv(os.path.join(C.OUT_DIR, "main_table.csv"))
    base = pd.read_csv(os.path.join(C.OUT_DIR, "baselines.csv"))
    full = main[main["channel"] == "full"].groupby("year").agg(
        hits=("hits@100", "mean"), ap=("ap", "mean")).reset_index()
    base["best_hits"] = base[["persistence_hits", "growth_hits", "volume_hits"]].max(axis=1)
    base["best_ap"] = base[["persistence_ap", "growth_ap", "volume_ap"]].max(axis=1)
    x = np.arange(len(full))
    years = full["year"].astype(int).astype(str).tolist()

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(x - 0.2, full["hits"], width=0.4, color=BLUE, label="model")
    ax.bar(x + 0.2, base["best_hits"], width=0.4, color=ORANGE, label="best post-hoc")
    ax.set_xticks(x, years)
    ax.set_ylabel("true surges in top 100")
    ax.set_xlabel("test year")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "diag_hits.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(x, full["ap"], marker="o", color=BLUE, label="model")
    ax.plot(x, base["best_ap"], marker="s", color=ORANGE, label="best post-hoc")
    ax.set_yscale("log")
    ax.set_xticks(x, years)
    ax.set_ylabel("average precision (log)")
    ax.set_xlabel("test year")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "diag_ap.png"), dpi=300)
    plt.close(fig)

    with open(os.path.join(C.OUT_DIR, "overlap.json"), encoding="utf-8") as f:
        overlap = json.load(f)
    ov = pd.DataFrame([
        {"year": int(y), "overlap": v.get("overlap", np.nan),
         "carry": v.get("carry_pos", np.nan)} for y, v in overlap.items()
    ]).sort_values("year")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(ov["year"].astype(str), ov["overlap"], color=GREY, label="top-100 overlap")
    ax.plot(ov["year"].astype(str), ov["carry"], "o", color=ORANGE,
            label="previous positives still positive")
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("share")
    ax.set_xlabel("test year")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "diag_overlap.png"), dpi=300)
    plt.close(fig)

    shap = pd.read_csv(os.path.join(C.OUT_DIR, "shap_features.csv"))
    shap["group"] = "other"
    for name, cols in C.FEATURE_GROUPS.items():
        shap.loc[shap["feature"].isin(cols), "group"] = name
    grp = shap.groupby("group")["share"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.barh(grp.index, grp.values * 100, color=BLUE)
    ax.set_xlabel("share of mean |SHAP| (%)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "diag_shap_groups.png"), dpi=300)
    plt.close(fig)

    print(f"[figures] diagnostics -> {C.OUT_DIR}")


if __name__ == "__main__":
    main()
