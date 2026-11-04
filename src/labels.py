# -*- coding: utf-8 -*-
"""Turn the panel into a binary "next-year surge" label.

Date   : 2026-11-04

A unit (exporter, taxon) is positive in year T when either

  country surge : V(T) > 4 * mean(V(T-1) ... V(T-5)) and that mean >= 100
  global surge  : world exports of the taxon in T exceed 4 * their previous
                  five-year mean, and this country holds at least 10% of them

The two conditions mirror the two "sharp increase" criteria used in the CITES
Review of Significant Trade (global trade increase / country increase).  They
are combined with OR.  The rule is fixed before any model is trained.

Run:  python src/labels.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

EPS = 1e-9


def roll_sum(mat, lo, hi):
    """Sum columns ``lo`` .. ``hi-1`` relative to each year column."""
    n, n_years = mat.shape
    pad = np.zeros((n, n_years + 1))
    pad[:, 1:] = np.cumsum(mat.astype(float), axis=1)
    out = np.zeros_like(mat, dtype=float)
    for j in range(n_years):
        a = int(np.clip(j + lo, 0, n_years))
        b = int(np.clip(j + hi, 0, n_years))
        out[:, j] = pad[:, b] - pad[:, a]
    return out


def build(verbose: bool = True):
    panel = pd.read_parquet(C.PANEL)
    years = list(range(C.YEAR_MIN, C.YEAR_MAX + 1))
    units = panel[["exporter", "taxon"]].drop_duplicates().reset_index(drop=True)
    units["uid"] = np.arange(len(units))

    p = units.merge(panel, on=["exporter", "taxon"], how="left")
    p["yi"] = p["year"] - years[0]
    n_units, n_years = len(units), len(years)

    V = np.zeros((n_units, n_years))
    ok = p["v_wild"].notna().values
    V[p.loc[ok, "uid"].values, p.loc[ok, "yi"].values] = p.loc[ok, "v_wild"].values

    # previous five years, excluding the label year
    M5 = roll_sum(V, -5, 0) / 5.0
    active3 = roll_sum((V > 0).astype(float), -3, 0)

    species, species_index = np.unique(units["taxon"].values, return_inverse=True)
    V_sp = np.zeros((len(species), n_years))
    np.add.at(V_sp, species_index, V)
    M5_sp = roll_sum(V_sp, -5, 0) / 5.0

    country = (V > C.SI_MULT * np.maximum(M5, EPS)) & (M5 >= C.SI_MIN_MEAN)
    world = (V_sp > C.SI_MULT * np.maximum(M5_sp, EPS)) & (M5_sp >= C.SI_MIN_MEAN)
    share = V / np.maximum(V_sp[species_index], EPS)
    global_in_country = world[species_index] & (share >= C.GLOBAL_SHARE)

    y = (country | global_in_country).astype(np.int8)

    rows = []
    for j, year in enumerate(years):
        if year < C.TRAIN_START:
            continue
        rows.append(pd.DataFrame({
            "year": year,
            "exporter": units["exporter"].values,
            "taxon": units["taxon"].values,
            "v_wild": V[:, j],
            "m5_wild": M5[:, j],
            "active3": active3[:, j].astype(int),
            "y": y[:, j],
        }))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(C.LABELS, index=False)

    if verbose:
        summary = out.groupby("year").agg(
            units=("y", "size"),
            candidates=("active3", lambda s: int((s > 0).sum())),
            positives=("y", "sum"),
        )
        print(summary.to_string())
        print(f"[labels] saved -> {C.LABELS}")
    return out


if __name__ == "__main__":
    build()
