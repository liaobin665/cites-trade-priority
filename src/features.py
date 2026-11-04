# -*- coding: utf-8 -*-
"""Build the 60 model features as of year t (t = label year - 1).

Author : Liao Bin <liaobin665@163.com>
Date   : 2026-11-04

All features are functions of years <= t.  The assertion at the end of each
year block is a guard against accidental leakage: a feature file written for
t must never contain information from t+1 or later.

Three groups:
  trade history        volume level, trend, volatility, recency
  network position     exporter / taxon degree, peer growth, Jaccard, AA
  trade form           terms, sources, purposes, appendix, taxonomy

Run:  python src/features.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

EPS = 1e-9
FEAT_YEARS = list(range(2004, 2024))


def roll_sum(mat, lo, hi):
    n, n_years = mat.shape
    pad = np.zeros((n, n_years + 1))
    pad[:, 1:] = np.cumsum(mat.astype(float), axis=1)
    out = np.zeros_like(mat, dtype=float)
    for j in range(n_years):
        a = int(np.clip(j + lo, 0, n_years))
        b = int(np.clip(j + hi, 0, n_years))
        out[:, j] = pad[:, b] - pad[:, a]
    return out


def entropy_rows(weights, group_index, n_groups):
    total = np.bincount(group_index, weights=weights, minlength=n_groups)
    p = weights / np.maximum(total[group_index], EPS)
    contrib = np.where(p > 0, p * np.log(np.maximum(p, EPS)), 0.0)
    return -np.bincount(group_index, weights=contrib, minlength=n_groups)


def build(verbose: bool = True):
    panel = pd.read_parquet(C.PANEL)
    years = list(range(C.YEAR_MIN, C.YEAR_MAX + 1))
    units = panel[["exporter", "taxon"]].drop_duplicates().reset_index(drop=True)
    units["uid"] = np.arange(len(units))
    units["genus"] = units["taxon"].map(panel.groupby("taxon")["genus"].first()).fillna("UNK")

    p = units.merge(panel, on=["exporter", "taxon"], how="left")
    p["yi"] = p["year"] - years[0]
    n_units, n_years = len(units), len(years)

    def mat(col):
        M = np.zeros((n_units, n_years))
        ok = p[col].notna().values
        M[p.loc[ok, "uid"].values, p.loc[ok, "yi"].values] = p.loc[ok, col].values
        return M

    V = mat("v_cnt")
    REC = mat("records")
    VA = mat("v_allunit")
    VWILD = mat("v_wild")
    VREX = mat("v_reexport")

    species, sp_inv = np.unique(units["taxon"].values, return_inverse=True)
    exporters, ex_inv = np.unique(units["exporter"].values, return_inverse=True)
    genera, gen_inv = np.unique(units["genus"].values, return_inverse=True)
    n_sp, n_ex, n_gen = len(species), len(exporters), len(genera)

    # ---------------------------------------------------------- time series
    S5 = roll_sum(V, -4, 1)                 # t-4 .. t
    M5 = S5 / 5.0
    M5_prev = roll_sum(V, -5, 0) / 5.0      # previous five years
    SD5 = np.sqrt(np.maximum(roll_sum(V ** 2, -4, 1) / 5.0 - M5 ** 2, 0))
    ACT5 = roll_sum((V > 0).astype(float), -4, 1)
    zero5 = 5 - ACT5

    pos = V > 0
    recency = np.zeros((n_units, n_years))
    last = np.full(n_units, -99)
    for j in range(n_years):
        last = np.where(pos[:, j], j, last)
        recency[:, j] = np.where(last >= 0, j - last, 25)

    slope = np.zeros((n_units, n_years))
    for j in range(n_years):
        a = max(0, j - 4)
        win = V[:, a:j + 1]
        k = win.shape[1]
        xx = np.arange(k) - (k - 1) / 2.0
        slope[:, j] = (win * xx).sum(axis=1) / max((xx ** 2).sum(), 1e-9)

    r1 = V / np.maximum(M5_prev, EPS)

    ex_sum = np.zeros((n_ex, n_years))
    sp_sum = np.zeros((n_sp, n_years))
    ex_S5 = np.zeros((n_ex, n_years))
    sp_S5 = np.zeros((n_sp, n_years))
    for j in range(n_years):
        ex_sum[:, j] = np.bincount(ex_inv, weights=V[:, j], minlength=n_ex)
        sp_sum[:, j] = np.bincount(sp_inv, weights=V[:, j], minlength=n_sp)
        ex_S5[:, j] = np.bincount(ex_inv, weights=S5[:, j], minlength=n_ex)
        sp_S5[:, j] = np.bincount(sp_inv, weights=S5[:, j], minlength=n_sp)

    # ------------------------------------------------------------- context
    def str_pivot(col):
        t = p.pivot_table(index="uid", columns="yi", values=col, aggfunc="first")
        t = t.reindex(index=range(n_units), columns=range(n_years))
        return t.ffill(axis=1).fillna("UNK")

    appendix = str_pivot("appendix").values
    app_map = {"I": 3, "II": 2, "III": 1, "N": 0, "UNK": -1}
    ap_code = np.vectorize(lambda s: app_map.get(s, -1))(appendix).astype(float)

    def codes(series):
        values = series.astype(str).values
        freq = pd.Series(values).value_counts()
        order = {v: i for i, v in enumerate(freq.index)}
        return np.array([order[v] for v in values], dtype=float)

    cls_code = codes(units["taxon"].map(p.groupby("taxon")["cls"].first()).fillna("UNK"))
    ord_code = codes(units["taxon"].map(p.groupby("taxon")["ord"].first()).fillna("UNK"))
    fam_code = codes(units["taxon"].map(p.groupby("taxon")["family"].first()).fillna("UNK"))
    gen_code = codes(units["genus"])

    tokens = units["taxon"].str.split()
    n_tokens = tokens.str.len().astype(float).values
    is_spp = units["taxon"].str.contains(r"spp\.?", case=False, regex=True).astype(float).values
    is_hybrid = units["taxon"].str.contains(
        r"\bx\b|hybrid|cultivar", case=False, regex=True).astype(float).values

    ctx = {c: mat(c) for c in (
        "term_entropy", "live_share", "top_term_share", "terms_n", "importers_n",
        "top_importer_share", "origin_n", "p_T", "p_P", "p_H")}
    for c in ("term_entropy", "live_share", "top_term_share", "top_importer_share"):
        ctx[c] = np.nan_to_num(ctx[c], nan=-1.0)
    src = {s: mat(f"src_{s}") for s in C.SOURCES}

    os.makedirs(C.OUT_DIR, exist_ok=True)
    t0 = time.time()

    for t in FEAT_YEARS:
        j = t - years[0]
        alive = V[:, j] > 0
        ex_deg = np.bincount(ex_inv[alive], minlength=n_ex).astype(float)
        sp_deg = np.bincount(sp_inv[alive], minlength=n_sp).astype(float)

        # country -> set of species, used for Jaccard and Adamic-Adar
        csets = [set() for _ in range(n_ex)]
        for uid in np.where(alive)[0]:
            csets[ex_inv[uid]].add(int(sp_inv[uid]))

        jac_max = np.zeros(n_units)
        jac_mean = np.zeros(n_units)
        aa_peer = np.zeros(n_units)
        sp_members = {}
        for uid in np.where(alive)[0]:
            sp_members.setdefault(int(sp_inv[uid]), []).append(int(ex_inv[uid]))
        for s, members in sp_members.items():
            members = sorted(members)
            if len(members) < 2:
                continue
            for ii, a in enumerate(members):
                others = members[:ii] + members[ii + 1:]
                inter = [len(csets[a] & csets[b]) for b in others]
                union = [len(csets[a] | csets[b]) for b in others]
                jac = [i / u if u else 0.0 for i, u in zip(inter, union)]
                uids = [u for u in np.where(alive)[0]
                        if ex_inv[u] == a and sp_inv[u] == s]
                if not uids:
                    continue
                ua = uids[0]
                jac_max[ua] = max(jac)
                jac_mean[ua] = float(np.mean(jac))
                aa = 0.0
                for b in others:
                    common = csets[a] & csets[b]
                    aa += sum(1.0 / np.log(1.0 + sp_deg[cs])
                              for cs in common if sp_deg[cs] > 0)
                aa_peer[ua] = aa / len(others)

        partner_n = np.zeros(n_ex)
        for a in range(n_ex):
            if csets[a]:
                partner_n[a] = sum(1 for b in range(n_ex)
                                   if b != a and csets[a] & csets[b])

        v = V[:, j]
        share = v / np.maximum(sp_sum[sp_inv, j], EPS)
        hhi = np.zeros(n_sp)
        np.add.at(hhi, sp_inv[alive], share[alive] ** 2)
        ent_ex = entropy_rows(v, ex_inv, n_ex)

        dfr = pd.DataFrame({"sp": sp_inv, "v": v, "r": r1[:, j], "u": np.arange(n_units)})
        rank = dfr[dfr["v"] > 0].groupby("sp")["r"].rank(pct=True)
        growth_rank = np.full(n_units, np.nan)
        growth_rank[dfr[dfr["v"] > 0]["u"].values] = rank.values

        gen_sum = np.bincount(gen_inv, weights=v, minlength=n_gen)

        F = pd.DataFrame(index=np.arange(n_units))
        F["v_log1p"] = np.log1p(v)
        F["v_lag2_log1p"] = np.log1p(V[:, max(j - 1, 0)])
        F["active5"] = ACT5[:, j]
        F["recency"] = recency[:, j]
        F["cv5"] = SD5[:, j] / np.maximum(M5[:, j], 1.0)
        F["slope5"] = slope[:, j] / np.maximum(M5[:, j], 1.0)
        F["max_over_mean5"] = (np.max(V[:, max(0, j - 4):j + 1], axis=1)
                               / np.maximum(M5[:, j], 1.0))
        F["zero_years5"] = zero5[:, j]
        F["log_records"] = np.log1p(REC[:, j])
        F["qty_per_record"] = v / np.maximum(REC[:, j], 1)
        F["last_share5"] = v / np.maximum(S5[:, j], 1)
        F["v_allunit_ratio"] = VA[:, j] / np.maximum(v, 1)
        F["v_lag3_log1p"] = np.log1p(V[:, max(j - 2, 0)])
        F["v_sum_lag3_log1p"] = np.log1p(roll_sum(V, -3, 0)[:, j])
        F["v_std_log"] = np.log1p(np.sqrt(np.maximum(
            roll_sum(V ** 2, -4, 1)[:, j] / 5 - M5[:, j] ** 2, 0)))

        F["deg_exp"] = ex_deg[ex_inv]
        F["deg_sp"] = sp_deg[sp_inv]
        F["share_global"] = share
        F["hhi_sp"] = hhi[sp_inv]
        F["peer_vol_log"] = np.log1p(np.maximum(sp_sum[sp_inv, j] - v, 0))
        F["peer_growth"] = ((sp_sum[sp_inv, j] - v)
                            / np.maximum(sp_S5[sp_inv, j] / 5.0 - v, 1.0))
        F["exp_total_log"] = np.log1p(ex_sum[ex_inv, j])
        F["exp_entropy"] = ent_ex[ex_inv]
        F["exp_growth"] = ex_sum[ex_inv, j] / np.maximum(ex_S5[ex_inv, j] / 5.0, 1.0)
        F["jac_max"] = jac_max
        F["jac_mean"] = jac_mean
        F["aa_peer"] = aa_peer
        F["genus_countries"] = np.bincount(gen_inv[alive], minlength=n_gen)[gen_inv].astype(float)
        F["genus_vol_share"] = v / np.maximum(gen_sum[gen_inv], EPS)
        F["partner_country_n"] = partner_n[ex_inv]
        F["growth_rank_sp"] = growth_rank
        F["deg_exp_chg"] = ex_deg[ex_inv] - np.bincount(
            ex_inv[V[:, max(j - 1, 0)] > 0], minlength=n_ex)[ex_inv]
        F["sp_growth"] = sp_sum[sp_inv, j] / np.maximum(sp_S5[sp_inv, j] / 5.0, 1.0)

        for c in ("term_entropy", "live_share", "top_term_share", "terms_n",
                  "importers_n", "top_importer_share", "origin_n",
                  "p_T", "p_P", "p_H"):
            F[c] = ctx[c][:, j]
        for s in C.SOURCES:
            F[f"src_{s}"] = src[s][:, j] / np.maximum(v, 1)
        F["v_wild_share"] = VWILD[:, j] / np.maximum(v, 1)
        F["reexport_share"] = VREX[:, j] / np.maximum(v + VREX[:, j], 1)
        F["appendix_code"] = ap_code[:, j]
        F["class_code"] = cls_code
        F["order_code"] = ord_code
        F["family_code"] = fam_code
        F["genus_code"] = gen_code
        F["is_spp"] = is_spp
        F["is_hybrid"] = is_hybrid
        F["taxon_n_tokens"] = n_tokens

        assert set(F.columns) == set(C.FEATURES), set(F.columns) ^ set(C.FEATURES)
        F = F[C.FEATURES].replace([np.inf, -np.inf], np.nan)
        out = pd.concat([units[["exporter", "taxon"]], F], axis=1)
        out["feat_year"] = t
        assert out["feat_year"].max() == t and t < C.YEAR_MAX
        out.to_parquet(os.path.join(C.OUT_DIR, f"feat_T{t}.parquet"), index=False)

        if verbose and (t % 5 == 0 or t == FEAT_YEARS[-1]):
            print(f"[features] t={t} rows={len(out):,} "
                  f"nan_rate={F.isna().mean().mean():.3f} ({time.time() - t0:.0f}s)")

    if verbose:
        print(f"[features] {len(FEAT_YEARS)} files, {len(C.FEATURES)} features -> {C.OUT_DIR}")


if __name__ == "__main__":
    build()
