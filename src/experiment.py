# -*- coding: utf-8 -*-
"""Rolling-origin training and evaluation.

Date   : 2026-11-04

For every test year T (2016-2024) the model is refitted on 2005..T-1 and used
to rank the candidate units of T.  Features come from T-1, so the ranking is
fixed before the T labels exist.

Three post-hoc baselines reproduce what a team can do after the annual summary
arrives: copy last year's list, rank by last year's growth, rank by last year's
volume.  The comparison is on the same 100 slots.

Run:  python src/experiment.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

os.environ.setdefault("OMP_NUM_THREADS", str(C.N_THREADS))


def load_master():
    labels = pd.read_parquet(C.LABELS)
    parts = []
    for year in range(C.TRAIN_START, C.YEAR_MAX + 1):
        feat = pd.read_parquet(os.path.join(C.OUT_DIR, f"feat_T{year - 1}.parquet"))
        lab = labels[labels["year"] == year]
        d = feat.merge(lab, on=["exporter", "taxon"], how="inner", validate="one_to_one")
        if len(d) != len(lab):
            raise RuntimeError(f"merge loss at {year}: {len(d)} vs {len(lab)}")
        parts.append(d)
    master = pd.concat(parts, ignore_index=True)
    master = master[master["active3"] > 0].reset_index(drop=True)

    prev = labels[["exporter", "taxon", "year", "y", "v_wild", "m5_wild"]].copy()
    prev["year"] += 1
    prev = prev.rename(columns={
        "y": "y_prev", "v_wild": "v_prev", "m5_wild": "m5_prev"})
    master = master.merge(prev, on=["exporter", "taxon", "year"], how="left")
    master["y_prev"] = master["y_prev"].fillna(0).astype(int)
    master["v_prev"] = master["v_prev"].fillna(0.0)
    master["m5_prev"] = master["m5_prev"].fillna(0.0)
    return master


def average_precision(y, score):
    order = np.argsort(-score)
    y = np.asarray(y, dtype=float)[order]
    if y.sum() == 0:
        return 0.0
    hits = np.cumsum(y)
    precision = hits / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def top_k(y, score, k=None):
    k = C.K if k is None else k
    order = np.argsort(-score)
    hit = int(np.asarray(y)[order[:k]].sum())
    n_pos = int(np.sum(y))
    return {
        f"hits@{k}": hit,
        f"precision@{k}": hit / k,
        f"recall@{k}": hit / n_pos if n_pos else 0.0,
    }


def lgb_params(n_pos, n_rows, seed):
    return dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        verbose=-1, num_threads=C.N_THREADS, seed=seed,
        bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
        scale_pos_weight=(n_rows - max(n_pos, 1)) / max(n_pos, 1),
    )


def fit_predict(X_train, y_train, X_test, seed):
    import lightgbm as lgb

    model = lgb.train(
        lgb_params(int(y_train.sum()), len(y_train), seed),
        lgb.Dataset(X_train, label=y_train.astype(float)),
        num_boost_round=300,
    )
    return model, model.predict(X_test)


def _baseline(y, base_score, rng, draws=20):
    """Rank by a post-hoc score; average over random tie-breaks."""
    y = np.asarray(y)
    base = np.asarray(base_score, dtype=float)
    n = len(y)
    hits, aps = [], []
    for _ in range(draws):
        order = np.lexsort((rng.random(n), -base))
        hits.append(int(y[order[:C.K]].sum()))
        ranked = np.empty(n)
        ranked[order] = np.arange(n, 0, -1)
        aps.append(average_precision(y, ranked))
    return float(np.mean(hits)), float(np.mean(aps))


def run(verbose: bool = True):
    master = load_master()
    os.makedirs(C.OUT_DIR, exist_ok=True)
    rows, new_rows, base_rows = [], [], []
    top_sets, carry = {}, {}
    perm_rows = []
    shap_parts = []
    rng = np.random.default_rng(0)
    labels = pd.read_parquet(C.LABELS)

    for year in C.TEST_YEARS:
        train = master[(master["year"] >= C.TRAIN_START) & (master["year"] < year)]
        test = master[master["year"] == year].reset_index(drop=True)
        X_train = train[C.FEATURES].values.astype(np.float32)
        X_test = test[C.FEATURES].values.astype(np.float32)
        y_train = train["y"].values.astype(int)
        y_test = test["y"].values.astype(int)

        scores = {}
        for seed in C.SEEDS:
            model, score = fit_predict(X_train, y_train, X_test, seed)
            scores[seed] = score
            m = top_k(y_test, score)
            rows.append({"year": year, "seed": seed, "channel": "full", "n": len(test),
                         "pos": int(y_test.sum()), "ap": average_precision(y_test, score),
                         **m})

        # strongest post-hoc baseline per year (best of the three on each metric)
        growth_score = test["v_prev"] / test["m5_prev"].clip(lower=1e-9)
        p_hit, p_ap = _baseline(y_test, test["y_prev"].values, rng)
        g_hit, g_ap = _baseline(y_test, growth_score.values, rng)
        v_hit, v_ap = _baseline(y_test, test["v_prev"].values, rng)
        base_rows.append({"year": year, "n": len(test), "pos": int(y_test.sum()),
                          "persistence_hits": p_hit, "persistence_ap": p_ap,
                          "growth_hits": g_hit, "growth_ap": g_ap,
                          "volume_hits": v_hit, "volume_ap": v_ap})

        # overlap with last year's model list (seed 0)
        order = np.argsort(-scores[0])[:C.K]
        current = {(test.loc[i, "exporter"], test.loc[i, "taxon"]) for i in order}
        if year - 1 in top_sets:
            inter = len(current & top_sets[year - 1])
            top_sets[year] = current
            carry[year] = {"overlap": inter / C.K}
        else:
            top_sets[year] = current
        prev_pos = labels[(labels["year"] == year - 1) & (labels["y"] == 1)]
        if len(prev_pos):
            stay = prev_pos[["exporter", "taxon"]].merge(
                labels[labels["year"] == year][["exporter", "taxon", "y"]],
                on=["exporter", "taxon"], how="left")
            stay["y"] = stay["y"].fillna(0)
            carry.setdefault(year, {})["carry_pos"] = float(stay["y"].mean())

        # new-case channel: drop units that were positive last year
        new = test[test["y_prev"] == 0]
        if len(new):
            for seed in C.SEEDS:
                score = pd.Series(scores[seed], index=test.index).loc[new.index].values
                m = top_k(new["y"].values, score)
                new_rows.append({"year": year, "seed": seed, "channel": "new",
                                 "n": len(new), "pos": int(new["y"].sum()),
                                 "ap": average_precision(new["y"].values, score), **m})

        # label-shuffle control (three independent shuffles)
        for perm_seed in (11, 23, 37):
            shuffled = np.random.default_rng(perm_seed).permutation(y_train)
            _, score = fit_predict(X_train, shuffled, X_test, 0)
            m = top_k(y_test, score)
            perm_rows.append({"year": year, "perm_seed": perm_seed,
                              "ap": average_precision(y_test, score), **m})

        # SHAP on the seed-0 model
        model, _ = fit_predict(X_train, y_train, X_test, 0)
        contrib = model.predict(X_test, pred_contrib=True)[:, :-1]
        shap_parts.append(pd.DataFrame({
            "year": year,
            "feature": C.FEATURES,
            "mean_abs": np.abs(contrib).mean(axis=0),
        }))
        if verbose:
            best = max(base_rows[-1]["persistence_hits"], base_rows[-1]["growth_hits"],
                       base_rows[-1]["volume_hits"])
            full = [r for r in rows if r["year"] == year]
            print(f"[T={year}] candidates={len(test):4d} positives={int(y_test.sum()):3d} "
                  f"ap={np.mean([r['ap'] for r in full]):.3f} "
                  f"hits={np.mean([r['hits@100'] for r in full]):.1f} "
                  f"| best baseline={best:.1f}")

    main = pd.DataFrame(rows + new_rows)
    main.to_csv(os.path.join(C.OUT_DIR, "main_table.csv"), index=False)
    pd.DataFrame(base_rows).to_csv(os.path.join(C.OUT_DIR, "baselines.csv"), index=False)
    pd.DataFrame(perm_rows).to_csv(os.path.join(C.OUT_DIR, "permutation.csv"), index=False)
    pd.DataFrame(new_rows).to_csv(os.path.join(C.OUT_DIR, "new_case_table.csv"), index=False)
    with open(os.path.join(C.OUT_DIR, "overlap.json"), "w", encoding="utf-8") as f:
        json.dump(carry, f, ensure_ascii=False, indent=2)

    shap = pd.concat(shap_parts, ignore_index=True)
    shap = shap.groupby("feature", as_index=False)["mean_abs"].mean()
    shap["share"] = shap["mean_abs"] / shap["mean_abs"].sum()
    shap.to_csv(os.path.join(C.OUT_DIR, "shap_features.csv"), index=False)
    return main


if __name__ == "__main__":
    run()
