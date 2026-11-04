# -*- coding: utf-8 -*-
"""Build the exporter x taxon x year panel.

Date   : 2026-11-04

Caliber (fixed before any modelling):
  * exporter-reported records (Reporter.type = 'E')
  * direct exports only (Origin is null or Origin = Exporter)
  * quantity in the count unit ('Number of specimens'); null units kept
  * one row per exporter, taxon and year

The panel keeps both the wild-source volume (``v_wild``) and the all-source
volume (``v_cnt``).  Labels and the candidate pool use ``v_wild``; features use
``v_cnt`` plus explicit source shares, so the model can tell wild trade from
captive-bred trade instead of treating every record as interchangeable.

Run:  python src/panel.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

PRIMARY = (
    "Year BETWEEN {y0} AND {y1} AND \"Reporter.type\"='E' "
    "AND (Origin IS NULL OR Origin=Exporter) "
    "AND (Unit IS NULL OR Unit='Number of specimens')"
).format(y0=C.YEAR_MIN, y1=C.YEAR_MAX)


def _connect():
    import duckdb

    con = duckdb.connect(C.CITES_DB, read_only=True)
    con.execute(f"SET threads={C.N_THREADS}")
    con.execute("PRAGMA memory_limit='6GB'")
    return con


def _base(con):
    src_sums = ",\n".join(
        f"sum(CASE WHEN Source='{s}' THEN Quantity ELSE 0 END) AS src_{s}"
        for s in C.SOURCES
    )
    live_in = ",".join(f"'{t}'" for t in C.LIVE_TERMS)
    sql = f"""
    SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
           sum(Quantity) AS v_cnt,
           sum(CASE WHEN Source='W' THEN Quantity ELSE 0 END) AS v_wild,
           sum(CASE WHEN Source='W' AND Term IN ({live_in}) THEN Quantity ELSE 0 END) AS v_wlive,
           count(*) AS records,
           count(DISTINCT Importer) AS importers_n,
           count(DISTINCT Term) AS terms_n,
           count(DISTINCT Origin) AS origin_n,
           any_value(Appendix) AS appendix,
           any_value(Class) AS cls,
           any_value("Order") AS ord,
           any_value(Family) AS family,
           any_value(Genus) AS genus,
           {src_sums}
    FROM trade
    WHERE {PRIMARY}
    GROUP BY 1, 2, 3
    """
    return con.execute(sql).df()


def _all_units(con):
    return con.execute(f"""
    SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
           sum(Quantity) AS v_allunit
    FROM trade
    WHERE Year BETWEEN {C.YEAR_MIN} AND {C.YEAR_MAX}
      AND "Reporter.type"='E'
      AND (Origin IS NULL OR Origin=Exporter)
    GROUP BY 1, 2, 3
    """).df()


def _terms(con):
    return con.execute(f"""
    SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
           coalesce(Term,'UNK') AS term, sum(Quantity) AS v
    FROM trade
    WHERE {PRIMARY}
    GROUP BY 1, 2, 3, 4
    """).df()


def _importers(con):
    return con.execute(f"""
    SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
           coalesce(Importer,'UNK') AS importer, sum(Quantity) AS v
    FROM trade
    WHERE {PRIMARY}
    GROUP BY 1, 2, 3, 4
    """).df()


def _purposes(con):
    return con.execute(f"""
    SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
           coalesce(Purpose,'UNK') AS purpose, sum(Quantity) AS v
    FROM trade
    WHERE {PRIMARY}
    GROUP BY 1, 2, 3, 4
    """).df()


def _reexports(con):
    return con.execute(f"""
    SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
           sum(Quantity) AS v_reexport
    FROM trade
    WHERE Year BETWEEN {C.YEAR_MIN} AND {C.YEAR_MAX}
      AND "Reporter.type"='E'
      AND Origin IS NOT NULL AND Origin <> Exporter
      AND (Unit IS NULL OR Unit='Number of specimens')
    GROUP BY 1, 2, 3
    """).df()


def build(verbose: bool = True):
    if not os.path.exists(C.CITES_DB):
        raise FileNotFoundError(
            f"CITES database not found: {C.CITES_DB}\n"
            "Download the trade records from https://trade.cites.org/ and see "
            "docs/data.md for the expected table."
        )
    os.makedirs(C.OUT_DIR, exist_ok=True)
    t0 = time.time()
    con = _connect()

    if verbose:
        n = con.execute("SELECT count(*) FROM trade").fetchone()[0]
        print(f"[db] trade rows = {n:,}")

    panel = _base(con)
    allunit = _all_units(con)
    term = _terms(con)
    importer = _importers(con)
    purpose = _purposes(con)
    reexport = _reexports(con)
    con.close()

    panel = panel.merge(allunit, on=["year", "exporter", "taxon"], how="left")

    # term composition
    tot = term.groupby(["year", "exporter", "taxon"])["v"].transform("sum")
    term["share"] = term["v"] / tot.replace(0, np.nan)
    top = term.loc[term.groupby(["year", "exporter", "taxon"])["v"].idxmax(),
                   ["year", "exporter", "taxon", "share"]]
    top = top.rename(columns={"share": "top_term_share"})
    ent = (term.assign(p=lambda d: d["share"] * np.log(d["share"].clip(lower=1e-12)))
              .groupby(["year", "exporter", "taxon"])["p"].sum().rename("term_entropy")
              .reset_index())
    ent["term_entropy"] = -ent["term_entropy"]
    live = (term[term["term"].isin(C.LIVE_TERMS)]
            .groupby(["year", "exporter", "taxon"])["v"].sum().rename("v_live").reset_index())
    panel = (panel.merge(top, on=["year", "exporter", "taxon"], how="left")
                  .merge(ent, on=["year", "exporter", "taxon"], how="left")
                  .merge(live, on=["year", "exporter", "taxon"], how="left"))
    panel["v_live"] = panel["v_live"].fillna(0.0)
    panel["live_share"] = panel["v_live"] / panel["v_cnt"].replace(0, np.nan)

    # purpose shares used in the form group
    ptot = purpose.groupby(["year", "exporter", "taxon"])["v"].transform("sum")
    purpose["share"] = purpose["v"] / ptot.replace(0, np.nan)
    for code in ("T", "P", "H"):
        s = (purpose[purpose["purpose"] == code]
             .groupby(["year", "exporter", "taxon"])["share"].sum()
             .rename(f"p_{code}").reset_index())
        panel = panel.merge(s, on=["year", "exporter", "taxon"], how="left")
        panel[f"p_{code}"] = panel[f"p_{code}"].fillna(0.0)

    itot = importer.groupby(["year", "exporter", "taxon"])["v"].transform("sum")
    importer["share"] = importer["v"] / itot.replace(0, np.nan)
    topimp = (importer.groupby(["year", "exporter", "taxon"])["share"].max()
              .rename("top_importer_share").reset_index())
    panel = panel.merge(topimp, on=["year", "exporter", "taxon"], how="left")
    panel = panel.merge(reexport, on=["year", "exporter", "taxon"], how="left")
    panel["v_reexport"] = panel["v_reexport"].fillna(0.0)

    for col in ("genus", "cls", "ord", "family", "appendix"):
        panel[col] = panel[col].fillna("UNK")

    panel = panel.sort_values(["exporter", "taxon", "year"]).reset_index(drop=True)
    assert panel.duplicated(["year", "exporter", "taxon"]).sum() == 0
    assert panel["v_cnt"].notna().all()
    assert (panel["v_cnt"] >= 0).all()
    panel.to_parquet(C.PANEL, index=False)

    if verbose:
        print(f"[panel] rows={len(panel):,}  exporters={panel.exporter.nunique():,}  "
              f"taxa={panel.taxon.nunique():,}")
        print(f"[panel] years={panel.year.min()}..{panel.year.max()}  "
              f"wild volume={panel.v_wild.sum():,.0f}")
        print(f"[panel] saved -> {C.PANEL}  ({time.time() - t0:.1f}s)")
    return panel


if __name__ == "__main__":
    build()
