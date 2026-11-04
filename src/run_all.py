# -*- coding: utf-8 -*-
"""Run the full analysis chain.

Author : Liao Bin <liaobin665@163.com>
Date   : 2026-11-04

Usage:
    CITES_DB=/path/to/cites.duckdb python src/run_all.py

The database is never copied or modified; the script only reads it.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import experiment
import features
import figures
import labels
import panel


def main():
    print(f"[input] {C.CITES_DB}")
    panel.build()
    labels.build()
    features.build()
    experiment.run()
    figures.main()
    print("[done] output/ contains the tables; figures/ contains the full set.")


if __name__ == "__main__":
    main()
