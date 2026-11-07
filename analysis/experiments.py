"""Controlled duplicate experiments for INSIGHTS.md. Results -> runs/analysis/<exp>.json (+ printed markdown).

python -m analysis.experiments <exp> --best store:best [--deals 600]
exps: shade, joker, dark, all
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from eval.harness import bootstrap_ci, deal_vs_field, vs_field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "analysis")
PROCS = int(os.environ.get("PROCS", "1"))


def ci_str(x):
    m, lo, hi = bootstrap_ci(x)
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]", (m, lo, hi)


def tilde(spec):
    return spec.replace(":", "~")


def run_seatwise(cand, best, sizes, deals, dtype="normal", label=""):
    rows = {}
    print(f"\n#### {label}  (`{cand}`)\n")
    print("| n_cards | место 1 | место 2 | место 3 | сдающий | все места |")
    print("|---|---|---|---|---|---|")
    for k in sizes:
        t0 = time.time()
        r = deal_vs_field(cand, best, k, deals, dtype=dtype, procs=PROCS)
        allv = np.concatenate([r[s] for s in range(4)])
        cells = [ci_str(r[s]) for s in range(4)] + [ci_str(allv)]
        rows[k] = [c[1] for c in cells]
        print(f"| {k}{' ' + dtype if dtype != 'normal' else ''} | " + " | ".join(c[0] for c in cells) + f" |", flush=True)
    return rows


def exp_shade(best, deals):
    res = {}
    for d in (-1, 1):
        res[d] = run_seatwise(f"wrap:base={tilde(best)},mode=shade,delta={d}", best, [1, 2, 3, 5, 7, 9], deals,
                              label=f"Сдвиг заказа на {d:+d} относительно лучшего агента (Δ очков/раздачу)")
    return res


def exp_joker(best, deals):
    res = {}
    for m, lab in [("joker_never_low", "Запрет играть джокер младшим"), ("joker_hold", "Держать джокер до конца"),
                   ("joker_asap", "Джокер старшим при первой возможности, если нужны взятки")]:
        res[m] = run_seatwise(f"wrap:base={tilde(best)},mode={m}", best, [3, 6, 9], deals, label=lab)
    return res


def exp_dark(best, deals):
    res = {}
    for b in range(0, 5):
        res[b] = run_seatwise(f"wrap:base={tilde(best)},mode=forced_bid,bid={b}", best, [9], deals, dtype="dark",
                              label=f"Тёмная: принудительный заказ {b} вместо заказа агента")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--best", required=True)
    ap.add_argument("--deals", type=int, default=600)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    exps = {"shade": exp_shade, "joker": exp_joker, "dark": exp_dark}
    todo = list(exps) if a.exp == "all" else [a.exp]
    for e in todo:
        r = exps[e](a.best, a.deals)
        with open(os.path.join(OUT, f"{e}.json"), "w") as f:
            json.dump(r, f, indent=1, default=float)


if __name__ == "__main__":
    main()
