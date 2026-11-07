"""Build leaderboard.md: duplicate delta vs heuristic field + pairwise Bradley-Terry Elo.

python -m eval.leaderboard --agents random heuristic pimc:samples=16 --games 100 --pair-games 50
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

from .harness import bootstrap_ci, bradley_terry_elo, pairwise, vs_field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", required=True)
    ap.add_argument("--field", default="heuristic")
    ap.add_argument("--games", type=int, default=100, help="games per vs-field eval (x5 runs each)")
    ap.add_argument("--pair-games", type=int, default=50)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "leaderboard.md"))
    a = ap.parse_args(argv)
    t0 = time.time()
    cache_p = os.path.join(ROOT, "runs", "eval", "lb_cache.json")
    os.makedirs(os.path.dirname(cache_p), exist_ok=True)
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}

    def cached(key, fn):
        if key not in cache:
            cache[key] = [np.asarray(x).tolist() for x in fn()]
            json.dump(cache, open(cache_p, "w"))
        return [np.array(x) for x in cache[key]]

    rows = []
    for sp in a.agents:
        if sp == a.field:
            rows.append((sp, 0.0, 0.0, 0.0))
            continue
        d, = cached(f"field|{sp}|{a.field}|{a.games}", lambda: [vs_field(sp, a.field, a.games, procs=a.procs)])
        m, lo, hi = bootstrap_ci(d)
        rows.append((sp, m, lo, hi))
        print(f"vs-field {sp}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] pts/deal", flush=True)
    results = []
    pair_rows = []
    names = list(a.agents)
    if a.pair_games > 0:
        for i, j in itertools.combinations(range(len(names)), 2):
            diffs, wins = cached(f"pair|{names[i]}|{names[j]}|{a.pair_games}",
                                 lambda: pairwise(names[i], names[j], a.pair_games, procs=a.procs))
            for w in wins:
                results.append((i, j, w))
            m, lo, hi = bootstrap_ci(diffs)
            pair_rows.append((names[i], names[j], m, lo, hi, wins.mean()))
            print(f"pair {names[i]} vs {names[j]}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] win {wins.mean():.2f}", flush=True)
        elo = bradley_terry_elo(names, results, anchor=a.field if a.field in names else None)
    else:
        elo = {n: float("nan") for n in names}
    rows.sort(key=lambda r: -r[1])
    lines = ["# Leaderboard", "",
             f"Field = `{a.field}`; duplicate format (same deals, candidate rotated through all 4 seats vs 3×field, "
             f"compared to all-field baseline on identical deals). Δ = points per deal vs field, 95% bootstrap CI over "
             f"{a.games} full games (28 deals each). Elo = Bradley–Terry over pairwise duplicate games "
             f"(ABAB/BABA seatings, {a.pair_games} game-pairs per pair), anchored {a.field}=1500.", "",
             "| agent | Δ vs field (pts/deal) | 95% CI | Elo |", "|---|---|---|---|"]
    for sp, m, lo, hi in rows:
        lines.append(f"| `{sp}` | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {elo.get(sp, float('nan')):.0f} |")
    if pair_rows:
        lines += ["", "## Pairwise (A − B, pts/deal; win-rate of A by game total)", "",
                  "| A | B | A−B | 95% CI | A win-rate |", "|---|---|---|---|---|"]
        for A, B, m, lo, hi, w in pair_rows:
            lines.append(f"| `{A}` | `{B}` | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {w:.2f} |")
    lines += ["", f"_generated in {time.time() - t0:.0f}s_"]
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.join(ROOT, "runs", "eval"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", "eval", f"leaderboard_{int(time.time())}.json"), "w") as f:
        json.dump(dict(rows=rows, pairs=pair_rows, elo=elo), f, indent=1, default=float)


if __name__ == "__main__":
    main()
