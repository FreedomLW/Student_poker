"""The user's target metric: one agent plays ALL 4 seats for full games; average total points per player
(excluding dark deals, and including them), with 95% CI over games, plus a per-deal-size breakdown.

python -m eval.selfplay_score store:best --games 200
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from engine import load_rules
from .harness import _game_task, bootstrap_ci, pmap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def selfplay_score(spec, games, procs=4, seed0=60_000, overrides=None):
    rules = load_rules(overrides=overrides)
    n = rules.n_players
    tasks = [((spec,) * n, seed0 + g, overrides) for g in range(games)]
    res = pmap(_game_task, tasks, procs)
    sched = rules.schedule
    nodark = [i for i, (k, t) in enumerate(sched) if t != "dark"]
    per_game_nd, per_game_all = [], []
    per_size = {}
    for _, totals, deal_scores in res:
        ds = np.array(deal_scores, float)  # [n_deals, n]
        per_game_nd.append(ds[nodark].sum(0).mean())
        per_game_all.append(ds.sum(0).mean())
        for i, (k, t) in enumerate(sched):
            per_size.setdefault((k, t), []).append(ds[i].mean())
    return np.array(per_game_nd), np.array(per_game_all), {k: float(np.mean(v)) for k, v in per_size.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(argv)
    out = {}
    for sp in a.specs:
        nd, al, ps = selfplay_score(sp, a.games, a.procs)
        m, lo, hi = bootstrap_ci(nd)
        m2, lo2, hi2 = bootstrap_ci(al)
        print(f"{sp}: per-player game total WITHOUT dark {m:.1f} [{lo:.1f}, {hi:.1f}] | with dark {m2:.1f} [{lo2:.1f}, {hi2:.1f}]"
              f" | per deal: " + ", ".join(f"{k}{t[0] if t != 'normal' else ''}:{v:.1f}" for (k, t), v in sorted(ps.items())),
              flush=True)
        out[sp] = dict(nodark=[m, lo, hi], all=[m2, lo2, hi2], per_size={f"{k}_{t}": v for (k, t), v in ps.items()})
    os.makedirs(os.path.join(ROOT, "runs", "eval"), exist_ok=True)
    p = os.path.join(ROOT, "runs", "eval", "selfplay_scores.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(out)
    json.dump(old, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
