"""Action-level statistics of an agent in self-play (tips T12, T14).

python -m analysis.actions --spec store:dmc_v2 --deals 4000
T12: how often the agent's card play coincides with heuristic / heuristic2 in the same position.
T14: first trick, leader needs tricks and holds a side (non-trump) ace -> how often is the ace led?
"""
from __future__ import annotations

import argparse
import random

import numpy as np

from agents.heuristic import heuristic_play
from agents.heuristic2 import smart_play
from engine import NR, Obs, default_rules, legal_actions, new_deal, step
from eval.harness import pmap, _get_agents


def _task(args):
    spec, seed, k, dtype, dealer = args
    R = default_rules()
    agents = _get_agents((spec,) * 4, seed, R)
    for a in agents:
        a.new_deal()
    s = new_deal(R, k, dealer, seed, dtype=dtype)
    out = []
    while not s.done:
        p = s.cur
        legal = legal_actions(s)
        if len(legal) == 1:
            step(s, legal[0])
            continue
        o = Obs(s, p)
        a = agents[p].act(o, legal)
        if s.phase == 1:
            h1 = heuristic_play(o, p, o.hand, legal)
            h2 = smart_play(o, p, o.hand, legal)
            need = s.bids[p] - s.won[p]
            leading = not s.t_players
            side_aces = [c for c in legal if c < 36 and c % NR == 8 and c // NR != s.trump]
            out.append((k, s.tricks_done, int(leading), need, int(a == h1), int(a == h2), len(legal),
                        int(bool(side_aces)), int(a in side_aces), int(a >= 36),
                        (1 if a in (36,) or 38 <= a < 42 else 2) if 36 <= a < 46 else 0, k - s.tricks_done))
        step(s, a)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="store:dmc_v2")
    ap.add_argument("--deals", type=int, default=4000)
    ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args(argv)
    R = default_rules()
    rng = random.Random(7)
    sched = list(R.schedule)
    tasks = [(a.spec, 900_000 + d, *sched[d % len(sched)], rng.randrange(4)) for d in range(a.deals)]
    rows = np.array([r for rs in pmap(_task, tasks, a.procs) for r in rs])
    k, ti, lead, need, eq1, eq2, nl, hasA, playA, isJ, jmode, rem = rows.T
    print(f"### Действия `{a.spec}` в self-play ({a.deals} раздач, {len(rows)} решений розыгрыша с >1 вариантом)\n")
    print("| n_cards | совпадение с heuristic | с heuristic2 | N |")
    print("|---|---|---|---|")
    for kk in sorted(set(k)):
        m = k == kk
        print(f"| {kk} | {eq1[m].mean():.2f} | {eq2[m].mean():.2f} | {m.sum()} |")
    print(f"| все | {eq1.mean():.2f} | {eq2.mean():.2f} | {len(rows)} |")
    m = (ti == 0) & (lead == 1) & (need > 0) & (hasA == 1)
    m2 = (ti == 0) & (lead == 1) & (need <= 0) & (hasA == 1)
    print(f"\nT14: первая взятка, ходящий нуждается во взятках и имеет некозырного туза: заходит с туза в "
          f"{playA[m].mean():.2f} случаев (N={m.sum()}); если взятки НЕ нужны: {playA[m2].mean():.2f} (N={m2.sum()})")
    for kk in sorted(set(k[m])):
        mm = m & (k == kk)
        if mm.sum() >= 30:
            print(f"  - {kk} карт: {playA[mm].mean():.2f} (N={mm.sum()})")
    j = isJ == 1
    print("\n### Когда играется джокер (по моменту хода)\n")
    print("| ситуация | N | старшим | младшим |")
    print("|---|---|---|---|")
    for lab, mm in [("нужны взятки (need>0)", j & (need > 0)), ("заказ уже выполнен (need=0)", j & (need == 0)),
                    ("перебор уже есть (need<0)", j & (need < 0)), ("need = оставшиеся взятки", j & (need == rem)),
                    ("заход джокером", j & (lead == 1)), ("не заход", j & (lead == 0))]:
        if mm.sum():
            print(f"| {lab} | {mm.sum()} | {np.mean(jmode[mm] == 1):.2f} | {np.mean(jmode[mm] == 2):.2f} |")
    # timing: fraction of joker plays at relative position in deal
    mm = j & (k >= 6)
    if mm.sum():
        rel = ti[mm] / np.maximum(1, k[mm] - 1)
        print(f"\nРаздачи ≥6 карт: джокер сыгран в первой трети раздачи {np.mean(rel < 1/3):.2f}, в средней {np.mean((rel >= 1/3) & (rel < 2/3)):.2f}, в последней {np.mean(rel >= 2/3):.2f} (N={mm.sum()})")
    print("\nДоступность хода джокером: доля решений, где джокер был среди вариантов, а агент сыграл не джокер — см. сырые данные.")


if __name__ == "__main__":
    main()


def joker_report(rows):
    pass
