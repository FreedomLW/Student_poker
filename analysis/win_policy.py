"""How does a WIN-oriented agent differ from a POINTS agent, as a function of the game score?

Plays full games with the win agent in every seat (or vs a field); at every decision of the win agent we also ask
the points agent what it would do in the very same observation. Reports:
  * bid difference (win − points) by deficit to the leader / lead, deals left, normal vs dark;
  * 'spoiling' play: decisions where the win agent, already having its bid (need <= 0), plays a card that wins
    the current trick while the points agent would not — split by whether that trick was needed by an opponent.

python -m analysis.win_policy --win store:p3_W --points store:p3_C --games 200
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.heuristic import card_key, trick_best_key  # noqa: E402
from engine import BID_OFFSET, Obs, default_rules, game_deals, legal_actions, new_deal, scores, step  # noqa: E402
from engine.runner import game_context  # noqa: E402
from eval.harness import pmap  # noqa: E402
from registry.store import make_agent  # noqa: E402

_A = {}


def _task(args):
    win_spec, pts_spec, field_spec, seed = args
    R = default_rules()
    n = R.n_players
    key = (win_spec, pts_spec, field_spec)
    if key not in _A:
        _A[key] = (make_agent(win_spec), make_agent(pts_spec), make_agent(field_spec) if field_spec else None)
    W, P, F = _A[key]
    seat_w = seed % n if F else None
    deals = list(game_deals(R, seed, seed % n))
    totals = [0] * n
    rows_bid, rows_play = [], []
    for i, (k, t, dealer, ds) in enumerate(deals):
        s = new_deal(R, k, dealer, ds, dtype=t)
        s.gctx = game_context(totals, i, deals)
        while not s.done:
            p = s.cur
            legal = legal_actions(s)
            is_w = F is None or p == seat_w
            ag = W if is_w else F
            if len(legal) == 1:
                a = legal[0]
            else:
                o = Obs(s, p)
                a = ag.act(o, legal)
                if is_w:
                    b = P.act(o, legal)
                    others = [totals[(p + r) % n] for r in range(1, n)]
                    lead = totals[p] - max(others)
                    if s.phase == 0:
                        rows_bid.append((k, int(t == "dark"), lead, len(deals) - i, a - BID_OFFSET, b - BID_OFFSET,
                                         int(sum(1 for v in others if v > totals[p]))))
                    else:
                        need = s.bids[p] - s.won[p]
                        if need <= 0 and s.t_players:
                            best = trick_best_key(s)
                            wins = lambda x: x >= 36 and x != 37 or (x < 36 and card_key(x, s.trump, s.lead_suit) > best)
                            opp_need = any(s.bids[q] - s.won[q] > 0 for q in s.t_players)
                            rows_play.append((lead, int(wins(a)), int(wins(b)), int(opp_need), k))
            step(s, a)
        sc = scores(s)
        for q in range(n):
            totals[q] += sc[q]
    return rows_bid, rows_play


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--field", default=None, help="opponents (default: the win agent itself in all seats)")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(argv)
    res = pmap(_task, [(a.win, a.points, a.field, 200_000 + g) for g in range(a.games)], a.procs)
    B = np.array([r for rb, _ in res for r in rb], float)
    Pl = np.array([r for _, rp in res for r in rp], float)
    k, dark, lead, left, bw, bp, nab = B.T
    diff = bw - bp
    print(f"### Заказ: модель «на победу» минус модель «на очки» в тех же ситуациях ({a.games} партий, {len(B)} заказов)\n")
    print("| ситуация | N | средняя разница заказа | доля заказов выше | доля ниже |")
    print("|---|---|---|---|---|")
    bins = [("веду ≥ 50", lead >= 50), ("веду 1–49", (lead > 0) & (lead < 50)), ("отстаю 1–49", (lead < 0) & (lead > -50)),
            ("отстаю 50–99", (lead <= -50) & (lead > -100)), ("отстаю ≥ 100", lead <= -100)]
    for lab, m0 in bins:
        for stage, ms in (("ранняя половина", left > 14), ("последние 14 раздач", (left <= 14) & (left > 3)),
                          ("последние 3 (тёмные)", left <= 3)):
            m = m0 & ms
            if m.sum() >= 30:
                print(f"| {lab}, {stage} | {int(m.sum())} | {diff[m].mean():+.2f} | {np.mean(diff[m] > 0):.2f} | {np.mean(diff[m] < 0):.2f} |")
    m = dark == 1
    if m.any():
        print(f"\nТёмная: заказ модели «на победу» в среднем {bw[m].mean():.2f}, «на очки» {bp[m].mean():.2f}; "
              f"при отставании ≥ 50: {bw[m & (lead <= -50)].mean() if (m & (lead <= -50)).any() else float('nan'):.2f} против "
              f"{bp[m & (lead <= -50)].mean() if (m & (lead <= -50)).any() else float('nan'):.2f}")
        for lab, mm in (("веду", lead > 0), ("отстаю <50", (lead <= 0) & (lead > -50)), ("отстаю ≥50", lead <= -50)):
            mm = mm & m
            if mm.sum() >= 10:
                print(f"  - {lab}: распределение заказов «на победу» "
                      + ", ".join(f"{int(v)}:{np.mean(bw[mm] == v):.2f}" for v in sorted(set(bw[mm]))))
    if len(Pl):
        lead, ww, pw, opp_need, kk = Pl.T
        print(f"\n### «Порча»: свой заказ уже набран, на столе взятка ({len(Pl)} решений)\n")
        print("| ситуация | N | «на победу» забирает взятку | «на очки» забрал бы |")
        print("|---|---|---|---|")
        for lab, m in (("соперник в этой взятке ещё добирает свой заказ", opp_need == 1),
                       ("никому из сыгравших она не нужна", opp_need == 0)):
            for l2, m2 in (("я веду", lead > 0), ("я отстаю", lead <= 0)):
                mm = m & m2
                if mm.sum() >= 30:
                    print(f"| {lab}; {l2} | {int(mm.sum())} | {ww[mm].mean():.2f} | {pw[mm].mean():.2f} |")


if __name__ == "__main__":
    main()
