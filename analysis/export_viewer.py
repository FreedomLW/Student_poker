"""Export full games played by a model (with its action-value estimates at every decision) into a self-contained
HTML viewer.

ODESSA_RULES=rules_variants/house_3p.yaml python -m analysis.export_viewer --spec store:p3_C --games 12 --out viewer/games_3p.html
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (A_J_FOLLOW_HIGH, A_J_FOLLOW_LOW, A_J_LEAD_HIGH, A_J_LEAD_LOW, BID_OFFSET, NR, Obs,  # noqa: E402
                    bit_list, default_rules, game_deals, legal_actions, new_deal, scores, step)
from engine.runner import game_context  # noqa: E402
from registry.store import make_agent  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUIT = "SCDH"
RANK = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]


def card_code(c):
    return RANK[c % NR] + SUIT[c // NR]


def action_label(a, jm_card):
    if a < 36:
        return card_code(a)
    if a == A_J_FOLLOW_HIGH:
        return "JK:high"
    if a == A_J_FOLLOW_LOW:
        return "JK:as7S"
    if A_J_LEAD_HIGH <= a < A_J_LEAD_LOW:
        return "JK:lead:" + SUIT[a - A_J_LEAD_HIGH]
    if A_J_LEAD_LOW <= a < BID_OFFSET:
        return "JK:leadlow:" + SUIT[a - A_J_LEAD_LOW]
    return str(a - BID_OFFSET)


def export(spec, n_games, seed0):
    R = default_rules()
    n = R.n_players
    jm_card = R.jokers[0] if R.jokers else -1
    agents = [make_agent(spec, seed=i) for i in range(n)]
    games = []
    for g in range(n_games):
        seed = seed0 + g
        deals = list(game_deals(R, seed, random.Random(seed).randrange(n)))
        totals = [0] * n
        gdeals = []
        for i, (k, t, dealer, ds) in enumerate(deals):
            s = new_deal(R, k, dealer, ds, dtype=t)
            s.gctx = game_context(totals, i, deals)
            for ag in agents:
                ag.new_deal()
            ev = []
            init = [[card_code(c) for c in bit_list(h)] for h in s.hands]
            totals_before = totals[:]
            while not s.done:
                p = s.cur
                legal = legal_actions(s)
                o = Obs(s, p)
                if len(legal) > 1 and hasattr(agents[p], "q_values"):
                    q = agents[p].q_values(o)
                    a = max(legal, key=lambda z: q[z])
                    opts = sorted(([action_label(z, jm_card), round(float(q[z]) * 10, 1)] for z in legal),
                                  key=lambda x: -x[1])
                elif len(legal) > 1:
                    a = agents[p].act(o, legal)
                    opts = []
                else:
                    a = legal[0]
                    opts = []
                e = dict(p=p, a=action_label(a, jm_card), opts=opts, forced=len(legal) == 1)
                if s.phase == 0:
                    e["type"] = "bid"
                    if R.hook_applies(k) and s.n_bids == n - 1:
                        e["forbidden"] = k - sum(b for b in s.bids if b >= 0)
                else:
                    e["type"] = "play"
                    e["need"] = s.bids[p] - s.won[p]
                n_before = s.tricks_done
                step(s, a)
                if s.phase == 1 and s.tricks_done > n_before or (s.done and s.tricks_done > n_before):
                    ld, cards, kinds, ls, w = s.history[-1]
                    e["trick_winner"] = w
                e["v"] = [[SUIT[x] for x in range(4) if s.voids[q] >> x & 1] for q in range(n)]  # engine's public voids
                ev.append(e)
            sc = scores(s)
            for q_ in range(n):
                totals[q_] += sc[q_]
            gdeals.append(dict(idx=i, n_cards=k, dtype=t, dealer=dealer, first=s.first_seat,
                               trump=SUIT[s.trump] if s.trump >= 0 else None, hands=init, events=ev,
                               bids=s.bids[:], won=s.won[:], scores=sc, totals_before=totals_before,
                               totals_after=totals[:]))
        games.append(dict(id=f"G{seed}", seed=seed, totals=totals, deals=gdeals))
        print(f"game {g + 1}/{n_games}: {totals}", flush=True)
    return dict(spec=spec, n_players=n, rules_hash=R.hash, games=games,
                rules=dict(over_mode=R.over_mode, dark_multiplier=R.dark_multiplier, hook_min_cards=R.hook_min_cards))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="store:p3_C")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=300_000)
    ap.add_argument("--out", default=os.path.join(ROOT, "viewer", "games.html"))
    ap.add_argument("--name", default=None, help="human-readable model name for the page header")
    a = ap.parse_args(argv)
    data = export(a.spec, a.games, a.seed0)
    data["model_name"] = a.name or a.spec
    with open(os.path.join(ROOT, "viewer", "template.html")) as f:
        html = f.read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(html)
    print("saved", a.out, len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
