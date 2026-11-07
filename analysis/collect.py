"""Collect rich per-player-per-deal records from self-play (or mixed) deals for analysis.

python -m analysis.collect --specs store:dmc_v1 --deals 20000 --out runs/data/dmc_v1.npz
Each record: deal-level context, hand features, bid, tricks, score, joker usage.
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np

from engine import NR, SUIT_MASK, bit_list, load_rules, new_deal, play_deal
from eval.harness import _get_agents, pmap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELDS = ["seed", "n_cards", "dtype", "seat", "is_dealer", "trump", "n_trump", "trump_A", "trump_K", "trump_Q",
          "trump_hi", "aces_side", "kings_side", "joker", "max_len", "voids", "singletons", "hcp",
          "bid", "prev_bid_sum", "forbidden", "tricks", "score", "agent",
          "j_trick", "j_lead", "j_high", "j_won", "j_call_trump", "trick_rank_sum"]
DTYPES = {"normal": 0, "notrump": 1, "dark": 2}


def hand_features(hand, trump, jm):
    plain = hand & ~jm
    f = {}
    tm = plain & SUIT_MASK[trump] if trump >= 0 else 0
    f["n_trump"] = tm.bit_count()
    tr = [c % NR for c in bit_list(tm)]
    f["trump_A"] = int(8 in tr)
    f["trump_K"] = int(7 in tr)
    f["trump_Q"] = int(6 in tr)
    f["trump_hi"] = sum(1 for r in tr if r >= 5)  # J and above
    aces = kings = 0
    lens = []
    for su in range(4):
        cs = plain & SUIT_MASK[su]
        lens.append(cs.bit_count())
        if su == trump:
            continue
        rs = [c % NR for c in bit_list(cs)]
        aces += int(8 in rs)
        kings += int(7 in rs)
    f["aces_side"] = aces
    f["kings_side"] = kings
    f["joker"] = int(bool(hand & jm))
    f["max_len"] = max(lens)
    f["voids"] = sum(1 for x in lens if x == 0)
    f["singletons"] = sum(1 for x in lens if x == 1)
    f["hcp"] = sum(max(0, c % NR - 4) for c in bit_list(plain))  # J=1,Q=2,K=3,A=4
    f["trick_rank_sum"] = sum(c % NR for c in bit_list(plain))
    return f


def _task(args):
    specs, seed, overrides, n_cards, dtype, dealer = args
    rules = load_rules(overrides=overrides)
    agents = _get_agents(specs, seed, rules)
    for ag in agents:
        ag.new_deal()
    s = new_deal(rules, n_cards, dealer, seed, dtype=dtype)
    sc = play_deal(s, agents)
    n = s.n
    jm = rules.joker_mask
    # joker usage
    jinfo = {}
    for ti, (leader, cards, kinds, ls, w) in enumerate(s.history):
        for i, c in enumerate(cards):
            if (1 << c) & jm:
                p = (leader + i) % n
                jinfo[p] = (ti, int(i == 0), int(kinds[i] == 1), int(w == p), int(i == 0 and ls == s.trump))
    recs = []
    for p in range(n):
        seat = (p - s.first_seat) % n
        f = hand_features(s.init_hands[p], s.trump, jm)
        order = [(s.first_seat + j) % n for j in range(n)]
        prev = sum(s.bids[q] for q in order[:seat])
        forb = (n_cards - prev) if (seat == n - 1 and rules.dealer_hook) else -1
        ji = jinfo.get(p, (-1, -1, -1, -1, -1))
        rec = [seed, n_cards, DTYPES[dtype], seat, int(seat == n - 1), s.trump] + [f[k] for k in FIELDS[6:18]] + \
              [s.bids[p], prev, forb, s.won[p], sc[p], sorted(set(specs)).index(specs[p])] + \
              list(ji) + [f["trick_rank_sum"]]
        recs.append(rec)
    return recs


def collect(specs, n_deals, overrides=None, sizes=None, seed0=500_000, procs=4):
    rules = load_rules(overrides=overrides)
    n = rules.n_players
    sched = list(rules.schedule) if sizes is None else sizes
    rng = random.Random(seed0)
    tasks = []
    for d in range(n_deals):
        k, t = sched[d % len(sched)]
        rot = d % n
        sp = tuple(specs[rot:]) + tuple(specs[:rot])
        tasks.append((sp, seed0 + d, overrides, k, t, rng.randrange(n)))
    res = pmap(_task, tasks, procs)
    rows = [r for rs in res for r in rs]
    return np.array(rows, dtype=np.int64)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True, help="1 spec (self-play) or n specs")
    ap.add_argument("--deals", type=int, default=10000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=500_000)
    a = ap.parse_args(argv)
    rules = load_rules()
    specs = a.specs * rules.n_players if len(a.specs) == 1 else a.specs
    arr = collect(specs, a.deals, procs=a.procs, seed0=a.seed0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, data=arr, fields=np.array(FIELDS), specs=np.array(specs))
    print("saved", a.out, arr.shape)


def load(path):
    z = np.load(path)
    return {f: z["data"][:, i] for i, f in enumerate(z["fields"])}


if __name__ == "__main__":
    main()
