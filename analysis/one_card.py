"""Exact analysis of 1-card deals (4 players, default rules) — 'ground truth' for the shortest rounds.

With one card the play is forced except the joker, whose optimal use is dominant:
  bid 1 -> joker HIGH (always wins); bid 0 -> joker LOW (follower: never wins; leader: lead-low calling a
  non-trump, non-spade suit; loses unless nobody else shows that suit or a trump).
So the 1-card deal is a pure Bayesian bidding game. We enumerate ALL deals exactly
(trump card t x ordered 4 cards, 36*35*34*33*32 = 45M) and run fictitious play over the 4 seats'
behavioural bid strategies sigma_i(card, trump card, previous bids). Exploitability (gain from best
response per seat) is reported, so the result is an epsilon-Nash equilibrium with known epsilon.

python -m analysis.one_card [--iters 60]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

from engine import NR, load_rules

R = load_rules(os.environ.get("RULES") or None)
J = R.jokers[0]
FIXED = R.trump_mode == "fixed"
TS = [0] if FIXED else list(range(36))  # 'trump card' index; a single dummy value when the trump suit is fixed
N = 4
C = 36
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def trump_of(t):
    if FIXED:
        return R.trump_fixed_suit
    return -1 if t == J else t // NR


def called_suit(T):
    return min(s for s in (1, 2, 3) if s != T)


def winners_for_t(t):
    """Return valid mask and two winner arrays (joker HIGH / joker LOW) over (c0,c1,c2,c3)."""
    T = trump_of(t)
    c = np.arange(C)
    suit = c // NR
    rank = c % NR
    grids = np.meshgrid(c, c, c, c, indexing="ij")
    valid = np.ones((C,) * 4, bool)
    for a, b in itertools.combinations(range(4), 2):
        valid &= grids[a] != grids[b]
    if not FIXED:
        for g in grids:
            valid &= g != t
    c0 = grids[0]
    lead_suit = np.where(c0 == J, called_suit(T), suit[c0])
    out = []
    for joker_high in (True, False):
        keys = []
        for j, g in enumerate(grids):
            s_, r_ = suit[g], rank[g]
            k = np.where(s_ == T, 101 + r_, np.where(s_ == lead_suit, 11 + r_, 0))
            if joker_high or (j == 0 and not R.joker_lead_low):
                k = np.where(g == J, 1000, k)  # (a leading joker is always HIGH if lead-low is not allowed)
            elif j == 0:
                k = np.where(g == J, 100 if called_suit(T) == T else 10, k)
            elif R.joker_follow_low != "as_card":
                k = np.where(g == J, 0, k)
            # else: follower joker 'as card' keeps the key of the printed card (e.g. 7S)
            keys.append(k)
        out.append(np.argmax(np.stack(keys), axis=0).astype(np.int8))
    return valid, out[0], out[1]


def score(b, win):
    return np.where(win == b, np.where(b == 0, R.pass_exact, R.exact_per_trick * b),
                    np.where(win > b, R.over_per_trick * (win - b), R.under_per_trick * b))


def hist_index(h):
    """prefix tuple -> integer id (length-aware)."""
    return (1 << len(h)) - 1 + int("".join(map(str, h)) or "0", 2)


N_HIST = 1 + 2 + 4 + 8


def dealer_forced(prefix):
    s = sum(prefix)
    if not R.hook_applies(1):
        return None
    if s == 0:
        return 0
    if s == 1:
        return 1
    return None


def seq_list():
    out = []
    for b in itertools.product((0, 1), repeat=4):
        f = dealer_forced(b[:3])
        if f is not None and b[3] != f:
            continue
        out.append(b)
    return out


SEQS = seq_list()


def run(iters=60, verbose=True):
    t0 = time.time()
    sigma = np.full((N, C, len(TS), N_HIST), 0.5)  # sigma[i][c, t, h] = P(bid 1)
    cache = []
    for t in TS:
        valid, wh, wl = winners_for_t(t)
        idx = np.nonzero(valid)
        cards = [x.astype(np.int64) for x in idx]
        jh = np.full(len(cards[0]), -1, np.int64)
        for j in range(4):
            jh = np.where(cards[j] == J, j, jh)
        cache.append((cards, wh[idx], wl[idx], jh))
    n_deals = sum(len(c[0][0]) for c in cache)
    if verbose:
        print(f"winner tables built in {time.time() - t0:.1f}s, {n_deals} deals", flush=True)
    history = []
    for it in range(iters):
        br_ev = np.zeros((N, C, len(TS), N_HIST, 2))
        cur_ev = np.zeros(N)
        for t in range(len(TS)):
            cards, wh, wl, jh = cache[t]
            fac = {}
            for j in range(4):
                for pre in itertools.product((0, 1), repeat=j):
                    if j == 3 and dealer_forced(pre) is not None:
                        f1 = None
                    else:
                        f1 = sigma[j][cards[j], t, hist_index(pre)]
                    fac[(j, pre, 1)] = f1
                    fac[(j, pre, 0)] = None if f1 is None else 1 - f1
            for b in SEQS:
                barr = np.array(b)
                jb = np.where(jh >= 0, barr[np.maximum(jh, 0)], 1)
                w = np.where(jb == 1, wh, wl)
                fs = [fac[(j, b[:j], b[j])] for j in range(4)]
                for i in range(4):
                    wex = np.ones(len(w))
                    for j in range(4):
                        if j != i and fs[j] is not None:
                            wex = wex * fs[j]
                    if b[i] == 0:  # pass: 5 if no trick, overtrick 1 pt if it takes the trick
                        u = np.where(w == i, R.over_per_trick, R.pass_exact)
                    else:
                        u = np.where(w == i, R.exact_per_trick, R.under_per_trick)
                    contrib = wex * u
                    own = fs[i] if fs[i] is not None else 1.0
                    cur_ev[i] += (contrib * own).sum()
                    if fs[i] is None:
                        continue
                    br_ev[i, :, t, hist_index(b[:i]), b[i]] += np.bincount(cards[i], contrib, minlength=C)
        br = (br_ev[..., 1] > br_ev[..., 0]).astype(float)
        tie = np.isclose(br_ev[..., 1], br_ev[..., 0])
        br = np.where(tie, sigma, br)
        gain = []
        for i in range(4):
            cur = (sigma[i] * br_ev[i, ..., 1] + (1 - sigma[i]) * br_ev[i, ..., 0]).sum()
            best = np.maximum(br_ev[i, ..., 1], br_ev[i, ..., 0]).sum()
            gain.append(best - cur)
        gain = np.array(gain) / n_deals
        ev = cur_ev / n_deals
        history.append(dict(it=it, ev=ev.tolist(), exploit=gain.tolist()))
        if verbose:
            print(f"it {it}: EV/seat {np.round(ev, 3)}  BR-gain/seat {np.round(gain, 4)}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        alpha = 1.0 / (it + 2) if it > 0 else 1.0
        sigma = (1 - alpha) * sigma + alpha * br
    return sigma, br_ev, history, cache


def summarize(sigma, br_ev, cache):
    """Human-readable policy summary: P(bid 1) per seat and card class, plus win prob of each class."""
    T_names = {}
    rows = []
    for i in range(4):
        for t in range(C):
            T = trump_of(t)
            for c in range(C):
                if c == t:
                    continue
                if c == J:
                    cls = "joker"
                elif c // NR == T:
                    cls = f"trump {['6','7','8','9','10','J','Q','K','A'][c % NR]}"
                else:
                    cls = f"side {['6','7','8','9','10','J','Q','K','A'][c % NR]}"
                for h in itertools.product((0, 1), repeat=i):
                    rows.append((i, "NT" if T < 0 else "T", cls, sum(h), sigma[i, c, t, hist_index(h)]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    a = ap.parse_args(argv)
    sigma, br_ev, hist, cache = run(a.iters)
    out = os.path.join(ROOT, "runs", "one_card")
    os.makedirs(out, exist_ok=True)
    tag = "_fixed" if FIXED else ""
    np.save(os.path.join(out, f"sigma{tag}.npy"), sigma)
    with open(os.path.join(out, f"history{tag}.json"), "w") as f:
        json.dump(hist, f)


if __name__ == "__main__":
    main()
