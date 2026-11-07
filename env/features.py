"""Fixed-size numeric encoding of an Obs (for learned agents / external RL libs).

All seat-indexed blocks are RELATIVE to the observing player: rel 0 = me, rel 1 = next to my left, ...
"""
from __future__ import annotations

import numpy as np

from engine.core import K_JHIGH, K_JLOW, NR, SUIT_MASK, bit_list


def obs_dim(n: int, max_cards: int) -> int:
    return (36 * 3                  # hand, played, unseen
            + (n - 1) * 36          # current trick: card by relative player
            + (n - 1) * 2           # current trick: joker high/low by relative player
            + 5 + 1                 # lead suit (4 + none), demand flag
            + 5                     # trump (4 + no-trump)
            + max_cards + 2         # n_cards one-hot, remaining tricks/n_cards, tricks_done/n_cards
            + n * 7                 # per rel player: bid/k, won/k, need/k, need>0, need==0, need<0, has_bid
            + n * 2                 # seat (rel to first bidder) one-hot, dealer one-hot (rel)
            + (n - 1) * 5           # voids per other player (4 suits + trump)
            + 4                     # dark, bidding phase, sum_bids/k, forbidden_bid/k (or -1)
            + 4                     # my hand suit lengths /k
            + 2)                    # joker in my hand, joker already played


def encode(o, out: np.ndarray | None = None) -> np.ndarray:
    n = o.n
    k = o.n_cards
    mc = o.rules.max_cards
    d = obs_dim(n, mc)
    x = np.zeros(d, dtype=np.float32) if out is None else out
    if out is not None:
        x[:] = 0
    me = o.me
    jm = o.rules.joker_mask
    i = 0
    for c in bit_list(o.hand):
        x[i + c] = 1
    i += 36
    for c in bit_list(o.played):
        x[i + c] = 1
    i += 36
    unseen = ((1 << 36) - 1) & ~o.hand & ~o.played
    for c in bit_list(unseen):
        x[i + c] = 1
    i += 36
    tb = i
    jb = i + (n - 1) * 36
    for p, c, kind in zip(o.t_players, o.t_cards, o.t_kinds):
        rel = (p - me) % n
        if rel == 0:
            continue
        x[tb + (rel - 1) * 36 + c] = 1
        if kind == K_JHIGH:
            x[jb + (rel - 1) * 2] = 1
        elif kind == K_JLOW:
            x[jb + (rel - 1) * 2 + 1] = 1
    i = jb + (n - 1) * 2
    x[i + (o.lead_suit if o.lead_suit >= 0 else 4)] = 1
    i += 5
    x[i] = float(o.demand)
    i += 1
    x[i + (o.trump if o.trump >= 0 else 4)] = 1
    i += 5
    x[i + k - 1] = 1
    i += mc
    x[i] = (k - o.tricks_done) / k
    x[i + 1] = o.tricks_done / k
    i += 2
    for rel in range(n):
        p = (me + rel) % n
        b = o.bids[p]
        base = i + rel * 7
        if b >= 0:
            need = b - o.won[p]
            x[base] = b / k
            x[base + 2] = need / k
            x[base + 3] = need > 0
            x[base + 4] = need == 0
            x[base + 5] = need < 0
            x[base + 6] = 1
        x[base + 1] = o.won[p] / k
    i += n * 7
    x[i + (me - o.first_seat) % n] = 1
    i += n
    x[i + (o.dealer - me) % n] = 1
    i += n
    for rel in range(1, n):
        p = (me + rel) % n
        v = o.voids[p]
        base = i + (rel - 1) * 5
        for su in range(4):
            if v >> su & 1:
                x[base + su] = 1
        if o.trump >= 0 and v >> o.trump & 1:
            x[base + 4] = 1
    i += (n - 1) * 5
    x[i] = float(o.dark)
    x[i + 1] = float(o.phase == 0)
    sb = sum(b for b in o.bids if b >= 0)
    x[i + 2] = sb / k
    nb = sum(1 for b in o.bids if b >= 0)
    x[i + 3] = (k - sb) / k if (nb == n - 1 and o.rules.dealer_hook and 0 <= k - sb <= k) else -1
    i += 4
    plain = o.hand & ~jm
    for su in range(4):
        x[i + su] = (plain & SUIT_MASK[su]).bit_count() / k
    i += 4
    x[i] = float(bool(o.hand & jm))
    x[i + 1] = float(bool(o.played & jm))
    i += 2
    assert i == d, (i, d)
    return x
