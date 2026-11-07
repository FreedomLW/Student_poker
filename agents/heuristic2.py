"""'Smart' rule-based agent (a strong club player), stronger than agents.heuristic:

Bidding
  * trick estimate from per-card values (trumps by rank & length, side aces/kings by suit length,
    no-trump long suits), adjusted for what earlier bidders already claimed;
  * the bid maximises EXPECTED SCORE under a discretised normal distribution of tricks around the estimate
    (so it automatically shades for the -10*bid vs +1 asymmetry) and respects the dealer hook;
  * 1-card deals: the exact equilibrium thresholds (analysis.one_card) if available;
  * dark: fixed blind bid table.
Play (card memory: knows all played cards, so it knows 'master' cards = highest unseen in a suit)
  * needs tricks: cash masters, win cheaply when last, overtake only when it holds, joker HIGH when the
    remaining need equals the remaining tricks or when it is the only way to win a needed trick late;
  * wants no more tricks: duck with the highest card that still loses, lead the card most likely to be
    beaten, joker LOW as an emergency discard, never wastes the joker high.
"""
from __future__ import annotations

import math
import random

from engine.core import (A_J_FOLLOW_HIGH, A_J_FOLLOW_LOW, A_J_LEAD_HIGH, A_J_LEAD_LOW, BID_OFFSET, NR, PH_BID,
                         SUIT_MASK, bit_list)
from .base import Agent
from .heuristic import card_key, trick_best_key

FULL = (1 << 36) - 1


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def estimate2(hand, k, n, trump, jm):
    if hand == 0:
        return k / n
    est = 0.0
    plain = hand & ~jm
    if hand & jm:
        est += 1.0
    n_tr = (plain & SUIT_MASK[trump]).bit_count() if trump >= 0 else 0
    for su in range(4):
        cs = plain & SUIT_MASK[su]
        ln = cs.bit_count()
        if not ln:
            continue
        rs = sorted((c % NR for c in bit_list(cs)), reverse=True)
        if su == trump:
            for i, r in enumerate(rs):
                # top trumps are near-certain; lower trumps win by ruffing or length
                v = {8: 0.95, 7: 0.75, 6: 0.55, 5: 0.45}.get(r, 0.33)
                if i >= 2:
                    v = max(v, 0.45)
                est += v
        elif trump < 0:
            for i, r in enumerate(rs):
                if r == 8:
                    est += 0.9
                elif r == 7 and ln >= 2:
                    est += 0.55
                elif r == 6 and ln >= 3:
                    est += 0.3
            if ln >= 4:
                est += 0.45 * (ln - 3)
        else:
            # side suits get ruffed more in longer deals / when others are short
            ruff = min(1.0, k / 9.0)
            if rs[0] == 8:
                est += 0.85 - 0.25 * ruff * (ln >= 3) - 0.15 * ruff
            if len(rs) >= 2 and rs[0] == 8 and rs[1] == 7:
                est += 0.45 - 0.2 * ruff
            elif rs[0] == 7 and ln >= 2:
                est += 0.35 - 0.15 * ruff
    # voids/singletons are ruffing chances if we have spare trumps
    if trump >= 0 and n_tr >= 2:
        shorts = sum(1 for su in range(4) if su != trump and (plain & SUIT_MASK[su]).bit_count() <= 1)
        est += 0.25 * min(shorts, n_tr - 1)
    return est


def ev_best_bid(legal, est, k, rules, dark=False, sd=None):
    if sd is None:
        sd = 0.45 + 0.17 * k ** 0.5
    probs = []
    for t in range(k + 1):
        lo = _norm_cdf((t - 0.5 - est) / sd) if t > 0 else 0.0
        hi = _norm_cdf((t + 0.5 - est) / sd) if t < k else 1.0
        probs.append(max(0.0, hi - lo))
    best, bv = None, -1e9
    for a in legal:
        b = a - BID_OFFSET
        v = sum(p * rules.score(b, t, dark) for t, p in enumerate(probs))
        if v > bv:
            bv, best = v, a
    return best


_SIG = None


def _one_card_sigma():
    global _SIG
    if _SIG is None:
        try:
            import os
            import numpy as np
            _SIG = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "runs", "one_card", "sigma.npy"))
        except Exception:
            _SIG = False
    return _SIG


DARK_BID = {0: 2, 1: 2, 2: 2, 3: 2}  # refined after experiments (see INSIGHTS.md)


_CALIB = {}


def calib_estimate(path, hand, k, trump, jm):
    """OLS estimate of tricks (+ residual sd) from analysis.calibration json."""
    import json
    from analysis.collect import hand_features
    if path not in _CALIB:
        with open(path) as f:
            _CALIB[path] = json.load(f)
    c = _CALIB[path]
    f = hand_features(hand, trump, jm)
    if trump < 0:
        cc = c["nt"]
    else:
        cc = c["trump"].get(str(k)) or c["trump"][max(c["trump"], key=int)]
    est = cc["const"] + sum(v * f[key] for key, v in cc.items() if not key.startswith("_") and key != "const")
    return est, cc.get("_rmse")


def smart_bid(x, me, hand, legal, calib=None):
    r = x.rules
    k = x.n_cards
    n = x.n
    seat = (me - x.first_seat) % n
    order = [(x.first_seat + j) % n for j in range(n)]
    prev = [x.bids[q] for q in order[:seat]]
    if k == 1 and n == 4 and x.trump_card >= 0 and not x.dark and hand:
        sig = _one_card_sigma()
        if sig is not False and sig is not None:
            c = hand.bit_length() - 1
            hid = (1 << len(prev)) - 1 + int("".join(map(str, prev)) or "0", 2)
            b = 1 if sig[seat, c, x.trump_card, hid] > 0.5 else 0
            a = BID_OFFSET + b
            return a if a in legal else legal[0]
    if x.dark:
        b = DARK_BID.get(seat, 2) if k >= 6 else round(k / n)
        return min(legal, key=lambda a: abs(a - BID_OFFSET - b))
    sd = None
    if calib:
        est, sd = calib_estimate(calib, hand, k, x.trump, r.joker_mask)
        if prev:  # calibrated slope of tricks on earlier bidders' excess (analysis.tables)
            est -= 0.2 * (sum(prev) - len(prev) * k / n)
        est = max(0.0, min(k, est))
        return ev_best_bid(legal, est, k, r, sd=sd)
    est = estimate2(hand, k, n, x.trump, r.joker_mask)
    # competition: earlier bidders claiming more than their share lowers my expectation
    if prev:
        excess = sum(prev) - len(prev) * k / n
        est -= 0.2 * excess
    est = max(0.0, min(k, est))
    return ev_best_bid(legal, est, k, r)


def _unseen(x, hand):
    u = FULL & ~hand & ~x.played
    if x.trump_card >= 0:
        u &= ~(1 << x.trump_card)
    return u


def _is_master(c, unseen, jm):
    su = c // NR
    higher = SUIT_MASK[su] & ~((1 << (c + 1)) - 1) & unseen & ~jm
    return higher == 0


def smart_play(x, me, hand, legal):
    jm = x.rules.joker_mask
    need = x.bids[me] - x.won[me]
    remaining = x.n_cards - x.tricks_done
    T = x.trump
    unseen = _unseen(x, hand)
    joker_out = bool(unseen & jm)  # an opponent may still hold the joker
    plain = [a for a in legal if a < 36]
    has_j = any(a >= 36 for a in legal)
    rank = lambda c: c % NR
    trumps_out = (unseen & SUIT_MASK[T] & ~jm).bit_count() if T >= 0 else 0

    if not x.t_players:  # ---------------- leading
        if need > 0:
            if has_j and (need >= remaining or not plain):
                return A_J_LEAD_HIGH + (T if T >= 0 else max(range(4), key=lambda s: (SUIT_MASK[s] & unseen).bit_count()))
            masters = [c for c in plain if _is_master(c, unseen, jm)]
            # prefer trump masters (cannot be ruffed), then side masters while trumps remain out
            tm = [c for c in masters if c // NR == T]
            sm = [c for c in masters if c // NR != T]
            if tm:
                return max(tm, key=rank)
            if sm and trumps_out == 0:
                return max(sm, key=rank)
            if sm:
                # side master: safest in the suit where opponents still have most cards
                return max(sm, key=lambda c: (SUIT_MASK[c // NR] & unseen).bit_count())
            if plain:
                # no master: lead low from longest side suit (develop length), keep high cards
                side = [c for c in plain if c // NR != T] or plain
                return min(side, key=lambda c: (-(hand & SUIT_MASK[c // NR]).bit_count(), rank(c)))
            return A_J_LEAD_HIGH + (T if T >= 0 else 0)
        # avoid tricks: lead the card most likely to be beaten (most unseen higher cards in suit)
        if plain:
            def beaten(c):
                su = c // NR
                hi = (SUIT_MASK[su] & ~((1 << (c + 1)) - 1) & unseen & ~jm).bit_count()
                return (hi, su != T, -rank(c))
            return max(plain, key=beaten)
        su = max(range(4), key=lambda s: ((SUIT_MASK[s] & unseen & ~jm).bit_count(), s != T))
        return (A_J_LEAD_LOW + su) if x.rules.joker_lead_low else (A_J_LEAD_HIGH + su)

    # ---------------- following
    last = len(x.t_players) == x.n - 1
    best = trick_best_key(x)
    L = x.lead_suit
    key = lambda c: card_key(c, T, L)
    winners = [c for c in plain if key(c) > best]
    losers = [c for c in plain if key(c) <= best]
    if need > 0:
        if winners:
            if last:
                return min(winners, key=key)
            # would my best winner hold against the players still to act?
            top = max(winners, key=key)
            safe = [c for c in winners if _is_master(c, unseen, jm) and (c // NR == T or trumps_out == 0
                                                                        or c // NR == L)]
            if safe:
                return min(safe, key=key)
            if need >= remaining:
                return top
            return min(winners, key=key) if not has_j else top
        if has_j and best < 1000 and (need >= remaining or remaining <= 2):
            return A_J_FOLLOW_HIGH
        if plain:  # cannot win: discard the least useful card
            return min(plain, key=lambda c: (key(c) > 0 and c // NR == T, _is_master(c, unseen, jm), rank(c)))
        return A_J_FOLLOW_HIGH if best < 1000 else A_J_FOLLOW_LOW
    # need <= 0 : avoid tricks
    j_low_safe = has_j and (x.rules.joker_follow_low != "as_card" or
                            key((jm & -jm).bit_length() - 1) <= best)
    if losers:
        # dump the highest card that still loses (gets rid of future winners)
        return max(losers, key=lambda c: (key(c), rank(c)))
    if j_low_safe or (has_j and not winners):
        return A_J_FOLLOW_LOW
    if last:
        return max(winners, key=key)
    return min(winners, key=key)


class SmartHeuristicAgent(Agent):
    name = "heuristic2"

    def __init__(self, seed: int = 0, calib: str | None = None, **kw):
        self.rng = random.Random(seed)
        import os
        if calib and not os.path.isabs(calib):
            calib = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), calib)
        self.calib = calib

    def act(self, obs, legal):
        if obs.phase == PH_BID:
            return smart_bid(obs, obs.me, obs.hand, legal, self.calib)
        return smart_play(obs, obs.me, obs.hand, legal)
