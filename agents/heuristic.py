"""Rule-based agent modelling a 'typical human' player.

The decision functions take any object `x` exposing the public fields shared by State and Obs
(bids, won, t_players, t_cards, t_kinds, lead_suit, demand, trump, n_cards, tricks_done, n,
rules, dark, played) plus the player's own `hand`, so they can run on a full State inside
fast rollouts (agents.pimc) without building an Obs.
"""
from __future__ import annotations

import random

from engine.core import (A_J_FOLLOW_HIGH, A_J_FOLLOW_LOW, A_J_LEAD_HIGH, A_J_LEAD_LOW, BID_OFFSET,
                         K_JHIGH, K_JLOW, NR, SUIT_MASK, bit_list)
from .base import Agent


def trick_best_key(x) -> int:
    """Strength key of the currently winning card in the trick (same scale as engine)."""
    T, L = x.trump, x.lead_suit
    best = -1
    for i, kind in enumerate(x.t_kinds):
        if kind == K_JHIGH:
            k = 1000 + i
        elif kind == K_JLOW:
            k = (100 if L == T else 10) if i == 0 else 0
        else:
            c = x.t_cards[i]
            su = c // NR
            k = 101 + c % NR if su == T else (11 + c % NR if su == L else 0)
        if k > best:
            best = k
    return best


def card_key(c: int, T: int, L: int) -> int:
    su = c // NR
    return 101 + c % NR if su == T else (11 + c % NR if su == L else 0)


def estimate_tricks(hand: int, n_cards: int, n: int, trump: int, jmask: int) -> float:
    """Hand-made trick estimate (what a decent club player would 'count')."""
    if hand == 0:  # dark: unseen hand
        return n_cards / n
    est = 0.0
    if hand & jmask:
        est += 0.95
    plain = hand & ~jmask
    long_factor = min(1.0, n_cards / 9.0)
    for su in range(4):
        cards = plain & SUIT_MASK[su]
        ln = cards.bit_count()
        if not ln:
            continue
        ranks = sorted((c % NR for c in bit_list(cards)), reverse=True)
        if su == trump:
            vals = {8: 0.95, 7: 0.8, 6: 0.6, 5: 0.45, 4: 0.35}
            for r in ranks:
                est += vals.get(r, 0.25)
            if ln >= 4:
                est += 0.3 * (ln - 3)
        elif trump < 0:  # no-trump deal
            for i, r in enumerate(ranks):
                if r == 8:
                    est += 0.85
                elif r == 7 and ln >= 2:
                    est += 0.5
                elif r == 6 and ln >= 3:
                    est += 0.25
            if ln >= 4:
                est += 0.35 * (ln - 3) * long_factor
        else:
            ace_v = 0.85 if ln <= 3 else 0.6
            for r in ranks:
                if r == 8:
                    est += ace_v
                elif r == 7 and 2 <= ln <= 3:
                    est += 0.4
    # small hands: with few cards low cards sometimes win
    if n_cards <= 2 and est < 0.3:
        est += 0.1
    return est


def choose_bid(legal_bids: list, est: float) -> int:
    """Nearest legal bid to the estimate (ties -> lower)."""
    best, bd = None, 1e9
    for a in legal_bids:
        d = abs((a - BID_OFFSET) - est)
        if d < bd - 1e-9:
            best, bd = a, d
    return best


def heuristic_bid(x, me: int, hand: int, legal: list) -> int:
    est = estimate_tricks(hand, x.n_cards, x.n, x.trump, x.rules.joker_mask)
    return choose_bid(legal, est)


def _lowest(cards):
    return min(cards, key=lambda c: c % NR)


def _highest(cards):
    return max(cards, key=lambda c: c % NR)


def heuristic_play(x, me: int, hand: int, legal: list) -> int:
    jm = x.rules.joker_mask
    need = x.bids[me] - x.won[me]
    remaining = x.n_cards - x.tricks_done
    T = x.trump
    plain_legal = [a for a in legal if a < 36]
    has_j = any(a >= 36 for a in legal)
    if not x.t_players:  # ---- leading
        if need > 0:
            if has_j and (need >= remaining or not plain_legal):
                callsuit = T if T >= 0 else 0
                return A_J_LEAD_HIGH + callsuit
            aces = [c for c in plain_legal if c % NR == 8 and c // NR != T]
            if aces:
                return aces[0]
            trumps = [c for c in plain_legal if c // NR == T]
            if trumps:
                return _highest(trumps)
            if plain_legal:
                return _highest(plain_legal)
            return A_J_LEAD_HIGH
        else:
            if plain_legal:
                non_tr = [c for c in plain_legal if c // NR != T] or plain_legal
                return _lowest(non_tr)
            # only joker: lead low in the suit with most unplayed cards outside my hand
            unseen = [(SUIT_MASK[su] & ~x.played & ~hand & ~jm).bit_count() for su in range(4)]
            su = max(range(4), key=lambda q: unseen[q])
            return (A_J_LEAD_LOW + su) if x.rules.joker_lead_low else (A_J_LEAD_HIGH + su)
    # ---- following
    last = len(x.t_players) == x.n - 1
    best = trick_best_key(x)
    L = x.lead_suit
    winners = [c for c in plain_legal if card_key(c, T, L) > best]
    losers = [c for c in plain_legal if card_key(c, T, L) <= best]
    if need > 0:
        if winners:
            if last:
                return min(winners, key=lambda c: card_key(c, T, L))
            # not last: play highest winner if it is a top card, else lowest winner
            return max(winners, key=lambda c: card_key(c, T, L)) if need >= remaining else \
                min(winners, key=lambda c: card_key(c, T, L))
        if has_j and best < 1000 and (need >= remaining or need * 2 >= remaining or not plain_legal):
            return A_J_FOLLOW_HIGH
        if plain_legal:
            return min(plain_legal, key=lambda c: (card_key(c, T, L), c % NR))
        return A_J_FOLLOW_HIGH if best < 1000 else A_J_FOLLOW_LOW
    else:
        if losers:
            return max(losers, key=lambda c: (card_key(c, T, L), c % NR))
        if has_j:
            return A_J_FOLLOW_LOW
        if last:
            return max(winners, key=lambda c: card_key(c, T, L))
        return min(winners, key=lambda c: card_key(c, T, L))


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self, seed: int = 0, bid_noise: float = 0.0):
        self.rng = random.Random(seed)
        self.bid_noise = bid_noise

    def act(self, obs, legal):
        if obs.phase == 0:
            est = estimate_tricks(obs.hand, obs.n_cards, obs.n, obs.trump, obs.rules.joker_mask)
            if self.bid_noise:
                est += self.rng.gauss(0, self.bid_noise)
            return choose_bid(legal, est)
        return heuristic_play(obs, obs.me, obs.hand, legal)
