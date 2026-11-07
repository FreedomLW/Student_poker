"""Agent that plays 1-card deals with the exact (epsilon-Nash) bidding equilibrium from analysis.one_card,
and delegates everything else to a fallback agent spec."""
from __future__ import annotations

import os
import random

import numpy as np

from engine.core import A_J_FOLLOW_HIGH, A_J_FOLLOW_LOW, A_J_LEAD_HIGH, A_J_LEAD_LOW, BID_OFFSET, NR, PH_BID
from .base import Agent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SIGMA = None


def _sigma():
    global _SIGMA
    if _SIGMA is None:
        _SIGMA = np.load(os.path.join(ROOT, "runs", "one_card", "sigma.npy"))
    return _SIGMA


class SmallExactAgent(Agent):
    name = "exact1"

    def __init__(self, seed: int = 0, fallback: str = "heuristic", model_dir=None, **kw):
        from registry.store import make_agent
        self.rng = random.Random(seed)
        self.fb = make_agent(fallback, seed=seed)

    def reseed(self, seed):
        self.rng.seed(seed)
        self.fb.reseed(seed)

    def new_deal(self):
        self.fb.new_deal()

    def act(self, obs, legal):
        if obs.n_cards != 1 or obs.n != 4 or obs.trump_card < 0 or obs.dark:
            return self.fb.act(obs, legal)
        jm = obs.rules.joker_mask
        if obs.phase == PH_BID:
            if len(legal) == 1:
                return legal[0]
            c = obs.hand.bit_length() - 1
            seat = obs.my_seat
            order = [(obs.first_seat + j) % obs.n for j in range(obs.n)]
            h = tuple(obs.bids[q] for q in order[:seat])
            hid = (1 << len(h)) - 1 + int("".join(map(str, h)) or "0", 2)
            p1 = _sigma()[seat, c, obs.trump_card, hid]
            b = 1 if self.rng.random() < p1 else 0
            a = BID_OFFSET + b
            return a if a in legal else legal[0]
        want = obs.bids[obs.me] > obs.won[obs.me]
        if obs.hand & jm:
            if not obs.t_players:
                T = obs.trump
                cs = min(s for s in (1, 2, 3) if s != T)
                a = (A_J_LEAD_HIGH + cs) if want else (A_J_LEAD_LOW + cs)
            else:
                a = A_J_FOLLOW_HIGH if want else A_J_FOLLOW_LOW
            if a in legal:
                return a
        return self.fb.act(obs, legal)
