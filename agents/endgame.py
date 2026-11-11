"""Hybrid: a learned agent everywhere, but when <= `tricks` tricks remain in the deal, decide by sampling
consistent worlds (PIMC) and solving each one exactly (max^n). Fixes noisy small-margin endgame choices of the net."""
from __future__ import annotations

from engine.core import PH_BID
from .base import Agent
from .pimc import PIMCAgent


class EndgameAgent(Agent):
    name = "endgame"

    def __init__(self, seed: int = 0, base: str = "store:best", tricks: int = 3, samples: int = 24, model_dir=None, **kw):
        from registry.store import make_agent
        self.base = make_agent(base.replace("~", ":"), seed=seed)
        self.k = tricks
        self.p = PIMCAgent(seed=seed, samples=samples, exact_tricks=tricks)

    def reseed(self, seed):
        self.base.reseed(seed)
        self.p.reseed(seed)

    def new_deal(self):
        self.base.new_deal()

    def act(self, o, legal):
        if o.phase != PH_BID and o.n_cards - o.tricks_done <= self.k:
            return self.p.act(o, legal)
        return self.base.act(o, legal)
