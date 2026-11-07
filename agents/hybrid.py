"""Hybrid agent: one spec for bidding, another for card play (e.g. learned bids + PIMC play).
Register via registry.save_agent(name, ver, 'hybrid', {'bid': spec, 'play': spec})."""
from __future__ import annotations

from engine.core import PH_BID
from .base import Agent


class HybridAgent(Agent):
    name = "hybrid"

    def __init__(self, seed: int = 0, bid: str = "heuristic", play: str = "heuristic", model_dir=None, **kw):
        from registry.store import make_agent
        self.b = make_agent(bid, seed=seed)
        self.p = self.b if play == bid else make_agent(play, seed=seed + 1)

    def reseed(self, seed):
        self.b.reseed(seed)
        self.p.reseed(seed + 1)

    def new_deal(self):
        self.b.new_deal()
        if self.p is not self.b:
            self.p.new_deal()

    def act(self, obs, legal):
        return (self.b if obs.phase == PH_BID else self.p).act(obs, legal)
