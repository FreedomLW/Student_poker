"""Agent interface: act(obs, legal) -> action id (see engine.core for ids)."""
from __future__ import annotations


class Agent:
    name = "base"

    def act(self, obs, legal: list) -> int:
        raise NotImplementedError

    def new_deal(self):
        pass

    def reseed(self, seed: int):
        rng = getattr(self, "rng", None)
        if rng is not None:
            rng.seed(seed)
