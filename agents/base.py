"""Agent interface: act(obs, legal) -> action id (see engine.core for ids)."""
from __future__ import annotations


class Agent:
    name = "base"

    def act(self, obs, legal: list) -> int:
        raise NotImplementedError

    def new_deal(self):
        pass
