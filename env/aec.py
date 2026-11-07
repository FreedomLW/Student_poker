"""PettingZoo-AEC-like wrapper (no pettingzoo dependency).

    env = OdessaAEC(rules, mode="deal"|"game")
    env.reset(seed)
    while not env.done:
        agent = env.agent_selection
        ob = env.observe(agent)        # {'observation': np.float32[...], 'action_mask': np.int8[59], 'obs': Obs}
        env.step(action)
    env.rewards                        # per-player score of last finished deal; env.cumulative for game
"""
from __future__ import annotations

import numpy as np

from engine import (N_ACTIONS, Obs, default_rules, game_deals, legal_actions, new_deal, scores, step)
from .features import encode, obs_dim


class OdessaAEC:
    metadata = {"name": "odessa_poker_v0"}

    def __init__(self, rules=None, mode: str = "deal", n_cards: int | None = None, dtype: str = "normal"):
        self.rules = rules or default_rules()
        self.mode = mode
        self.fixed_n_cards = n_cards
        self.fixed_dtype = dtype
        self.possible_agents = list(range(self.rules.n_players))
        self.observation_size = obs_dim(self.rules.n_players, self.rules.max_cards)
        self.action_size = N_ACTIONS

    def reset(self, seed: int = 0):
        import random
        self._rng = random.Random(seed)
        self.cumulative = [0] * self.rules.n_players
        self.rewards = [0] * self.rules.n_players
        if self.mode == "game":
            self._deals = list(game_deals(self.rules, seed, self._rng.randrange(self.rules.n_players)))
        else:
            k = self.fixed_n_cards or self._rng.randint(1, self.rules.max_cards)
            self._deals = [(k, self.fixed_dtype, self._rng.randrange(self.rules.n_players), self._rng.getrandbits(63))]
        self._di = 0
        self._start_deal()

    def _start_deal(self):
        k, t, dealer, ds = self._deals[self._di]
        self.state = new_deal(self.rules, k, dealer, ds, dtype=t)

    @property
    def agent_selection(self) -> int:
        return self.state.cur

    @property
    def done(self) -> bool:
        return self._di >= len(self._deals)

    def observe(self, agent: int) -> dict:
        o = Obs(self.state, agent)
        mask = np.zeros(N_ACTIONS, dtype=np.int8)
        if agent == self.state.cur:
            mask[legal_actions(self.state)] = 1
        return {"observation": encode(o), "action_mask": mask, "obs": o}

    def step(self, action: int):
        assert action in legal_actions(self.state), f"illegal action {action}"
        step(self.state, action)
        self.rewards = [0] * self.rules.n_players
        if self.state.done:
            self.rewards = scores(self.state)
            for p in range(self.rules.n_players):
                self.cumulative[p] += self.rewards[p]
            self._di += 1
            if not self.done:
                self._start_deal()
