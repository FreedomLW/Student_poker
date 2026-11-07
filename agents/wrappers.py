"""Policy-modifying wrappers around any base agent (for controlled experiments).

spec: wrap:base=<spec with ':' written as '~'>,mode=<mode>[,delta=..][,bid=..][,coefs=path]
modes:
  shade          bid := base bid + delta (nearest legal)
  forced_bid     bid := `bid` (nearest legal)
  napkin         bid := round(napkin estimate - delta) using coefs json (analysis.calibration output)
  joker_never_low   never play the joker as LOW (when another option exists)
  joker_hold     never play the joker while any other legal card exists
  joker_asap     when tricks are still needed, play the joker HIGH at the first opportunity
"""
from __future__ import annotations

import json

from engine.core import (A_J_FOLLOW_HIGH, A_J_FOLLOW_LOW, A_J_LEAD_HIGH, A_J_LEAD_LOW, BID_OFFSET, PH_BID,
                         is_joker_action)
from .base import Agent


def nearest_legal_bid(legal, b):
    return min(legal, key=lambda a: (abs(a - BID_OFFSET - b), a))


class WrapAgent(Agent):
    name = "wrap"

    def __init__(self, seed: int = 0, base: str = "heuristic", mode: str = "shade", delta: float = 0,
                 bid: int = 0, coefs: str | None = None, model_dir=None, **kw):
        from registry.store import make_agent
        self.base = make_agent(base.replace("~", ":"), seed=seed)
        self.mode = mode
        self.delta = delta
        self.fbid = bid
        self.coefs = None
        self.coefs_path = coefs
        if coefs and coefs.endswith(".json"):
            with open(coefs) as f:
                self.coefs = json.load(f)

    def reseed(self, seed):
        self.base.reseed(seed)

    def new_deal(self):
        self.base.new_deal()

    def tree_bid(self, o):
        import pickle
        import numpy as np
        from analysis.collect import hand_features
        if not hasattr(self, "_tree"):
            with open(self.coefs_path, "rb") as f:
                self._tree = pickle.load(f)
        f = hand_features(o.hand, o.trump, o.rules.joker_mask)
        order = [(o.first_seat + j) % o.n for j in range(o.n)]
        seat = o.my_seat
        prev = sum(o.bids[q] for q in order[:seat])
        f.update(n_cards=o.n_cards, seat=seat, is_trump_deal=int(o.trump >= 0), prev_bid_sum=prev,
                 forbidden_rel=(o.n_cards - prev) if (seat == o.n - 1 and o.rules.dealer_hook) else -1)
        x = np.array([[f[k] for k in self._tree["feats"]]])
        return int(self._tree["clf"].predict(x)[0])

    def napkin_est(self, o):
        from analysis.collect import hand_features
        f = hand_features(o.hand, o.trump, o.rules.joker_mask)
        if o.hand == 0:
            return o.n_cards / o.n
        if o.trump < 0:
            c = self.coefs["nt"]
            return c["const"] + sum(c[k] * f[k] for k in c if k != "const")
        c = self.coefs["trump"].get(str(o.n_cards))
        if c is None:
            c = self.coefs["trump"][max(self.coefs["trump"], key=int)]
        return c["const"] + sum(c[k] * f[k] for k in c if k != "const")

    def act(self, o, legal):
        m = self.mode
        if o.phase == PH_BID:
            if m == "shade":
                return nearest_legal_bid(legal, self.base.act(o, legal) - BID_OFFSET + self.delta)
            if m == "forced_bid":
                return nearest_legal_bid(legal, self.fbid)
            if m == "tree":
                return nearest_legal_bid(legal, self.tree_bid(o))
            if m == "napkin":
                est = self.napkin_est(o) - self.delta
                return nearest_legal_bid(legal, est)
            return self.base.act(o, legal)
        if m == "ace_first":  # T14: lead a side ace on the first trick when tricks are needed
            from engine.core import NR
            if not o.t_players and o.tricks_done == 0 and o.bids[o.me] > o.won[o.me]:
                aces = [a for a in legal if a < 36 and a % NR == 8 and a // NR != o.trump]
                if aces:
                    return aces[0]
            return self.base.act(o, legal)
        if m == "joker_never_low":
            f = [a for a in legal if a != A_J_FOLLOW_LOW and not (A_J_LEAD_LOW <= a < BID_OFFSET)]
            return self.base.act(o, f or legal)
        if m == "joker_hold":
            f = [a for a in legal if not is_joker_action(a)]
            return self.base.act(o, f or legal)
        if m == "joker_asap":
            if o.bids[o.me] > o.won[o.me]:
                if A_J_FOLLOW_HIGH in legal:
                    return A_J_FOLLOW_HIGH
                hi = [a for a in legal if A_J_LEAD_HIGH <= a < A_J_LEAD_LOW]
                if hi:
                    return self.base.act(o, hi)
            return self.base.act(o, legal)
        return self.base.act(o, legal)
