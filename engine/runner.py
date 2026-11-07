"""Run deals / full games with a list of agents (one per seat)."""
from __future__ import annotations

import random

from .core import new_deal, legal_actions, step, scores, game_deals
from .obs import Obs


def play_deal(s, agents, record=None):
    """Play state `s` to the end in place. agents[p].act(obs, legal) -> action."""
    while s.phase != 2:
        p = s.cur
        legal = legal_actions(s)
        if len(legal) == 1:
            a = legal[0]
            agents[p].observe_forced(s, p, a) if hasattr(agents[p], "observe_forced") else None
        else:
            a = agents[p].act(Obs(s, p), legal)
        if record is not None:
            record.append((p, a))
        step(s, a)
    return scores(s)


def play_game(rules, agents, seed, first_dealer=None):
    """Full game over rules.schedule. Returns (totals, per-deal list of dicts)."""
    n = rules.n_players
    if first_dealer is None:
        first_dealer = random.Random(seed ^ 0x5EED).randrange(n)
    totals = [0] * n
    log = []
    for k, t, dealer, dseed in game_deals(rules, seed, first_dealer):
        s = new_deal(rules, k, dealer, dseed, dtype=t)
        for ag in agents:
            if hasattr(ag, "new_deal"):
                ag.new_deal()
        sc = play_deal(s, agents)
        for p in range(n):
            totals[p] += sc[p]
        log.append(dict(n_cards=k, dtype=t, dealer=dealer, bids=s.bids[:], won=s.won[:], scores=sc))
    return totals, log
