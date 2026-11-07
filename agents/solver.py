"""Perfect-information (double-dummy-like) solver for the play phase.

Multi-player general-sum => max^n: at every node the player to move picks the action that maximises
their own final deal score; ties broken by minimising the sum of the other players' scores
(mildly paranoid, makes the solution unique and robust).
Memoisation at trick boundaries on (hands, won, leader).
Feasible for <= ~4 cards per player in pure Python.
"""
from __future__ import annotations

from engine.core import PH_DONE, legal_actions, scores, step




def solve(s, memo=None) -> tuple:
    """Return the max^n score vector from state s (must be in play phase or done)."""
    if memo is None:
        memo = {}
    return _solve(s, memo)


def _solve(s, memo):
    if s.phase == PH_DONE:
        return tuple(scores(s))
    at_trick_start = not s.t_players
    if at_trick_start:
        key = (tuple(s.hands), tuple(s.won), s.leader)
        v = memo.get(key)
        if v is not None:
            return v
    p = s.cur
    best = None
    bk = None
    for a in legal_actions(s):
        c = s.clone()
        step(c, a)
        v = _solve(c, memo)
        k = (v[p], -(sum(v) - v[p]))
        if bk is None or k > bk:
            bk, best = k, v
    if at_trick_start:
        memo[key] = best
    return best


def best_action(s, memo=None):
    """Return (action, value_vector) of the max^n choice at state s, and all action values."""
    if memo is None:
        memo = {}
    p = s.cur
    vals = {}
    for a in legal_actions(s):
        c = s.clone()
        step(c, a)
        vals[a] = _solve(c, memo)
    a = max(vals, key=lambda a: (vals[a][p], -(sum(vals[a]) - vals[a][p])))
    return a, vals
