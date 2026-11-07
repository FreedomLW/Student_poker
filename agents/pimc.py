"""Perfect-Information Monte Carlo agent.

Decision = average over K determinizations ("worlds") of the opponents' hands, consistent with
  * hand sizes, cards already played, the face-up trump card,
  * public voids (failed to follow suit / to trump),
  * 'demand highest' reveals (after a high joker lead calling X a player showed their highest X),
and importance-resampled by how well each world explains the opponents' bids
(likelihood exp(-(est-bid)^2 / 2σ^2) with the heuristic trick estimate of their reconstructed hand).
Each (world, action) is evaluated by
  * exact max^n solve when <= exact_tricks tricks remain (agents.solver), else
  * a heuristic rollout (all seats play agents.heuristic) to the end of the deal.
Bidding: for each world and each legal bid, the remaining bidders bid heuristically, then a rollout.
Common random numbers: the same worlds are used for all candidate actions.
"""
from __future__ import annotations

import math
import random

from engine.core import (BID_OFFSET, K_JHIGH, NR, PH_BID, PH_DONE, SUIT_MASK, bit_list, legal_actions,
                         scores, step)
from engine.obs import state_from_obs
from .base import Agent
from .heuristic import estimate_tricks, heuristic_bid, heuristic_play
from .solver import solve

FULL = (1 << 36) - 1


def rollout(s, policy_bid=heuristic_bid, policy_play=heuristic_play):
    while s.phase != PH_DONE:
        p = s.cur
        legal = legal_actions(s)
        if len(legal) == 1:
            a = legal[0]
        elif s.phase == PH_BID:
            a = policy_bid(s, p, 0 if s.dark else s.hands[p], legal)
        else:
            a = policy_play(s, p, s.hands[p], legal)
        step(s, a)
    return scores(s)


def played_by(o) -> list:
    """Cards each player has played so far in this deal (from history + current trick)."""
    n = o.n
    out = [0] * n
    for leader, cards, kinds, lead_suit, w in o.history:
        for i, c in enumerate(cards):
            out[(leader + i) % n] |= 1 << c
    for p, c in zip(o.t_players, o.t_cards):
        out[p] |= 1 << c
    return out


def forbidden_masks(o) -> list:
    """Per-player masks of cards they cannot hold (voids + 'demand highest' reveals)."""
    n = o.n
    jm = o.rules.joker_mask
    forb = [0] * n
    for p in range(n):
        for su in range(4):
            if o.voids[p] >> su & 1:
                forb[p] |= SUIT_MASK[su] & ~jm
    if o.rules.joker_lead_demand_highest:
        tricks = [(l, c, k, ls) for (l, c, k, ls, w) in o.history]
        if o.t_players:
            tricks.append((o.t_players[0], o.t_cards, o.t_kinds, o.lead_suit))
        for leader, cards, kinds, ls in tricks:
            if kinds and kinds[0] == K_JHIGH:
                for i in range(1, len(cards)):
                    c = cards[i]
                    if kinds[i] == 0 and c // NR == ls:
                        p = (leader + i) % n
                        above = SUIT_MASK[ls] & ~((1 << (c + 1)) - 1)
                        forb[p] |= above & ~jm
    return forb


class PIMCAgent(Agent):
    name = "pimc"

    def __init__(self, seed: int = 0, samples: int = 24, exact_tricks: int = 3, bid_samples: int | None = None,
                 sigma: float = 0.9, oversample: int = 3, bid_mode: str = "rollout", **kw):
        self.rng = random.Random(seed)
        self.K = samples
        self.KB = bid_samples or samples
        self.exact_tricks = exact_tricks
        self.sigma = sigma
        self.oversample = oversample
        self.bid_mode = bid_mode

    # ------------------------------------------------------------ sampling
    def _sample_raw(self, o, sample_me: bool):
        n = o.n
        me = o.me
        jm = o.rules.joker_mask
        pool = FULL & ~o.played
        if not sample_me:
            pool &= ~o.hand
        if o.trump_card >= 0:
            pool &= ~(1 << o.trump_card)
        forb = forbidden_masks(o)
        need = o.hand_sizes
        players = [p for p in range(n) if p != me or sample_me]
        players.sort(key=lambda p: (pool & ~forb[p]).bit_count() - need[p])
        rng = self.rng
        for _attempt in range(50):
            hands = [0] * n
            if not sample_me:
                hands[me] = o.hand
            rem = pool
            ok = True
            for p in players:
                allowed = bit_list(rem & ~forb[p])
                k = need[p]
                if len(allowed) < k:
                    ok = False
                    break
                pick = rng.sample(allowed, k)
                m = 0
                for c in pick:
                    m |= 1 << c
                hands[p] = m
                rem &= ~m
            if ok:
                return hands
        return None

    def sample_worlds(self, o, K: int, sample_me: bool = False):
        weighted = (not o.dark) and any(b >= 0 for p, b in enumerate(o.bids) if p != o.me)
        M = K * self.oversample if weighted else K
        worlds = []
        for _ in range(M):
            h = self._sample_raw(o, sample_me)
            if h is not None:
                worlds.append(h)
        if not worlds:
            return []
        if not weighted or len(worlds) <= K:
            return worlds[:K]
        pb = played_by(o)
        jm = o.rules.joker_mask
        logw = []
        s2 = 2 * self.sigma ** 2
        for h in worlds:
            lw = 0.0
            for p in range(o.n):
                b = o.bids[p]
                if p == o.me or b < 0:
                    continue
                est = estimate_tricks(h[p] | pb[p], o.n_cards, o.n, o.trump, jm)
                lw -= (est - b) ** 2 / s2
            logw.append(lw)
        mx = max(logw)
        w = [math.exp(x - mx) for x in logw]
        tot = sum(w)
        # systematic resampling
        out = []
        step_ = tot / K
        u = self.rng.random() * step_
        c = w[0]
        i = 0
        for j in range(K):
            t = u + j * step_
            while c < t and i < len(w) - 1:
                i += 1
                c += w[i]
            out.append(worlds[i])
        return out

    # ------------------------------------------------------------ evaluation
    def _value(self, s, me):
        remaining = s.n_cards - s.tricks_done
        if s.phase != PH_BID and remaining <= self.exact_tricks:
            return solve(s)[me]
        return rollout(s)[me]

    def action_values(self, o, legal):
        me = o.me
        bidding = o.phase == PH_BID
        K = self.KB if bidding else self.K
        worlds = self.sample_worlds(o, K, sample_me=(o.hand == 0 and bidding))
        tot = {a: 0.0 for a in legal}
        for h in worlds:
            base = state_from_obs(o, h)
            for a in legal:
                s = base.clone()
                step(s, a)
                tot[a] += self._value(s, me)
        k = max(1, len(worlds))
        return {a: v / k for a, v in tot.items()}

    def act(self, obs, legal):
        if obs.phase == PH_BID and self.bid_mode == "heuristic":
            return heuristic_bid(obs, obs.me, obs.hand, legal)
        vals = self.action_values(obs, legal)
        best = max(vals.values())
        cands = [a for a in legal if vals[a] >= best - 1e-9]
        return cands[0] if len(cands) == 1 else self.rng.choice(cands)
