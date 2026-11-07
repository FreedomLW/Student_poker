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
                    if kinds[i] == 0 and c // NR == ls and not ((1 << c) & jm):
                        p = (leader + i) % n
                        above = SUIT_MASK[ls] & ~((1 << (c + 1)) - 1)
                        forb[p] |= above & ~jm
    return forb


def history_actions(o):
    """Reconstruct the full action sequence (bids, then plays) of the current deal from public info."""
    from engine.core import A_J_FOLLOW_HIGH, A_J_FOLLOW_LOW, A_J_LEAD_HIGH, A_J_LEAD_LOW, K_JLOW, K_NORMAL
    jm = o.rules.joker_mask
    seq = [(p, BID_OFFSET + b) for p, b in o.bid_history]
    tricks = [(l, c, k, ls) for (l, c, k, ls, w) in o.history]
    if o.t_players:
        tricks.append((o.t_players[0], o.t_cards, o.t_kinds, o.lead_suit))
    for leader, cards, kinds, ls in tricks:
        for i, (c, kd) in enumerate(zip(cards, kinds)):
            p = (leader + i) % o.n
            if not ((1 << c) & jm):
                a = c
            elif i == 0:
                a = (A_J_LEAD_HIGH + ls) if kd == K_JHIGH else (A_J_LEAD_LOW + ls)
            else:
                a = A_J_FOLLOW_HIGH if kd == K_JHIGH else A_J_FOLLOW_LOW
            seq.append((p, a))
    return seq


def policy_loglik(o, worlds, nets, tau=0.5):
    """Rebstock et al. policy-based inference: log P(observed opponent actions | world) under the NN policy
    softmax(Q/tau), for each hypothesised world (current hands). Batched over worlds."""
    import numpy as np
    import torch
    from engine.core import new_deal
    from engine.obs import Obs
    from env.features import encode, obs_dim
    n = o.n
    pb = played_by(o)
    seq = history_actions(o)
    D = obs_dim(n, o.rules.max_cards)
    feats, legals, chosen, owner, phases = [], [], [], [], []
    for wi, h in enumerate(worlds):
        init = [h[p] | pb[p] for p in range(n)]
        init[o.me] = o.hand | pb[o.me]
        st = new_deal(o.rules, o.n_cards, o.dealer, hands=init, trump_card=o.trump_card, dtype=o.dtype)
        for p, a in seq:
            legal = legal_actions(st)
            if p != o.me and len(legal) > 1 and a in legal and not (o.dark and st.phase == PH_BID):
                x = np.zeros(D, np.float32)
                encode(Obs(st, p), x)
                feats.append(x)
                legals.append(legal)
                chosen.append(a)
                owner.append(wi)
                phases.append(0 if st.phase == PH_BID else 1)
            if a not in legal:  # world inconsistent with the observed play
                feats.append(None)
                owner.append(wi)
                legals.append(None)
                chosen.append(None)
                phases.append(-1)
                break
            step(st, a)
    ll = np.zeros(len(worlds))
    idx = [i for i, f in enumerate(feats) if f is not None]
    for i, f in enumerate(feats):
        if f is None:
            ll[owner[i]] = -1e9
    if idx:
        X = torch.from_numpy(np.stack([feats[i] for i in idx]))
        ph = np.array([phases[i] for i in idx])
        Q = np.zeros((len(idx), 59), np.float32)
        with torch.no_grad():
            for k in (0, 1):
                m = ph == k
                if m.any():
                    Q[m] = nets[k](X[torch.from_numpy(m)]).numpy()
        for j, i in enumerate(idx):
            q = np.array([Q[j][a] for a in legals[i]]) / tau
            q -= q.max()
            lp = q[legals[i].index(chosen[i])] - np.log(np.exp(q).sum())
            ll[owner[i]] += lp
    return ll


class PIMCAgent(Agent):
    name = "pimc"

    def __init__(self, seed: int = 0, samples: int = 24, exact_tricks: int = 3, bid_samples: int | None = None,
                 sigma: float = 0.9, oversample: int = 3, bid_mode: str = "rollout", rollout: str = "h1", **kw):
        self.rng = random.Random(seed)
        self.K = samples
        self.KB = bid_samples or samples
        self.exact_tricks = exact_tricks
        self.sigma = sigma
        self.oversample = oversample
        self.bid_mode = bid_mode
        self.infer = kw.get("infer", "bids")
        self.tau = float(kw.get("tau", 0.5))
        self.inf_nets = None
        if self.infer == "policy":
            import torch
            from .learned import load_qnet
            torch.set_num_threads(1)
            self.inf_nets = {0: load_qnet(kw["inf_bid"]), 1: load_qnet(kw.get("inf_play", kw["inf_bid"]))}
        self.net = None
        if rollout.startswith("net"):
            # rollout="net" with net_bid/net_play paths: batched NN rollouts
            import torch
            from .learned import load_qnet
            torch.set_num_threads(1)
            self.net = {0: load_qnet(kw["net_bid"]), 1: load_qnet(kw.get("net_play", kw["net_bid"]))}
            self.exact_tricks = kw.get("net_exact", 2)
        if rollout == "h2c":
            import functools
            import os
            from .heuristic2 import smart_bid, smart_play
            cal = kw.get("calib") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_default.json")
            self.pol = (functools.partial(smart_bid, calib=cal), smart_play)
        elif rollout == "h2" or rollout.startswith("net"):
            from .heuristic2 import smart_bid, smart_play
            self.pol = (smart_bid, smart_play)
        else:
            self.pol = (heuristic_bid, heuristic_play)

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
        if self.infer == "policy":
            weighted = bool(o.bid_history or o.history or o.t_players)
        M = K * self.oversample if weighted else K
        worlds = []
        for _ in range(M):
            h = self._sample_raw(o, sample_me)
            if h is not None:
                worlds.append(h)
        if not worlds:
            return []
        if self.infer == "policy" and len(worlds) > K and (o.bid_history or o.history or o.t_players):
            import numpy as np
            logw = list(policy_loglik(o, worlds, self.inf_nets, self.tau))
            mx = max(logw)
            w = [math.exp(x - mx) for x in logw]
            return self._resample(worlds, w, K)
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
        return self._resample(worlds, w, K)

    def _resample(self, worlds, w, K):
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
        return rollout(s, *self.pol)[me]

    def _net_rollouts(self, states, me):
        """Play all states to the end with the NN policy (batched). Returns my score per state."""
        import numpy as np
        import torch
        from engine.obs import Obs
        from env.features import encode, obs_dim
        out = [None] * len(states)
        live = []
        for i, s in enumerate(states):
            if s.phase == PH_DONE:
                out[i] = scores(s)[me]
            elif s.phase != PH_BID and s.n_cards - s.tricks_done <= self.exact_tricks:
                out[i] = solve(s)[me]
            else:
                live.append(i)
        D = obs_dim(states[0].n, states[0].rules.max_cards)
        buf = np.zeros((len(states), D), np.float32)
        while live:
            nxt = []
            pend = []
            for i in live:
                s = states[i]
                legal = legal_actions(s)
                if len(legal) == 1:
                    step(s, legal[0])
                else:
                    pend.append((i, legal))
            if pend:
                phs = np.array([0 if states[i].phase == PH_BID else 1 for i, _ in pend])
                for j, (i, _) in enumerate(pend):
                    encode(Obs(states[i], states[i].cur), buf[j])
                q = np.zeros((len(pend), 59), np.float32)
                xb = torch.from_numpy(buf[: len(pend)])
                with torch.no_grad():
                    for ph in (0, 1):
                        m = phs == ph
                        if m.any():
                            q[m] = self.net[ph](xb[torch.from_numpy(m)]).numpy()
                for j, (i, legal) in enumerate(pend):
                    step(states[i], max(legal, key=lambda a: q[j][a]))
            for i in live:
                s = states[i]
                if s.phase == PH_DONE:
                    out[i] = scores(s)[me]
                elif s.phase != PH_BID and s.n_cards - s.tricks_done <= self.exact_tricks and not s.t_players:
                    out[i] = solve(s)[me]
                else:
                    nxt.append(i)
            live = nxt
        return out

    def action_values(self, o, legal):
        me = o.me
        if self.net is not None:
            bidding = o.phase == PH_BID
            K = self.KB if bidding else self.K
            worlds = self.sample_worlds(o, K, sample_me=(o.hand == 0 and bidding))
            states, keys = [], []
            for h in worlds:
                base = state_from_obs(o, h)
                for a in legal:
                    s = base.clone()
                    step(s, a)
                    states.append(s)
                    keys.append(a)
            vals = self._net_rollouts(states, me) if states else []
            tot = {a: 0.0 for a in legal}
            for a, v in zip(keys, vals):
                tot[a] += v
            k = max(1, len(worlds))
            return {a: v / k for a, v in tot.items()}
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
            return self.pol[0](obs, obs.me, obs.hand, legal)
        vals = self.action_values(obs, legal)
        best = max(vals.values())
        cands = [a for a in legal if vals[a] >= best - 1e-9]
        return cands[0] if len(cands) == 1 else self.rng.choice(cands)
