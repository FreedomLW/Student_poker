"""Evaluation harness: duplicate games, vs-field deltas, pairwise matches, Elo (Bradley–Terry), bootstrap CIs."""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import random

import numpy as np

from engine import load_rules, new_deal, play_deal, play_game

_AGENT_CACHE = {}


def _get_agents(specs, seed, rules):
    from registry.store import make_agent
    out = []
    for i, sp in enumerate(specs):
        key = (sp, i)
        if key not in _AGENT_CACHE:
            _AGENT_CACHE[key] = make_agent(sp, seed=seed * 7919 + i, rules=rules)
        ag = _AGENT_CACHE[key]
        if hasattr(ag, "reseed"):
            ag.reseed(seed * 7919 + i)
        out.append(ag)
    return out


def _game_task(args):
    specs, seed, overrides = args
    rules = load_rules(overrides=overrides)
    agents = _get_agents(specs, seed, rules)
    totals, log = play_game(rules, agents, seed)
    return seed, totals, [d["scores"] for d in log]


def _deal_task(args):
    specs, seed, overrides, n_cards, dtype, dealer = args
    rules = load_rules(overrides=overrides)
    agents = _get_agents(specs, seed, rules)
    for ag in agents:
        ag.new_deal()
    s = new_deal(rules, n_cards, dealer, seed, dtype=dtype)
    sc = play_deal(s, agents)
    return seed, sc, s.bids[:], s.won[:]


def pmap(fn, tasks, procs=None):
    procs = procs or min(4, os.cpu_count() or 1)
    if procs <= 1 or len(tasks) < 4:
        return [fn(t) for t in tasks]
    ctx = mp.get_context("fork")
    with ctx.Pool(procs) as pool:
        return pool.map(fn, tasks, chunksize=max(1, len(tasks) // (procs * 8)))


def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def rotations(n):
    return [list(range(i, n)) + list(range(i)) for i in range(n)]


def vs_field(cand: str, field: str, n_games: int, overrides=None, seed0=10_000, procs=None):
    """Duplicate: candidate in each seat vs 3x field, compared with all-field on identical deals.
    Returns per-game delta (candidate - field in same seat), in points per DEAL."""
    rules = load_rules(overrides=overrides)
    n = rules.n_players
    n_deals = len(rules.schedule)
    tasks = []
    for g in range(n_games):
        seed = seed0 + g
        tasks.append(((field,) * n, seed, overrides))
        for seat in range(n):
            specs = [field] * n
            specs[seat] = cand
            tasks.append((tuple(specs), seed, overrides))
    res = pmap(_game_task, tasks, procs)
    by = {}
    for (specs, seed, _), (_, totals, _) in zip(tasks, res):
        by[(specs, seed)] = totals
    deltas = []
    for g in range(n_games):
        seed = seed0 + g
        base = by[((field,) * n, seed)]
        d = 0.0
        for seat in range(n):
            specs = [field] * n
            specs[seat] = cand
            d += by[(tuple(specs), seed)][seat] - base[seat]
        deltas.append(d / n / n_deals)
    return np.array(deltas)


def pairwise(a: str, b: str, n_games: int, overrides=None, seed0=20_000, procs=None):
    """Duplicate pair match: seatings ABAB.. and BABA.. on the same seeds.
    Returns per-game (mean A total - mean B total) / n_deals and per-game win indicator for A."""
    rules = load_rules(overrides=overrides)
    n = rules.n_players
    n_deals = len(rules.schedule)
    seat_sets = []
    for start in range(2):
        specs = [a if (i + start) % 2 == 0 else b for i in range(n)]
        seat_sets.append(tuple(specs))
    tasks = [(sp, seed0 + g, overrides) for g in range(n_games) for sp in seat_sets]
    res = pmap(_game_task, tasks, procs)
    diffs, wins = [], []
    for g in range(n_games):
        ta, tb = [], []
        for k in range(len(seat_sets)):
            specs, _, _ = tasks[g * len(seat_sets) + k]
            totals = res[g * len(seat_sets) + k][1]
            for i, sp in enumerate(specs):
                (ta if sp == a else tb).append(totals[i])
        d = (np.mean(ta) - np.mean(tb)) / n_deals
        diffs.append(d)
        wins.append(1.0 if d > 0 else (0.5 if d == 0 else 0.0))
    return np.array(diffs), np.array(wins)


def bradley_terry_elo(names, results, anchor=None, anchor_elo=1500.0, iters=500):
    """results: list of (i, j, w) with w = wins of i over j (fractional allowed), per game.
    Returns Elo per name (MM algorithm for BT, with a tiny prior to avoid infinities)."""
    k = len(names)
    W = np.full((k, k), 0.0)
    N = np.zeros((k, k))
    for i, j, w in results:
        W[i, j] += w
        W[j, i] += 1 - w
        N[i, j] += 1
        N[j, i] += 1
    # prior: 1 virtual draw vs each opponent played
    W += 0.5 * (N > 0)
    N += 1.0 * (N > 0)
    p = np.ones(k)
    for _ in range(iters):
        for i in range(k):
            denom = sum(N[i, j] / (p[i] + p[j]) for j in range(k) if j != i and N[i, j] > 0)
            if denom > 0:
                p[i] = W[i].sum() / denom
        p /= np.exp(np.mean(np.log(p)))
    elo = 400 * np.log10(p)
    if anchor is not None:
        elo += anchor_elo - elo[names.index(anchor)]
    else:
        elo += anchor_elo
    return dict(zip(names, elo))


def deal_eval(specs, n_cards, n_deals, dtype="normal", overrides=None, seed0=30_000, procs=None):
    """Play n_deals single deals of fixed size with seat rotation. Returns list of (seed, scores, bids, won)."""
    rules = load_rules(overrides=overrides)
    n = rules.n_players
    tasks = [(tuple(specs), seed0 + d, overrides, n_cards, dtype, d % n) for d in range(n_deals)]
    return pmap(_deal_task, tasks, procs)
