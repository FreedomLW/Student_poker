"""Teacher labels for targeted fine-tuning (user's idea: fix the n-1 other agents, enumerate the last player's moves).

For a deal (true cards) and a target player p, with all other seats played by a FIXED deterministic agent:
  * label A ("enum", hindsight): for every decision node of p and every legal action a, the best final score p can get
    after a, maximising over ALL of p's own continuations (exhaustive enumeration of p's action sequences);
  * label B ("net", one-step): the final score p gets after a if p then continues with the base net itself
    (one-step policy improvement with common random numbers: all actions are compared on the same deal).
Each node also stores p's observation encoding and legal actions -> all-action regression targets.

python -m train.teacher --deals 20000 --k 3 --out runs/teacher/k3_4p.npz
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import N_ACTIONS, Obs, default_rules, legal_actions, new_deal, scores, step  # noqa: E402
from env.features import encode  # noqa: E402
from eval.harness import pmap  # noqa: E402
from registry.store import make_agent  # noqa: E402

_AG = {}


def _agent(spec):
    if spec not in _AG:
        _AG[spec] = make_agent(spec)
    return _AG[spec]


def _act(ag, s):
    legal = legal_actions(s)
    if len(legal) == 1:
        return legal[0]
    return ag.act(Obs(s, s.cur), legal)


def _advance_others(s, me, ag):
    while not s.done and s.cur != me:
        step(s, _act(ag, s))


def _net_value(s, me, ag):
    """Everybody (incl. me) follows the base agent to the end."""
    s = s.clone()
    while not s.done:
        step(s, _act(ag, s))
    return scores(s)[me]


def _enum(s, me, ag, rows):
    """Return best achievable final score for `me` from state s (me to move or deal over), recording labels."""
    s = s.clone()
    _advance_others(s, me, ag)
    if s.done:
        return scores(s)[me]
    legal = legal_actions(s)
    best_after, net_after = {}, {}
    for a in legal:
        c = s.clone()
        step(c, a)
        best_after[a] = _enum(c, me, ag, rows)
        net_after[a] = _net_value(c, me, ag)
    if len(legal) > 1:
        rows.append((encode(Obs(s, me)).astype(np.float16), 0 if s.phase == 0 else 1, legal,
                     [best_after[a] for a in legal], [net_after[a] for a in legal]))
    return max(best_after.values())


def _task(args):
    spec, seed, k, dealer = args
    R = default_rules()
    ag = _agent(spec)
    s = new_deal(R, k, dealer, seed)
    rows = []
    for me in range(R.n_players):
        _enum(s, me, ag, rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="store:best")
    ap.add_argument("--deals", type=int, default=20000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=7_000_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R = default_rules()
    rng = random.Random(a.seed0)
    tasks = [(a.spec, a.seed0 + d, a.k, rng.randrange(R.n_players)) for d in range(a.deals)]
    rows = [r for rs in pmap(_task, tasks, a.procs) for r in rs]
    X = np.stack([r[0] for r in rows])
    PH = np.array([r[1] for r in rows], np.int8)
    M = np.zeros((len(rows), N_ACTIONS), bool)
    YA = np.zeros((len(rows), N_ACTIONS), np.float32)
    YB = np.zeros((len(rows), N_ACTIONS), np.float32)
    for i, (_, _, legal, ya, yb) in enumerate(rows):
        M[i, legal] = True
        YA[i, legal] = ya
        YB[i, legal] = yb
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, X=X, PH=PH, M=M, YA=YA, YB=YB)
    print("saved", a.out, len(rows), "decision nodes; bid nodes", int((PH == 0).sum()))


if __name__ == "__main__":
    main()
