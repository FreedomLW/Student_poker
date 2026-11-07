"""Trick-distribution bidding model (the 'regression' design from the task spec).

1. Self-play with a fixed play policy (default store:dmc_v3); at each bid decision the bid is the agent's own with
   prob 1-eps, uniform legal otherwise (exploration gives counterfactual coverage of bids).
2. Train an MLP classifier P(tricks = t | bid-time observation, my bid b)  (t = 0..12).
3. BidDistAgent: bid = argmax_b sum_t score(b, t) P(t | x, b) over LEGAL bids (dealer hook respected via the mask);
   card play delegated to the play policy.

python -m train.bid_dist --deals 60000 --epochs 6
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from engine import BID_OFFSET, Obs, default_rules, legal_actions, new_deal, step
from env.features import encode, obs_dim
from eval.harness import _get_agents, pmap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAXB = 13


class DistNet(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + MAXB, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, MAXB))

    def forward(self, x, b_onehot):
        return self.net(torch.cat([x, b_onehot], 1))


def _task(args):
    spec, seed, k, dtype, dealer, eps = args
    R = default_rules()
    agents = _get_agents((spec,) * 4, seed, R)
    for a in agents:
        a.new_deal()
    rng = random.Random(seed)
    s = new_deal(R, k, dealer, seed, dtype=dtype)
    rec = []
    while not s.done:
        p = s.cur
        legal = legal_actions(s)
        o = Obs(s, p)
        if s.phase == 0:
            x = encode(o).astype(np.float16)
            a = rng.choice(legal) if rng.random() < eps else (agents[p].act(o, legal) if len(legal) > 1 else legal[0])
            rec.append((p, x, a - BID_OFFSET))
        else:
            a = agents[p].act(o, legal) if len(legal) > 1 else legal[0]
        step(s, a)
    return [(x, b, s.won[p]) for p, x, b in rec]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="store:dmc_v3")
    ap.add_argument("--deals", type=int, default=60000)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "bid_dist"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    R = default_rules()
    rng = random.Random(3)
    sched = list(R.schedule)
    data_p = os.path.join(a.out, "data.npz")
    if os.path.exists(data_p):
        z = np.load(data_p)
        X, B, T = z["X"], z["B"], z["T"]
    else:
        tasks = [(a.spec, 2_000_000 + d, *sched[d % len(sched)], rng.randrange(4), a.eps) for d in range(a.deals)]
        rows = [r for rs in pmap(_task, tasks, a.procs) for r in rs]
        X = np.stack([r[0] for r in rows])
        B = np.array([r[1] for r in rows], np.int64)
        T = np.array([r[2] for r in rows], np.int64)
        np.savez_compressed(data_p, X=X, B=B, T=T)
    print("samples", len(T))
    torch.set_num_threads(4)
    idx = np.random.default_rng(0).permutation(len(T))
    te, tr = idx[: len(T) // 10], idx[len(T) // 10:]
    D = X.shape[1]
    net = DistNet(D)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.from_numpy(X.astype(np.float32))
    Bo = torch.nn.functional.one_hot(torch.from_numpy(B), MAXB).float()
    Tt = torch.from_numpy(T)
    lossf = nn.CrossEntropyLoss()
    for ep in range(a.epochs):
        perm = np.random.default_rng(ep).permutation(tr)
        net.train()
        for i in range(0, len(perm), 1024):
            bi = torch.from_numpy(perm[i:i + 1024])
            loss = lossf(net(Xt[bi], Bo[bi]), Tt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            tb = torch.from_numpy(te)
            lg = net(Xt[tb], Bo[tb])
            tl = lossf(lg, Tt[tb]).item()
            acc = (lg.argmax(1) == Tt[tb]).float().mean().item()
            ev = (torch.softmax(lg, 1) * torch.arange(MAXB)).sum(1)
            mae = (ev - Tt[tb].float()).abs().mean().item()
        print(f"epoch {ep}: test NLL {tl:.4f} acc {acc:.3f} MAE(E[t]) {mae:.3f}", flush=True)
    torch.save(dict(state_dict=net.state_dict(), in_dim=D), os.path.join(a.out, "distnet.pt"))
    print("saved", os.path.join(a.out, "distnet.pt"))


if __name__ == "__main__":
    main()
