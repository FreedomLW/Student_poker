"""Bid by argmax expected score under the learned trick distribution P(t | obs, bid) (train/bid_dist.py);
play with any other agent spec."""
from __future__ import annotations

import os

import numpy as np
import torch

from engine.core import BID_OFFSET, PH_BID
from env.features import encode
from .base import Agent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BidDistAgent(Agent):
    name = "biddist"

    def __init__(self, seed: int = 0, play: str = "store:dmc_v3", dist: str = "runs/bid_dist/distnet.pt",
                 model_dir=None, **kw):
        from registry.store import make_agent
        from train.bid_dist import DistNet, MAXB
        torch.set_num_threads(1)
        self.p = make_agent(play, seed=seed)
        path = dist if os.path.isabs(dist) else os.path.join(ROOT, dist)
        if model_dir and os.path.exists(os.path.join(model_dir, "distnet.pt")):
            path = os.path.join(model_dir, "distnet.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = DistNet(ck["in_dim"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.MAXB = MAXB

    def reseed(self, seed):
        self.p.reseed(seed)

    def new_deal(self):
        self.p.new_deal()

    def act(self, obs, legal):
        if obs.phase != PH_BID:
            return self.p.act(obs, legal)
        x = torch.from_numpy(encode(obs))[None].repeat(len(legal), 1)
        bs = torch.tensor([a - BID_OFFSET for a in legal])
        bo = torch.nn.functional.one_hot(bs, self.MAXB).float()
        with torch.no_grad():
            pr = torch.softmax(self.net(x, bo), 1).numpy()
        r = obs.rules
        best, bv = legal[0], -1e9
        for i, a in enumerate(legal):
            b = a - BID_OFFSET
            ev = sum(pr[i, t] * r.score(b, t, obs.dark) for t in range(obs.n_cards + 1))
            if ev > bv:
                bv, best = ev, a
        return best
