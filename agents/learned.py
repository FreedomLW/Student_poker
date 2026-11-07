"""Learned agent: one MLP Q-network over the unified 59-action space (bids + plays), trained by Deep Monte Carlo
(train/dmc.py). Optional: use heuristic/other source for bids or plays (for ablations)."""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

from engine.core import N_ACTIONS, PH_BID
from env.features import encode, obs_dim
from .base import Agent
from .heuristic import heuristic_bid, heuristic_play


class QNet(nn.Module):
    def __init__(self, in_dim: int, hidden=(512, 512, 256), out_dim: int = N_ACTIONS):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.hidden = tuple(hidden)

    def forward(self, x):
        return self.net(x)

    def trunk(self, x):
        """Last hidden layer (used by auxiliary heads during training)."""
        return self.net[:-1](x)

    def forward_with_trunk(self, x):
        h = self.net[:-1](x)
        return self.net[-1](h), h


def load_qnet(path: str) -> QNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = QNet(ck["in_dim"], ck["hidden"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


def save_qnet(net: QNet, path: str, **extra):
    torch.save(dict(in_dim=net.in_dim, hidden=net.hidden, state_dict=net.state_dict(), **extra), path)


class LearnedAgent(Agent):
    name = "learned"

    def __init__(self, seed: int = 0, model_dir: str | None = None, path: str | None = None, eps: float = 0.0,
                 bid: str = "net", play: str = "net", bid_path: str | None = None, play_path: str | None = None,
                 **kw):
        """Either a single net (path / model_dir/qnet.pt) or separate bid & play nets
        (bid_path+play_path, or model_dir/{bidnet,playnet}.pt)."""
        torch.set_num_threads(1)
        self.rng = random.Random(seed)
        if bid_path is None and model_dir and os.path.exists(os.path.join(model_dir, "bidnet.pt")):
            bid_path = os.path.join(model_dir, "bidnet.pt")
            play_path = os.path.join(model_dir, "playnet.pt")
        if bid_path:
            self.bid_net = load_qnet(bid_path)
            self.net = load_qnet(play_path)
        else:
            if path is None:
                path = os.path.join(model_dir, "qnet.pt")
            self.net = load_qnet(path)
            self.bid_net = self.net
        self.eps = eps
        self.bid_src = bid
        self.play_src = play
        self._buf = None

    def q_values(self, obs) -> np.ndarray:
        if self._buf is None:
            self._buf = np.zeros(obs_dim(obs.n, obs.rules.max_cards), np.float32)
        x = encode(obs, self._buf)
        net = self.bid_net if obs.phase == PH_BID else self.net
        with torch.no_grad():
            return net(torch.from_numpy(x)[None])[0].numpy()

    def act(self, obs, legal):
        if obs.phase == PH_BID and self.bid_src == "heuristic":
            return heuristic_bid(obs, obs.me, obs.hand, legal)
        if obs.phase != PH_BID and self.play_src == "heuristic":
            return heuristic_play(obs, obs.me, obs.hand, legal)
        if self.eps and self.rng.random() < self.eps:
            return self.rng.choice(legal)
        q = self.q_values(obs)
        return max(legal, key=lambda a: q[a])
