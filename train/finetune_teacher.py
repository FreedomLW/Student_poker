"""Targeted fine-tuning on teacher labels (train/teacher.py) with an anti-forgetting anchor.

loss = masked all-action MSE on teacher states (label A = enumeration/hindsight, or B = one-step with net continuation)
     + anchor_w * masked MSE on ordinary self-play states of ALL deal sizes towards the frozen base net's own Q-values.

python -m train.finetune_teacher --data runs/teacher/k3_4p.npz --label B --out runs/teacher/ftB
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.learned import load_qnet, save_qnet  # noqa: E402
from engine import N_ACTIONS, Obs, default_rules, legal_actions, new_deal, step  # noqa: E402
from env.features import encode  # noqa: E402
from registry.store import STORE, make_agent  # noqa: E402


def anchor_states(spec, n_deals, seed=123):
    """Encoded observations of ordinary self-play decisions (all schedule deal sizes) + legal masks + phase."""
    R = default_rules()
    ag = make_agent(spec)
    rng = random.Random(seed)
    X, M, PH = [], [], []
    for d in range(n_deals):
        k, t = R.schedule[d % len(R.schedule)]
        s = new_deal(R, k, rng.randrange(R.n_players), rng.getrandbits(63), dtype=t)
        while not s.done:
            legal = legal_actions(s)
            o = Obs(s, s.cur)
            if len(legal) > 1:
                X.append(encode(o))
                m = np.zeros(N_ACTIONS, bool)
                m[legal] = True
                M.append(m)
                PH.append(0 if s.phase == 0 else 1)  # 0 = bid net, 1 = play net
                a = ag.act(o, legal)
            else:
                a = legal[0]
            step(s, a)
    return np.stack(X), np.stack(M), np.array(PH, np.int8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", choices=["A", "B"], required=True)
    ap.add_argument("--base", default="store:best")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--anchor-deals", type=int, default=3000)
    ap.add_argument("--anchor-w", type=float, default=1.0)
    ap.add_argument("--phases", default="01", help="which nets to fine-tune: 0=bid, 1=play")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    name = a.base.split(":", 1)[1]
    vdir = os.path.join(STORE, name, sorted(os.listdir(os.path.join(STORE, name)), key=lambda v: (len(v), v))[-1])
    nets = {0: load_qnet(os.path.join(vdir, "bidnet.pt")), 1: load_qnet(os.path.join(vdir, "playnet.pt"))}
    frozen = {0: load_qnet(os.path.join(vdir, "bidnet.pt")), 1: load_qnet(os.path.join(vdir, "playnet.pt"))}
    z = np.load(a.data)
    TX, TM, TPH = z["X"].astype(np.float32), z["M"], z["PH"]
    TY = (z["YA"] if a.label == "A" else z["YB"]) / 10.0
    AX, AM, APH = anchor_states(a.base, a.anchor_deals)
    with torch.no_grad():
        AY = np.zeros(AM.shape, np.float32)
        for ph in (0, 1):
            m = APH == ph
            AY[m] = frozen[ph](torch.from_numpy(AX[m])).numpy()
    print(f"teacher states {len(TX)} (bid {int((TPH == 0).sum())}), anchor states {len(AX)}", flush=True)
    opts = {ph: torch.optim.Adam(nets[ph].parameters(), lr=a.lr) for ph in (0, 1)}
    for ep in range(a.epochs):
        losses = {0: [], 1: []}
        for ph in (0, 1):
            if str(ph) not in a.phases:
                continue
            ti = np.flatnonzero(TPH == ph)
            ai = np.flatnonzero(APH == ph)
            if len(ti) == 0:
                continue
            nets[ph].train()
            perm = np.random.default_rng(ep).permutation(ti)
            for j in range(0, len(perm), a.bs):
                b = perm[j:j + a.bs]
                ab = np.random.default_rng(ep * 1000 + j).choice(ai, size=len(b))
                xt, mt, yt = map(torch.from_numpy, (TX[b], TM[b], TY[b]))
                xa, ma, ya = map(torch.from_numpy, (AX[ab], AM[ab], AY[ab]))
                qt = nets[ph](xt)
                qa = nets[ph](xa)
                lt = (((qt - yt) ** 2) * mt).sum() / mt.sum()
                la = (((qa - ya) ** 2) * ma).sum() / ma.sum()
                loss = lt + a.anchor_w * la
                opts[ph].zero_grad()
                loss.backward()
                opts[ph].step()
                losses[ph].append((lt.item(), la.item()))
        print(f"epoch {ep}: " + "  ".join(f"{'bid' if ph == 0 else 'play'} teacher {np.mean([x[0] for x in v]):.4f} anchor {np.mean([x[1] for x in v]):.4f}"
                                     for ph, v in losses.items() if v), flush=True)
    os.makedirs(a.out, exist_ok=True)
    for ph, fn in ((0, "bidnet.pt"), (1, "playnet.pt")):
        nets[ph].eval()
        save_qnet(nets[ph], os.path.join(a.out, fn), features="v1", teacher=a.label, data=a.data)
    print("saved", a.out)


if __name__ == "__main__":
    main()
