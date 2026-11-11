"""Regret of a net's choices on held-out teacher states: how many points (by label A / label B) its chosen action
loses vs the best action, and how often it picks a non-best action (ties within 0.5 points count as best).

python -m train.teacher_eval runs/teacher/k3_4p_test.npz store:best learned:bid_path=...,play_path=...
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registry.store import make_agent  # noqa: E402

z = np.load(sys.argv[1])
X, M, PH, YA, YB = z["X"].astype(np.float32), z["M"], z["PH"], z["YA"], z["YB"]
for spec in sys.argv[2:]:
    ag = make_agent(spec)
    nets = {0: ag.bid_net, 1: ag.net}
    Q = np.zeros(M.shape, np.float32)
    with torch.no_grad():
        for ph in (0, 1):
            m = PH == ph
            Q[m] = nets[ph](torch.from_numpy(X[m])).numpy()
    Q[~M] = -1e9
    ch = Q.argmax(1)
    out = []
    for lab, Y in (("A(перебор)", YA), ("B(ход+сеть)", YB)):
        Yb = np.where(M, Y, -1e9)
        best = Yb.max(1)
        got = Y[np.arange(len(Y)), ch]
        reg = best - got
        for ph, nm in ((0, "заказ"), (1, "ход")):
            m = PH == ph
            out.append(f"{lab} {nm}: потеря {reg[m].mean():.2f} очк, ошибка {np.mean(reg[m] > 0.5):.3f}")
    print(spec[:60], "|", " | ".join(out), flush=True)
