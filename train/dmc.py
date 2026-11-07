"""Deep Monte Carlo self-play training (DouZero-style, action-value head over the 59 unified actions).

Target for every decision (bid or card) of a learning seat = that player's final DEAL score / 10.
Actors: N processes, each running B deals in lockstep with batched inference; epsilon-greedy over legal actions.
Deals are drawn uniformly from the game schedule (sizes 1..9, normal / no-trump / dark), random dealer.
Opponent mix per deal: with prob p_heur 1..3 seats are the rule-based heuristic (not trained on),
with prob p_past seats are a frozen past snapshot (league), else full self-play.

python -m train.dmc --name dmc_v1 --minutes 120 --actors 3
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import random
import time

import numpy as np
import torch

from agents.heuristic import heuristic_bid, heuristic_play
from agents.learned import QNet, save_qnet, load_qnet
from engine import N_ACTIONS, Obs, default_rules, legal_actions, new_deal, scores, step
from env.features import encode, obs_dim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def actor(aid, shared, past_shared, cfg, out_q, stop):
    torch.set_num_threads(1)
    rng = random.Random(1000 + aid)
    R = default_rules()
    n = R.n_players
    D = obs_dim(n, R.max_cards)
    local = QNet(D, cfg["hidden"])
    local.load_state_dict(shared.state_dict())
    past = QNet(D, cfg["hidden"])
    past.load_state_dict(past_shared.state_dict())
    B = cfg["batch_deals"]
    sched = R.schedule
    eps = cfg["eps"]

    def new_slot():
        k, t = rng.choice(sched)
        s = new_deal(R, k, rng.randrange(n), rng.getrandbits(63), dtype=t)
        r = rng.random()
        roles = ["net"] * n
        if r < cfg["p_heur"]:
            for p in rng.sample(range(n), rng.randint(1, 3)):
                roles[p] = "heur"
        elif r < cfg["p_heur"] + cfg["p_past"]:
            for p in rng.sample(range(n), rng.randint(1, 3)):
                roles[p] = "past"
        return dict(s=s, roles=roles, traj=[[] for _ in range(n)])

    slots = [new_slot() for _ in range(B)]
    xs, acts, rets, phases = [], [], [], []
    steps = 0
    buf = np.zeros((B, D), np.float32)
    while not stop.is_set():
        # advance heuristic seats / forced moves; collect net decisions
        pend = []
        for i, sl in enumerate(slots):
            s = sl["s"]
            while True:
                if s.phase == 2:
                    sc = scores(s)
                    for p in range(n):
                        for (x, a, ph) in sl["traj"][p]:
                            xs.append(x)
                            acts.append(a)
                            rets.append(sc[p] / 10.0)
                            phases.append(ph)
                    slots[i] = sl = new_slot()
                    s = sl["s"]
                    continue
                p = s.cur
                legal = legal_actions(s)
                role = sl["roles"][p]
                if len(legal) == 1:
                    step(s, legal[0])
                    continue
                if role == "heur":
                    h = 0 if (s.dark and s.phase == 0) else s.hands[p]
                    a = heuristic_bid(s, p, h, legal) if s.phase == 0 else heuristic_play(s, p, h, legal)
                    step(s, a)
                    continue
                pend.append((i, p, legal, role))
                break
        if not pend:
            continue
        for j, (i, p, legal, role) in enumerate(pend):
            encode(Obs(slots[i]["s"], p), buf[j])
        xb = torch.from_numpy(buf[: len(pend)])
        with torch.no_grad():
            qn = local(xb).numpy()
            if any(r == "past" for (_, _, _, r) in pend):
                qp = past(xb).numpy()
            else:
                qp = None
        for j, (i, p, legal, role) in enumerate(pend):
            q = qp[j] if role == "past" else qn[j]
            if role == "net" and rng.random() < eps:
                a = rng.choice(legal)
            else:
                a = max(legal, key=lambda z: q[z])
            s = slots[i]["s"]
            if role == "net":
                slots[i]["traj"][p].append((buf[j].astype(np.float16), a, s.phase))
            step(s, a)
        steps += 1
        if len(xs) >= cfg["chunk"]:
            out_q.put((np.stack(xs), np.array(acts, np.int16), np.array(rets, np.float32), np.array(phases, np.int8)))
            xs, acts, rets, phases = [], [], [], []
        if steps % cfg["sync_every"] == 0:
            local.load_state_dict(shared.state_dict())
            past.load_state_dict(past_shared.state_dict())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dmc_v1")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--actors", type=int, default=3)
    ap.add_argument("--batch-deals", type=int, default=64)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--p-heur", type=float, default=0.2)
    ap.add_argument("--p-past", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 256])
    ap.add_argument("--init", default=None, help="init weights from qnet.pt")
    ap.add_argument("--ckpt-minutes", type=float, default=10)
    ap.add_argument("--learner-threads", type=int, default=1)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.learner_threads)
    R = default_rules()
    D = obs_dim(R.n_players, R.max_cards)
    model = QNet(D, a.hidden)
    if a.init:
        model.load_state_dict(load_qnet(a.init).state_dict())
    shared = QNet(D, a.hidden)
    shared.load_state_dict(model.state_dict())
    shared.share_memory()
    past_shared = QNet(D, a.hidden)
    past_shared.load_state_dict(model.state_dict())
    past_shared.share_memory()
    cfg = dict(hidden=a.hidden, batch_deals=a.batch_deals, eps=a.eps, p_heur=a.p_heur, p_past=a.p_past,
               chunk=4096, sync_every=50)
    out_dir = os.path.join(ROOT, "runs", "dmc", a.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    ctx = mp.get_context("fork")
    q = ctx.Queue(maxsize=64)
    stop = ctx.Event()
    procs = [ctx.Process(target=actor, args=(i, shared, past_shared, cfg, q, stop), daemon=True)
             for i in range(a.actors)]
    for p in procs:
        p.start()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    cap = a.buffer
    X = np.zeros((cap, D), np.float16)
    A = np.zeros(cap, np.int64)
    G = np.zeros(cap, np.float32)
    PH = np.zeros(cap, np.int8)
    size = 0
    ptr = 0
    total = 0
    t0 = time.time()
    last_ck = t0
    last_log = t0
    upd = 0
    losses = []
    past_pool = []
    logf = open(os.path.join(out_dir, "log.csv"), "a", newline="")
    lw = csv.writer(logf)
    lw.writerow(["time_s", "samples", "updates", "loss", "loss_bid", "mean_ret"])
    rng = np.random.default_rng(0)
    rets_recent = []
    while time.time() - t0 < a.minutes * 60:
        got = 0
        while True:
            try:
                xs, acts, rets, ph = q.get(timeout=0.01 if size > 0 else 5)
            except queue.Empty:
                break
            m = len(acts)
            idx = (ptr + np.arange(m)) % cap
            X[idx], A[idx], G[idx], PH[idx] = xs, acts, rets, ph
            ptr = (ptr + m) % cap
            size = min(cap, size + m)
            total += m
            got += m
            rets_recent.append(rets.mean())
            if got > 20000:
                break
        if size < 20000:
            continue
        # a few SGD steps per arriving chunk (replay ratio ~ 1-2)
        n_upd = max(1, int(2 * got / a.bs)) if got else 1
        for _ in range(n_upd):
            bi = rng.integers(0, size, a.bs)
            xb = torch.from_numpy(X[bi].astype(np.float32))
            ab = torch.from_numpy(A[bi])
            gb = torch.from_numpy(G[bi])
            qv = model(xb).gather(1, ab[:, None])[:, 0]
            loss = ((qv - gb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            upd += 1
            losses.append(loss.item())
        if upd % 20 == 0:
            shared.load_state_dict(model.state_dict())
        now = time.time()
        if now - last_log > 30:
            with torch.no_grad():
                bi = rng.integers(0, size, 4096)
                isb = PH[bi] == 0
                xb = torch.from_numpy(X[bi].astype(np.float32))
                qv = model(xb).gather(1, torch.from_numpy(A[bi])[:, None])[:, 0].numpy()
                lb = float(((qv - G[bi]) ** 2)[isb].mean()) if isb.any() else float("nan")
            lw.writerow([round(now - t0), total, upd, round(float(np.mean(losses[-200:])), 4), round(lb, 4),
                         round(float(np.mean(rets_recent[-50:])), 4)])
            logf.flush()
            print(f"[{(now - t0) / 60:.1f}m] samples={total} upd={upd} loss={np.mean(losses[-200:]):.4f} "
                  f"bidloss={lb:.4f} ret={np.mean(rets_recent[-50:]):.3f} rate={total / (now - t0):.0f}/s", flush=True)
            last_log = now
        if now - last_ck > a.ckpt_minutes * 60:
            path = os.path.join(out_dir, f"ckpt_{total // 1000}k.pt")
            save_qnet(model, path, samples=total, updates=upd)
            past_pool.append(path)
            # league: past snapshot drawn uniformly from previous checkpoints
            past_shared.load_state_dict(load_qnet(random.choice(past_pool)).state_dict())
            last_ck = now
    stop.set()
    path = os.path.join(out_dir, "qnet.pt")
    save_qnet(model, path, samples=total, updates=upd)
    print("saved", path, "samples", total)
    for p in procs:
        p.terminate()


if __name__ == "__main__":
    main()
