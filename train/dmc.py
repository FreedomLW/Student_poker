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
from agents.heuristic2 import smart_bid, smart_play
from agents.learned import QNet, save_qnet, load_qnet
from engine import N_ACTIONS, Obs, default_rules, legal_actions, new_deal, scores, step
from env.features import encode, obs_dim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def actor(aid, shared, past_shared, cfg, out_q, stop):
    """shared / past_shared: dict phase(0 bid,1 play) -> QNet (may be the same object)."""
    torch.set_num_threads(1)
    rng = random.Random(1000 + aid)
    R = default_rules()
    n = R.n_players
    D = obs_dim(n, R.max_cards)
    def clone(dct):
        out = {}
        for ph, m in dct.items():
            c = QNet(D, m.hidden)
            c.load_state_dict(m.state_dict())
            out[ph] = c
        return out

    def sync(dst, src):
        for ph in dst:
            dst[ph].load_state_dict(src[ph].state_dict())

    local = clone(shared)
    past = clone(past_shared)
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
                roles[p] = "heur2" if rng.random() < cfg["p_heur2"] else "heur"
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
                if role in ("heur", "heur2"):
                    h = 0 if (s.dark and s.phase == 0) else s.hands[p]
                    if role == "heur":
                        a = heuristic_bid(s, p, h, legal) if s.phase == 0 else heuristic_play(s, p, h, legal)
                    else:
                        a = smart_bid(s, p, h, legal) if s.phase == 0 else smart_play(s, p, h, legal)
                    step(s, a)
                    continue
                pend.append((i, p, legal, role))
                break
        if not pend:
            continue
        phs = np.zeros(len(pend), np.int64)
        for j, (i, p, legal, role) in enumerate(pend):
            encode(Obs(slots[i]["s"], p), buf[j])
            phs[j] = 0 if slots[i]["s"].phase == 0 else 1
        xb = torch.from_numpy(buf[: len(pend)])
        qn = np.zeros((len(pend), N_ACTIONS), np.float32)
        qp = np.zeros((len(pend), N_ACTIONS), np.float32)
        has_past = np.array([r == "past" for (_, _, _, r) in pend])
        with torch.no_grad():
            for ph in (0, 1):
                m = phs == ph
                if m.any():
                    qn[m] = local[ph](xb[torch.from_numpy(m)]).numpy()
                    mp_ = m & has_past
                    if mp_.any():
                        qp[mp_] = past[ph](xb[torch.from_numpy(mp_)]).numpy()
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
            sync(local, shared)
            sync(past, past_shared)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dmc_v1")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--actors", type=int, default=3)
    ap.add_argument("--batch-deals", type=int, default=64)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--p-heur", type=float, default=0.2)
    ap.add_argument("--p-heur2", type=float, default=0.0, help="fraction of heuristic seats that are heuristic2")
    ap.add_argument("--p-past", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 256])
    ap.add_argument("--bid-hidden", type=int, nargs="+", default=None)
    ap.add_argument("--separate", action="store_true", help="separate bid and play networks")
    ap.add_argument("--init", default=None, help="init weights from qnet.pt (both nets when --separate)")
    ap.add_argument("--init-bid", default=None)
    ap.add_argument("--init-play", default=None)
    ap.add_argument("--ckpt-minutes", type=float, default=10)
    ap.add_argument("--learner-threads", type=int, default=1)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.learner_threads)
    R = default_rules()
    D = obs_dim(R.n_players, R.max_cards)
    phases = (0, 1) if a.separate else (0,)
    hid = {0: a.bid_hidden or a.hidden, 1: a.hidden}
    models = {}
    for ph in phases:
        m = QNet(D, hid[ph])
        ip = (a.init_bid if ph == 0 else a.init_play) or a.init
        if ip:
            src = load_qnet(ip)
            if src.hidden == m.hidden:
                m.load_state_dict(src.state_dict())
        models[ph] = m
    if not a.separate:
        models[1] = models[0]

    def make_shared():
        d = {}
        for ph in phases:
            sm = QNet(D, hid[ph])
            sm.load_state_dict(models[ph].state_dict())
            sm.share_memory()
            d[ph] = sm
        if not a.separate:
            d[1] = d[0]
        return d

    shared = make_shared()
    past_shared = make_shared()
    cfg = dict(hidden=a.hidden, batch_deals=a.batch_deals, eps=a.eps, p_heur=a.p_heur, p_heur2=a.p_heur2,
               p_past=a.p_past, chunk=4096, sync_every=50)
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
    opts = {ph: torch.optim.Adam(models[ph].parameters(), lr=a.lr) for ph in phases}
    # buffers per learning group (group = phase if separate else 0)
    groups = phases
    cap = {g: (a.buffer if (g == 1 or not a.separate) else a.buffer // 4) for g in groups}
    BUF = {g: dict(X=np.zeros((cap[g], D), np.float16), A=np.zeros(cap[g], np.int64), G=np.zeros(cap[g], np.float32),
                   PH=np.zeros(cap[g], np.int8), size=0, ptr=0) for g in groups}
    total = 0
    t0 = time.time()
    last_ck = t0
    last_log = t0
    upd = 0
    losses = {g: [] for g in groups}
    past_pool = []
    logf = open(os.path.join(out_dir, "log.csv"), "a", newline="")
    lw = csv.writer(logf)
    lw.writerow(["time_s", "samples", "updates", "loss", "loss_bid", "mean_ret"])
    rng = np.random.default_rng(0)
    rets_recent = []

    def add(g, xs, acts, rets, ph):
        B = BUF[g]
        m = len(acts)
        idx = (B["ptr"] + np.arange(m)) % cap[g]
        B["X"][idx], B["A"][idx], B["G"][idx], B["PH"][idx] = xs, acts, rets, ph
        B["ptr"] = (B["ptr"] + m) % cap[g]
        B["size"] = min(cap[g], B["size"] + m)

    def save_all(tag, **extra):
        if a.separate:
            save_qnet(models[0], os.path.join(out_dir, f"bidnet{tag}.pt"), **extra)
            save_qnet(models[1], os.path.join(out_dir, f"playnet{tag}.pt"), **extra)
            return os.path.join(out_dir, f"bidnet{tag}.pt"), os.path.join(out_dir, f"playnet{tag}.pt")
        path = os.path.join(out_dir, f"qnet{tag}.pt" if tag == "" else f"ckpt{tag}.pt")
        save_qnet(models[0], path, **extra)
        return path, path

    while time.time() - t0 < a.minutes * 60:
        got = {g: 0 for g in groups}
        while True:
            try:
                xs, acts, rets, ph = q.get(timeout=0.01 if total > 0 else 5)
            except queue.Empty:
                break
            if a.separate:
                for g in groups:
                    m = ph == g
                    if m.any():
                        add(g, xs[m], acts[m], rets[m], ph[m])
                        got[g] += int(m.sum())
            else:
                add(0, xs, acts, rets, ph)
                got[0] += len(acts)
            total += len(acts)
            rets_recent.append(rets.mean())
            if sum(got.values()) > 20000:
                break
        for g in groups:
            B = BUF[g]
            if B["size"] < (20000 if g == 1 or not a.separate else 4000):
                continue
            n_upd = max(1, int(2 * got[g] / a.bs)) if got[g] else (1 if g == 1 or not a.separate else 0)
            for _ in range(n_upd):
                bi = rng.integers(0, B["size"], a.bs)
                xb = torch.from_numpy(B["X"][bi].astype(np.float32))
                ab = torch.from_numpy(B["A"][bi])
                gb = torch.from_numpy(B["G"][bi])
                qv = models[g](xb).gather(1, ab[:, None])[:, 0]
                loss = ((qv - gb) ** 2).mean()
                opts[g].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(models[g].parameters(), 10.0)
                opts[g].step()
                upd += 1
                losses[g].append(loss.item())
                if upd % 20 == 0:
                    for ph in phases:
                        shared[ph].load_state_dict(models[ph].state_dict())
        now = time.time()
        if now - last_log > 30 and all(BUF[g]["size"] > 0 for g in groups):
            lg = groups[-1]
            lb = float(np.mean(losses[0][-200:])) if (a.separate and losses[0]) else float("nan")
            ml = float(np.mean(losses[lg][-200:])) if losses[lg] else float("nan")
            lw.writerow([round(now - t0), total, upd, round(ml, 4), round(lb, 4),
                         round(float(np.mean(rets_recent[-50:])), 4)])
            logf.flush()
            print(f"[{(now - t0) / 60:.1f}m] samples={total} upd={upd} loss={ml:.4f} bidloss={lb:.4f} "
                  f"ret={np.mean(rets_recent[-50:]):.3f} rate={total / (now - t0):.0f}/s", flush=True)
            last_log = now
        if now - last_ck > a.ckpt_minutes * 60:
            tag = f"_{total // 1000}k"
            pb, pp = save_all(tag, samples=total, updates=upd)
            past_pool.append((pb, pp))
            cb, cp = random.choice(past_pool)
            past_shared[0].load_state_dict(load_qnet(cb).state_dict())
            if a.separate:
                past_shared[1].load_state_dict(load_qnet(cp).state_dict())
            last_ck = now
    stop.set()
    pb, pp = save_all("", samples=total, updates=upd)
    print("saved", pb, pp, "samples", total)
    for p in procs:
        p.terminate()


if __name__ == "__main__":
    main()
