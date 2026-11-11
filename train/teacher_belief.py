"""Denoised teacher for BIDS: at every on-policy bid decision, average the value of each legal bid over K worlds
consistent with the bidder's information (opponents' hands resampled, reweighted by the likelihood of their earlier
bids under the net = policy-based inference), each world played to the end by the net (no search beyond).

python -m train.teacher_belief --deals 6000 --k 3 --out runs/teacher/k3_bid_belief.npz
"""
import argparse, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import N_ACTIONS, Obs, default_rules, legal_actions, new_deal, step  # noqa: E402
from env.features import encode  # noqa: E402
from eval.harness import pmap  # noqa: E402
from registry.store import make_agent, STORE, latest_version  # noqa: E402

_C = {}


def _get(spec, K):
    if spec not in _C:
        from agents.pimc import PIMCAgent
        name = spec.split(":", 1)[1]
        v = os.path.join(STORE, name, latest_version(name))
        b, p = os.path.join(v, "bidnet.pt"), os.path.join(v, "playnet.pt")
        _C[spec] = (make_agent(spec), PIMCAgent(samples=K, rollout="net", net_bid=b, net_play=p, net_exact=0,
                                                infer="policy", inf_bid=b, inf_play=p, oversample=3))
    return _C[spec]


def _task(args):
    spec, seed, k, dealer, K = args
    R = default_rules()
    ag, teacher = _get(spec, K)
    teacher.rng.seed(seed)
    s = new_deal(R, k, dealer, seed)
    rows = []
    while s.phase == 0:
        legal = legal_actions(s)
        o = Obs(s, s.cur)
        if len(legal) > 1:
            vals = teacher.action_values(o, legal)
            rows.append((encode(o).astype(np.float16), legal, [vals[a] for a in legal]))
            a = ag.act(o, legal)
        else:
            a = legal[0]
        step(s, a)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="store:best")
    ap.add_argument("--deals", type=int, default=6000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=8_000_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R = default_rules()
    rng = random.Random(a.seed0)
    rows = [r for rs in pmap(_task, [(a.spec, a.seed0 + d, a.k, rng.randrange(R.n_players), a.K) for d in range(a.deals)], a.procs) for r in rs]
    X = np.stack([r[0] for r in rows]); M = np.zeros((len(rows), N_ACTIONS), bool); Y = np.zeros(M.shape, np.float32)
    for i, (_, legal, v) in enumerate(rows):
        M[i, legal] = True; Y[i, legal] = v
    PH = np.zeros(len(rows), np.int8)  # 0 = bid states
    np.savez_compressed(a.out, X=X, PH=PH, M=M, YA=Y, YB=Y)
    print("saved", a.out, len(rows))


if __name__ == "__main__":
    main()
