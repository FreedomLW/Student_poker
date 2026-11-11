"""Build viewer/advisor.html (offline bid/play advisor) from the best 3- and 4-player models, plus parity test vectors.

python -m analysis.export_advisor
"""
from __future__ import annotations

import base64
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {4: ("rules.yaml", "best"), 3: ("rules_variants/house_3p.yaml", "best_3p")}


def net_spec(qnet):
    layers = []
    lin = [m for m in qnet.net if hasattr(m, "weight")]
    for m in lin:
        W = m.weight.detach().numpy().astype(np.float16)
        b = m.bias.detach().numpy().astype(np.float16)
        layers.append(dict(**{"in": int(W.shape[1]), "out": int(W.shape[0])},
                           W=base64.b64encode(W.tobytes()).decode(), b=base64.b64encode(b.tobytes()).decode()))
    return dict(layers=layers)


def test_vectors(n, spec, n_deals=60):
    """Random self-play decisions: setup + action log + expected legal/encode/Q (python)."""
    from engine import Obs, default_rules, legal_actions, new_deal, step
    from env.features import encode
    from registry.store import make_agent
    import engine.rules as er
    er._DEFAULT = None
    R = default_rules()
    ag = make_agent(spec)
    rng = random.Random(5)
    out = []
    for d in range(n_deals):
        k = rng.randint(1, R.max_cards)
        dealer = rng.randrange(n)
        s = new_deal(R, k, dealer, 900_000 + d)
        init = [list(int(c) for c in range(36) if h >> c & 1) for h in s.hands]
        log = []
        while not s.done:
            p = s.cur
            legal = legal_actions(s)
            o = Obs(s, p)
            if len(legal) > 1 and rng.random() < 0.3:
                q = ag.q_values(o)
                out.append(dict(setup=dict(n=n, k=k, dealer=dealer, me=p, myHand=init[p], dark=False, hookMin=R.hook_min_cards),
                                log=log[:], legal=legal, x=encode(o).round(5).tolist(), q=[float(q[a]) for a in legal]))
            a = ag.act(o, legal) if len(legal) > 1 else legal[0]
            log.append(int(a))
            step(s, a)
    return out


def main():
    from agents.learned import load_qnet
    from registry.store import STORE, latest_version
    data, tests = {}, {}
    for n, (rules_path, name) in MODELS.items():
        os.environ["ODESSA_RULES"] = rules_path
        vdir = os.path.join(STORE, name, latest_version(name))
        meta = json.load(open(os.path.join(vdir, "meta.json")))
        data[str(n)] = dict(name=f"{name} ({meta.get('alias_of', '')})",
                            bid=net_spec(load_qnet(os.path.join(vdir, "bidnet.pt"))),
                            play=net_spec(load_qnet(os.path.join(vdir, "playnet.pt"))))
        tests[str(n)] = test_vectors(n, f"store:{name}")
        print(n, "players:", name, "tests", len(tests[str(n)]), flush=True)
    core = open(os.path.join(ROOT, "viewer", "advisor_core.js")).read()
    html = open(os.path.join(ROOT, "viewer", "advisor_template.html")).read()
    html = html.replace("/*__CORE__*/", core).replace("/*__MODELS__*/null", json.dumps(data, separators=(",", ":")))
    with open(os.path.join(ROOT, "viewer", "advisor.html"), "w") as f:
        f.write(html)
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", "advisor_models.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))
    with open(os.path.join(ROOT, "runs", "advisor_tests.json"), "w") as f:
        json.dump(tests, f)
    print("saved viewer/advisor.html", os.path.getsize(os.path.join(ROOT, "viewer", "advisor.html")) // 1024, "KB")


if __name__ == "__main__":
    main()
