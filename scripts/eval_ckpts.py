"""Evaluate all checkpoints of a DMC run vs a field (duplicate, full games).
python scripts/eval_ckpts.py runs/dmc/<name> [games=60] [procs=1] [field=heuristic2]"""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.harness import vs_field, bootstrap_ci
run = sys.argv[1]
games = int(sys.argv[2]) if len(sys.argv) > 2 else 60
procs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
field = sys.argv[4] if len(sys.argv) > 4 else "heuristic2"
res_path = os.path.join(run, f"ckpt_eval_{field}.json")
res = json.load(open(res_path)) if os.path.exists(res_path) else {}
single = sorted(glob.glob(os.path.join(run, "ckpt_*.pt")), key=os.path.getmtime) + glob.glob(os.path.join(run, "qnet.pt"))
sep = sorted(glob.glob(os.path.join(run, "bidnet_*.pt")), key=os.path.getmtime) + glob.glob(os.path.join(run, "bidnet.pt"))
specs = [(os.path.basename(c), f"learned:path={c}") for c in single] + \
        [(os.path.basename(b), f"learned:bid_path={b},play_path={b.replace('bidnet', 'playnet')}") for b in sep]
for k, spec in specs:
    if k in res and k not in ("qnet.pt", "bidnet.pt"):
        continue
    res[k] = bootstrap_ci(vs_field(spec, field, games, procs=procs))
    print(k, "%+.2f [%+.2f, %+.2f]" % tuple(res[k]), flush=True)
    json.dump(res, open(res_path, "w"), indent=1)
