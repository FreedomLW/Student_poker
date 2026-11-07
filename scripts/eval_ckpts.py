"""Evaluate all checkpoints of a DMC run vs heuristic field (duplicate, full games)."""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.harness import vs_field, bootstrap_ci
run = sys.argv[1]; games = int(sys.argv[2]) if len(sys.argv) > 2 else 60
procs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
res_path = os.path.join(run, "ckpt_eval.json")
res = json.load(open(res_path)) if os.path.exists(res_path) else {}
cks = sorted(glob.glob(os.path.join(run, "ckpt_*.pt")), key=os.path.getmtime) + glob.glob(os.path.join(run, "qnet.pt"))
for ck in cks:
    k = os.path.basename(ck)
    if k in res and k != "qnet.pt":
        continue
    d = vs_field(f"learned:path={ck}", "heuristic", games, procs=procs)
    res[k] = bootstrap_ci(d)
    print(k, "%+.2f [%+.2f, %+.2f]" % tuple(res[k]), flush=True)
    json.dump(res, open(res_path, "w"), indent=1)
