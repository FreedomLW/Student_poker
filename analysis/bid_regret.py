"""Hindsight bid regret: replay each deal with one seat's bid shifted (+1, +2, -1), same cards & opponents
(deterministic agents => exact counterfactual). Reports how often and by how much a different bid would have paid."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from eval.harness import deal_vs_field

spec = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
procs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
base = spec.replace(":", "~")
print(f"| n | P(+1 или +2 дали бы больше) | средний выигрыш «задним числом» от большего заказа | P(−1 дал бы больше) | выигрыш от меньшего | P(текущий заказ лучший из 4) |")
print("|---|---|---|---|---|---|")
for k in (2, 3, 5, 7, 9):
    r = {}
    for d in (1, 2, -1):
        res = deal_vs_field(f"wrap:base={base},mode=shade,delta={d}", spec, k, N, procs=procs, seed0=80_000)
        r[d] = np.concatenate([res[s] for s in range(4)])
    up = np.maximum(r[1], r[2])
    print(f"| {k} | {np.mean(up > 0):.2f} | {np.mean(np.maximum(up, 0)):.2f} | {np.mean(r[-1] > 0):.2f} | "
          f"{np.mean(np.maximum(r[-1], 0)):.2f} | {np.mean((up <= 0) & (r[-1] <= 0)):.2f} |", flush=True)
