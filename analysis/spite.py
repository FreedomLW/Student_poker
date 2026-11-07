"""T13: does 'spiting' after a certainly-missed bid hurt the opponents? Deal-level duplicate:
all seats dmc_v3; one seat uses wrap mode=spite. Reports delta of opponents' total and own score."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from eval.harness import pmap, _deal_task, bootstrap_ci

BASE = sys.argv[1] if len(sys.argv) > 1 else "store:dmc_v3"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
cand = f"wrap:base={BASE.replace(':', '~')},mode=spite"
print(f"| n_cards | Δ очков соперников (сумма) | Δ своих очков | доля раздач, где спайт сработал |")
print("|---|---|---|---|")
for k in (5, 7, 9):
    tasks = []
    for d in range(N):
        seed = 70_000 + d
        dealer = d % 4
        tasks.append(((BASE,) * 4, seed, None, k, "normal", dealer))
        p = (dealer + 1 + d % 4) % 4
        sp = [BASE] * 4
        sp[p] = cand
        tasks.append((tuple(sp), seed, None, k, "normal", dealer))
    res = pmap(_deal_task, tasks, 4)
    dopp, down, changed = [], [], 0
    for d in range(N):
        b, c = res[2 * d][1], res[2 * d + 1][1]
        dealer = d % 4
        p = (dealer + 1 + d % 4) % 4
        dopp.append((sum(c) - c[p]) - (sum(b) - b[p]))
        down.append(c[p] - b[p])
        changed += int(c != b)
    m1, l1, h1 = bootstrap_ci(dopp)
    m2, l2, h2 = bootstrap_ci(down)
    print(f"| {k} | {m1:+.2f} [{l1:+.2f}, {h1:+.2f}] | {m2:+.2f} [{l2:+.2f}, {h2:+.2f}] | {changed / N:.2f} |", flush=True)
