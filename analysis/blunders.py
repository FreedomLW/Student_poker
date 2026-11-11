"""Count 'burned high cards': following, I still need tricks, my chosen plain card does not beat the current best card
of the trick, yet I hold a LOWER card of the same suit that also does not beat it (so the higher card is wasted)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agents.heuristic import card_key, trick_best_key
from engine import Obs, default_rules, legal_actions, new_deal, step
from eval.harness import pmap, _get_agents


def _task(args):
    spec, seed, k, t, dealer = args
    R = default_rules()
    ags = _get_agents((spec,) * R.n_players, seed, R)
    s = new_deal(R, k, dealer, seed, dtype=t)
    out = []
    while not s.done:
        p = s.cur
        legal = legal_actions(s)
        a = ags[p].act(Obs(s, p), legal) if len(legal) > 1 else legal[0]
        if s.phase == 1 and s.t_players and a < 36 and s.bids[p] - s.won[p] > 0:
            best = trick_best_key(s)
            if card_key(a, s.trump, s.lead_suit) <= best:
                lower = [c for c in legal if c < 36 and c // 9 == a // 9 and c < a and card_key(c, s.trump, s.lead_suit) <= best]
                out.append((k, int(bool(lower)), int(a // 9 == s.trump)))
        step(s, a)
    return out


if __name__ == "__main__":
    spec, n = sys.argv[1], int(sys.argv[2])
    R = default_rules()
    import random
    rng = random.Random(1)
    sched = list(R.schedule)
    rows = np.array([r for rs in pmap(_task, [(spec, 500_000 + d, *sched[d % len(sched)], rng.randrange(R.n_players)) for d in range(n)], 4) for r in rs])
    k, waste, tr = rows.T
    print(f"{spec}: ситуаций «нужна взятка, но эту не взять»: {len(rows)}; сожжена старшая карта при наличии младшей той же масти: "
          f"{waste.mean():.3f} (козырем: {waste[tr == 1].mean() if (tr == 1).any() else float('nan'):.3f}, N козырных {int((tr == 1).sum())})")
