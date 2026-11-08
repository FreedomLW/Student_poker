"""Print full self-play deals of an agent (all hands visible) for manual review."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import default_rules, new_deal, play_deal, NR, bit_list, card_str
from registry.store import make_agent

spec, k, n, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0
R = default_rules()
ags = [make_agent(spec, seed=i) for i in range(4)]


def hand_str(h):
    by = {}
    for c in bit_list(h):
        by.setdefault(c // NR, []).append(card_str(c) if not (1 << c) & R.joker_mask else "JOKER")
    return "  ".join(" ".join(reversed(v)) for su, v in sorted(by.items(), key=lambda t: (t[0] != 2, t[0])))


for d in range(n):
    s = new_deal(R, k, d % 4, seed0 + d)
    init = s.hands[:]
    for a in ags:
        a.new_deal()
    sc = play_deal(s, ags)
    print(f"--- deal {seed0 + d}: {k} cards, dealer P{s.dealer}, bid order from P{s.first_seat}")
    for p in [(s.first_seat + j) % 4 for j in range(4)]:
        print(f"  P{p}: bid {s.bids[p]} took {s.won[p]} score {sc[p]:+3d} | {hand_str(init[p])}")
    print(f"  sum bids {sum(s.bids)} / {k}")
