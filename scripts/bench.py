"""Engine speed benchmark: random & heuristic full games / deals per second."""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import default_rules, new_deal, legal_actions, step, play_game
from agents import RandomAgent, HeuristicAgent

R = default_rules()

def raw_random(n_deals=3000, k=9):
    rng = random.Random(0); t = time.time(); moves = 0
    for i in range(n_deals):
        s = new_deal(R, k, i % 4, rng)
        while s.phase != 2:
            a = legal_actions(s); step(s, a[rng.randrange(len(a))]); moves += 1
    dt = time.time() - t
    return n_deals / dt, moves / dt

def games(agent_cls, n=50):
    t = time.time()
    for g in range(n):
        play_game(R, [agent_cls() for _ in range(4)], g)
    return n / (time.time() - t)

if __name__ == "__main__":
    d, m = raw_random()
    print(f"raw engine (random policy, 9-card deals): {d:.0f} deals/s, {m:.0f} actions/s")
    print(f"full games random agents (with Obs): {games(RandomAgent):.1f} games/s")
    print(f"full games heuristic agents: {games(HeuristicAgent):.1f} games/s")
