import random

from engine import Obs, default_rules, legal_actions, new_deal, play_deal, step
from registry.store import make_agent

R = default_rules()


def _check_agent(spec, deals=4, sizes=(1, 3, 9)):
    ag = make_agent(spec)
    for i in range(deals):
        for k in sizes:
            s = new_deal(R, k, i % 4, 1000 * i + k, dtype="dark" if (k == 9 and i % 2) else "normal")
            agents = [ag, make_agent("heuristic"), make_agent("heuristic2"), make_agent("random", seed=i)]
            while not s.done:
                p = s.cur
                legal = legal_actions(s)
                a = agents[p].act(Obs(s, p), legal) if len(legal) > 1 else legal[0]
                assert a in legal, (spec, a, legal)
                step(s, a)
            assert sum(s.won) == k


def test_heuristics_legal():
    _check_agent("heuristic", 10)
    _check_agent("heuristic2", 10)


def test_pimc_legal():
    _check_agent("pimc:samples=4", 2)
    _check_agent('pimc:samples=4,rollout="h2"', 1)


def test_wrappers_legal():
    for m in ["shade,delta=-1", "joker_hold", "joker_asap", "joker_never_low", "forced_bid,bid=2"]:
        _check_agent(f"wrap:base=heuristic2,mode={m}", 2)


def test_sampler_respects_voids():
    from agents.pimc import PIMCAgent
    ag = PIMCAgent(samples=20)
    rng = random.Random(0)
    for seed in range(30):
        s = new_deal(R, 6, 0, seed)
        while s.phase == 0:
            step(s, legal_actions(s)[0])
        for _ in range(rng.randrange(1, 12)):
            step(s, rng.choice(legal_actions(s)))
            if s.done:
                break
        if s.done:
            continue
        o = Obs(s, s.cur)
        for h in ag.sample_worlds(o, 10):
            assert h[o.me] == s.hands[o.me]
            for p in range(4):
                assert h[p].bit_count() == s.hands[p].bit_count()
                for su in range(4):
                    if s.voids[p] >> su & 1:
                        assert h[p] & (((1 << 9) - 1) << (9 * su)) & ~R.joker_mask == 0
            u = 0
            for p in range(4):
                assert u & h[p] == 0
                u |= h[p]
            assert u & s.played == 0
