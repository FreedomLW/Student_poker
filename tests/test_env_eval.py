import numpy as np

from engine import default_rules, legal_actions
from env.aec import OdessaAEC
from env.features import obs_dim
from eval.harness import bootstrap_ci, bradley_terry_elo, vs_field
from registry.store import make_agent


def test_aec_game_runs():
    env = OdessaAEC(mode="game")
    env.reset(3)
    rng = np.random.default_rng(0)
    steps = 0
    while not env.done:
        ob = env.observe(env.agent_selection)
        assert ob["observation"].shape == (obs_dim(4, 9),)
        legal = np.flatnonzero(ob["action_mask"])
        assert len(legal) > 0
        assert set(legal) == set(legal_actions(env.state))
        env.step(int(rng.choice(legal)))
        steps += 1
    assert steps > 100 and len(env.cumulative) == 4


def test_registry_specs():
    for sp in ["random", "heuristic", "heuristic:bid_noise=0.5"]:
        ag = make_agent(sp)
        assert ag.name == sp


def test_eval_deterministic_and_bt():
    d1 = vs_field("random", "heuristic", 4, procs=1)
    d2 = vs_field("random", "heuristic", 4, procs=2)
    assert np.allclose(d1, d2)
    m, lo, hi = bootstrap_ci(d1)
    assert lo <= m <= hi
    elo = bradley_terry_elo(["a", "b"], [(0, 1, 1.0)] * 30 + [(0, 1, 0.0)] * 10, anchor="b")
    assert elo["a"] > elo["b"] == 1500
