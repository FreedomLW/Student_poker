import random

import pytest

from agents import HeuristicAgent, RandomAgent
from engine import (A_J_FOLLOW_HIGH, A_J_FOLLOW_LOW, A_J_LEAD_HIGH, A_J_LEAD_LOW, BID_OFFSET, NC,
                    Obs, card_from_str as C, default_rules, legal_actions, load_rules, new_deal,
                    play_deal, play_game, scores, state_from_obs, step)

import os
CLASSIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rules_variants', 'classic.yaml')
R = load_rules(CLASSIC)
OURS = default_rules()


def H(*cards):
    m = 0
    for c in cards:
        m |= 1 << C(c)
    return m


def run_random(rules, k, seed, dtype="normal"):
    rng = random.Random(seed)
    s = new_deal(rules, k, rng.randrange(rules.n_players), rng, dtype=dtype)
    init = s.hands[:]
    acts = []
    while not s.done:
        legal = legal_actions(s)
        assert legal, "legal actions must be non-empty"
        assert len(set(legal)) == len(legal)
        a = rng.choice(legal)
        acts.append(a)
        step(s, a)
    return s, init, acts


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_properties_random_play(n):
    rules = load_rules(CLASSIC if n % 2 else None, overrides={"n_players": n})
    for seed in range(300):
        k = 1 + seed % rules.max_cards
        s, init, _ = run_random(rules, k, seed, dtype=["normal", "notrump", "dark"][seed % 3])
        # card conservation: every initially held card was played exactly once
        all_init = 0
        for h in init:
            assert h.bit_count() == k
            assert all_init & h == 0
            all_init |= h
        assert s.played == all_init
        assert all(h == 0 for h in s.hands)
        assert sum(s.won) == k
        assert len(s.history) == k
        if rules.hook_applies(k):
            assert sum(s.bids) != k
        assert len(scores(s)) == n


def test_determinism_by_seed():
    a = run_random(R, 7, 123)
    b = run_random(R, 7, 123)
    assert a[1] == b[1] and a[2] == b[2] and a[0].won == b[0].won
    t1, _ = play_game(R, [HeuristicAgent() for _ in range(4)], 42)
    t2, _ = play_game(R, [HeuristicAgent() for _ in range(4)], 42)
    assert t1 == t2


def test_schedule_default():
    sch = R.schedule
    ks = [k for k, _ in sch]
    assert ks[:9] == list(range(1, 10))
    assert ks.count(9) == 4 + 4 + 4  # 4 normal at max + 4 notrump + 4 dark
    assert [t for _, t in sch].count("dark") == 4


def test_dealer_hook():
    s = new_deal(R, 3, dealer=3, rng=1)
    for b in (1, 0, 1):
        step(s, BID_OFFSET + b)
    assert s.cur == 3
    legal = [a - BID_OFFSET for a in legal_actions(s)]
    assert 1 not in legal and legal == [0, 2, 3]
    r2 = load_rules(CLASSIC, overrides={"bidding": {"dealer_hook": False}})
    s = new_deal(r2, 3, dealer=3, rng=1)
    for b in (1, 0, 1):
        step(s, BID_OFFSET + b)
    assert [a - BID_OFFSET for a in legal_actions(s)] == [0, 1, 2, 3]


def make(hands, trump_card=-1, dealer=3, bids=(0, 0, 0, 1), rules=R):
    s = new_deal(rules, len(hands[0]), dealer, hands=[H(*h) for h in hands], trump_card=trump_card)
    for b in bids:
        step(s, BID_OFFSET + b)
    return s


def test_follow_suit_and_trump_obligation():
    # trump = hearts (trump card 6H)
    s = make([["AS", "6D"], ["KS", "7D"], ["9H", "8D"], ["8C", "9D"]], trump_card=C("6H"), bids=(1, 1, 1, 1))
    step(s, C("AS"))
    assert legal_actions(s) == [C("KS")]
    step(s, C("KS"))
    assert legal_actions(s) == [C("9H")]  # void in spades -> must trump
    step(s, C("9H"))
    assert set(legal_actions(s)) == {C("8C"), C("9D")}  # no spade, no trump -> any
    step(s, C("8C"))
    assert s.won == [0, 0, 1, 0]  # trump wins
    assert s.voids[2] & 1  # player 2 known void in spades
    assert s.voids[3] & (1 << 0) and s.voids[3] & (1 << 3)  # void spades and trump


def test_joker_any_time_high_low():
    # P0 leads AS, P1 holds KS + joker (7S) -> may play joker even holding spades
    s = make([["AS", "6D"], ["KS", "7S"], ["8S", "8D"], ["9S", "9D"]], bids=(1, 1, 0, 0))
    step(s, C("AS"))
    la = legal_actions(s)
    assert set(la) == {C("KS"), A_J_FOLLOW_HIGH, A_J_FOLLOW_LOW}
    step(s, A_J_FOLLOW_HIGH)
    step(s, C("8S"))
    step(s, C("9S"))
    assert s.won == [0, 1, 0, 0]
    # joker low never wins when following
    s = make([["AS", "6D"], ["KS", "7S"], ["8S", "8D"], ["9S", "9D"]], bids=(1, 1, 0, 0))
    step(s, C("AS"))
    step(s, A_J_FOLLOW_LOW)
    step(s, C("8S"))
    step(s, C("9S"))
    assert s.won == [1, 0, 0, 0]


def test_joker_lead_demand_highest():
    s = make([["7S", "6D"], ["KD", "7D"], ["8S", "8D"], ["9C", "9H"]], trump_card=C("6C"), bids=(1, 1, 0, 0))
    step(s, A_J_LEAD_HIGH + 2)  # demand highest diamonds
    assert legal_actions(s) == [C("KD")]
    step(s, C("KD"))
    assert legal_actions(s) == [C("8D")]
    step(s, C("8D"))
    assert legal_actions(s) == [C("9C")]  # no diamonds -> must trump (clubs)
    step(s, C("9C"))
    assert s.won == [1, 0, 0, 0]  # led high joker wins even vs trump


def test_joker_lead_low():
    # lead low calling diamonds: joker = lowest diamond; highest diamond wins
    s = make([["7S", "6S"], ["KD", "7D"], ["8S", "8D"], ["9C", "9H"]], bids=(1, 1, 0, 0))
    step(s, A_J_LEAD_LOW + 2)
    assert set(legal_actions(s)) == {C("KD"), C("7D")}  # normal follow (no demand)
    step(s, C("7D"))
    step(s, C("8D"))
    step(s, C("9C"))
    assert s.won == [0, 0, 1, 0]
    # nobody has the called suit and no trump -> low joker still takes the trick
    s = make([["7S", "6S"], ["KC", "7C"], ["8S", "8C"], ["9C", "9H"]], bids=(1, 1, 0, 0))
    step(s, A_J_LEAD_LOW + 2)
    for _ in range(3):
        step(s, legal_actions(s)[0])
    assert s.won == [1, 0, 0, 0]


def test_trump_turned_joker_is_notrump():
    s = new_deal(R, 2, 0, hands=[H("AS", "KS"), H("QS", "JS"), H("10S", "9S"), H("8S", "6S")],
                 trump_card=C("7S"))
    assert s.trump == -1
    s = new_deal(R, 2, 0, hands=[H("AS", "KS"), H("QS", "JS"), H("10S", "9S"), H("8S", "6S")],
                 trump_card=C("6H"))
    assert s.trump == 3


def test_full_deal_no_trump_when_no_cards_left():
    for seed in range(20):
        s = new_deal(R, 9, 0, seed)
        assert s.trump == -1 and s.trump_card == -1


def test_scoring():
    assert R.score(3, 3) == 30
    assert R.score(0, 0) == 5
    assert R.score(1, 3) == 2
    assert R.score(0, 2) == 2
    assert R.score(3, 1) == -30
    assert R.score(2, 2, dark=True) == 40
    assert R.score(2, 0, dark=True) == -40
    r2 = load_rules(CLASSIC, overrides={"scoring": {"under_mode": "missing"}})
    assert r2.score(3, 1) == -20


def test_obs_hides_other_hands_and_dark():
    s = new_deal(R, 9, 0, 5, dtype="dark")
    o = Obs(s, 1)
    assert o.hand == 0  # dark: hand hidden while bidding
    assert not hasattr(o, "hands")
    for _ in range(4):
        step(s, legal_actions(s)[0])
    assert Obs(s, 1).hand == s.hands[1]
    s2 = state_from_obs(Obs(s, 1), s.hands)
    assert legal_actions(s2) == legal_actions(s)


def test_agents_complete_games():
    for n in (3, 5, 6):
        rules = load_rules(CLASSIC if n % 2 else None, overrides={"n_players": n})
        t, log = play_game(rules, [RandomAgent(i) for i in range(n)], 7)
        assert len(log) == len(rules.schedule)
        t, log = play_game(rules, [HeuristicAgent() for _ in range(n)], 7)
        assert all(sum(d["won"]) == d["n_cards"] for d in log)


def test_variant_flags():
    r = load_rules(CLASSIC, overrides={"trump": {"no_cards_left": "dealer_last"}})
    s = new_deal(r, 9, 0, 3)
    assert s.trump >= 0 or (s.hands[0] & r.joker_mask)
    r = load_rules(overrides={"deck": {"jokers": []}})
    for seed in range(50):
        run_random(r, 5, seed)
    r = load_rules(overrides={"deck": {"jokers": ["7S", "6C"]}})
    for seed in range(100):
        run_random(r, 4, seed)
    r = load_rules(overrides={"play": {"joker_lead_demand_highest": False, "joker_lead_low": False}})
    for seed in range(50):
        run_random(r, 6, seed)


# ----------------------------------------------------------- our house rules (default rules.yaml)
def test_house_trump_always_diamonds():
    for seed in range(30):
        for k in (1, 5, 9):
            s = new_deal(OURS, k, seed % 4, seed)
            assert s.trump == 2 and s.trump_card == -1


def test_house_joker_lead_only_high():
    s = make([["7S", "6S"], ["KD", "7D"], ["8S", "8D"], ["9C", "9H"]], bids=(1, 1, 0, 0), rules=OURS)
    la = legal_actions(s)
    assert set(la) == {C("6S")} | {A_J_LEAD_HIGH + x for x in range(4)}


def test_house_joker_follow_as_7s_can_win():
    # spades led with 6S; P1 plays the joker "as 7S": it beats 6S if nobody trumps / plays higher spades
    s = make([["6S", "6C"], ["7S", "7C"], ["6H", "8C"], ["9C", "9H"]], bids=(0, 1, 0, 0), rules=OURS)
    step(s, C("6S"))
    assert set(legal_actions(s)) == {C("7C"), A_J_FOLLOW_HIGH, A_J_FOLLOW_LOW}
    step(s, A_J_FOLLOW_LOW)
    step(s, C("6H"))
    step(s, C("9C"))
    assert s.won == [0, 1, 0, 0]
    assert s.voids[1] == 0  # joker play never reveals a void
    # as 7S on a heart lead it cannot win
    s = make([["6H", "6C"], ["7S", "7C"], ["8H", "8C"], ["9C", "9H"]], bids=(0, 1, 0, 0), rules=OURS)
    step(s, C("6H"))
    step(s, A_J_FOLLOW_LOW)
    step(s, C("8H"))
    step(s, C("9H"))
    assert s.won == [0, 0, 0, 1]
    # as 7S it loses to the 8S
    s = make([["6S", "6C"], ["7S", "7C"], ["8S", "8C"], ["9C", "9H"]], bids=(0, 1, 0, 0), rules=OURS)
    step(s, C("6S"))
    step(s, A_J_FOLLOW_LOW)
    step(s, C("8S"))
    step(s, C("9C"))
    assert s.won == [0, 0, 1, 0]


def test_house_hook_only_from_4_cards():
    for k, restricted in ((1, False), (3, False), (4, True), (9, True)):
        s = new_deal(OURS, k, 3, 7)
        for b in (1, 0, 0):
            step(s, BID_OFFSET + b)
        bids = [a - BID_OFFSET for a in legal_actions(s)]
        assert ((k - 1) in bids) != restricted


def test_house_scoring_over_taken_and_dark_x1():
    assert OURS.score(2, 5) == 5 and OURS.score(0, 3) == 3 and OURS.score(3, 3) == 30
    assert OURS.score(4, 2) == -40 and OURS.score(2, 2, dark=True) == 20
    assert R.score(2, 5) == 3  # classic: +1 per extra trick
