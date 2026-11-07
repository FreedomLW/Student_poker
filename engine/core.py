"""Pure, deterministic engine for Odessa poker (одесский покер).

Cards: idx = suit*9 + rank, rank 0='6' .. 8='A'; suits S,C,D,H.
Hands are int bitmasks.

Action ids (shared by all phases):
  0..35        play a (non-joker) card
  36           joker, follow, HIGH (beats everything)
  37           joker, follow, LOW  (discard, never wins)
  38+s         joker, lead, HIGH, calling suit s
  42+s         joker, lead, LOW,  calling suit s (acts as lowest card of suit s)
  46+b         bid b (0..12)
"""
from __future__ import annotations

import random

from .rules import NC, NR, Rules, card_str, SUIT_SYM

A_J_FOLLOW_HIGH = 36
A_J_FOLLOW_LOW = 37
A_J_LEAD_HIGH = 38
A_J_LEAD_LOW = 42
BID_OFFSET = 46
MAX_BID = 12
N_ACTIONS = BID_OFFSET + MAX_BID + 1  # 59

PH_BID, PH_PLAY, PH_DONE = 0, 1, 2
K_NORMAL, K_JHIGH, K_JLOW = 0, 1, 2

SUIT_MASK = [((1 << NR) - 1) << (NR * s) for s in range(4)]
FULL = (1 << NC) - 1


def bit_list(m: int) -> list:
    out = []
    while m:
        lb = m & -m
        out.append(lb.bit_length() - 1)
        m ^= lb
    return out


def action_str(a: int) -> str:
    if a < 36:
        return card_str(a)
    if a == A_J_FOLLOW_HIGH:
        return "JOKER-high"
    if a == A_J_FOLLOW_LOW:
        return "JOKER-low"
    if a < A_J_LEAD_LOW:
        return f"JOKER-lead-high({SUIT_SYM[a - A_J_LEAD_HIGH]})"
    if a < BID_OFFSET:
        return f"JOKER-lead-low({SUIT_SYM[a - A_J_LEAD_LOW]})"
    return f"bid {a - BID_OFFSET}"


def is_joker_action(a: int) -> bool:
    return 36 <= a < BID_OFFSET


class State:
    __slots__ = (
        "rules", "n", "dealer", "n_cards", "dtype", "dark", "hands", "trump", "trump_card",
        "bids", "n_bids", "phase", "cur", "t_players", "t_cards", "t_kinds", "lead_suit",
        "demand", "leader", "won", "played", "tricks_done", "history", "voids", "first_seat",
        "init_hands", "bid_history",
    )

    def clone(self) -> "State":
        s = State.__new__(State)
        s.rules = self.rules
        s.n = self.n
        s.dealer = self.dealer
        s.n_cards = self.n_cards
        s.dtype = self.dtype
        s.dark = self.dark
        s.hands = self.hands[:]
        s.trump = self.trump
        s.trump_card = self.trump_card
        s.bids = self.bids[:]
        s.n_bids = self.n_bids
        s.phase = self.phase
        s.cur = self.cur
        s.t_players = self.t_players[:]
        s.t_cards = self.t_cards[:]
        s.t_kinds = self.t_kinds[:]
        s.lead_suit = self.lead_suit
        s.demand = self.demand
        s.leader = self.leader
        s.won = self.won[:]
        s.played = self.played
        s.tricks_done = self.tricks_done
        s.history = self.history[:]  # tricks are immutable tuples
        s.voids = self.voids[:]
        s.first_seat = self.first_seat
        s.init_hands = self.init_hands
        s.bid_history = self.bid_history[:]
        return s

    # ------------------------------------------------------------------ info
    @property
    def done(self) -> bool:
        return self.phase == PH_DONE

    def jokers_in(self, hand: int) -> int:
        return hand & self.rules.joker_mask

    def __repr__(self):
        hs = " | ".join(
            f"P{p}:" + ",".join(card_str(c) for c in bit_list(self.hands[p])) for p in range(self.n)
        )
        tr = SUIT_SYM[self.trump] if self.trump >= 0 else "NT"
        return (f"<State n_cards={self.n_cards} {self.dtype} trump={tr} dealer={self.dealer} "
                f"phase={self.phase} cur={self.cur} bids={self.bids} won={self.won} {hs}>")


def new_deal(rules: Rules, n_cards: int, dealer: int, rng: random.Random | int | None = None,
             dtype: str = "normal", hands: list | None = None, trump_card: int | None = None) -> State:
    """Create a deal. Either random (rng/seed) or fully specified (hands + trump_card(-1=none))."""
    n = rules.n_players
    assert 1 <= n_cards <= rules.max_cards
    s = State.__new__(State)
    s.rules = rules
    s.n = n
    s.dealer = dealer
    s.n_cards = n_cards
    s.dtype = dtype
    s.dark = dtype == "dark"
    if hands is None:
        if not isinstance(rng, random.Random):
            rng = random.Random(rng)
        deck = list(range(NC))
        rng.shuffle(deck)
        hands = [0] * n
        k = 0
        for i in range(n_cards):  # deal round-robin from left of dealer
            for j in range(n):
                p = (dealer + 1 + j) % n
                hands[p] |= 1 << deck[k]
                k += 1
        last_dealer_card = deck[k - 1]
        tc = deck[k] if k < NC else -1
    else:
        hands = list(hands)
        tc = -1 if trump_card is None else trump_card
        last_dealer_card = -1
    s.hands = hands
    s.init_hands = tuple(hands)
    jm = rules.joker_mask
    trump = -1
    if tc >= 0:
        if (1 << tc) & jm:
            trump = -1 if rules.trump_joker_turned == "notrump" else tc // NR
        else:
            trump = tc // NR
    elif rules.trump_no_cards_left == "dealer_last" and last_dealer_card >= 0:
        if not ((1 << last_dealer_card) & jm):
            trump = last_dealer_card // NR
    if rules.trump_mode == "fixed":
        trump = rules.trump_fixed_suit
        tc = -1  # no card is turned up
    if dtype == "notrump":
        trump = -1
    s.trump = trump
    s.trump_card = tc
    s.bids = [-1] * n
    s.n_bids = 0
    s.phase = PH_BID
    s.first_seat = (dealer + 1) % n
    s.cur = s.first_seat
    s.t_players = []
    s.t_cards = []
    s.t_kinds = []
    s.lead_suit = -1
    s.demand = False
    s.leader = s.first_seat
    s.won = [0] * n
    s.played = 0
    s.tricks_done = 0
    s.history = []
    s.voids = [0] * n  # bit s set => player publicly known void in suit s (bit 4 => void in trump)
    s.bid_history = []
    return s


# ---------------------------------------------------------------------- legal
def legal_bids(s: State) -> list:
    out = list(range(s.n_cards + 1))
    if s.rules.dealer_hook and s.n_bids == s.n - 1:
        forbidden = s.n_cards - sum(b for b in s.bids if b >= 0)
        if 0 <= forbidden <= s.n_cards:
            out.remove(forbidden)
    return [BID_OFFSET + b for b in out]


def legal_actions(s: State) -> list:
    if s.phase == PH_BID:
        return legal_bids(s)
    if s.phase == PH_DONE:
        return []
    r = s.rules
    hand = s.hands[s.cur]
    jm = r.joker_mask
    has_j = hand & jm
    plain = hand & ~jm
    if not s.t_players:  # leading
        acts = bit_list(plain)
        if has_j:
            acts += [A_J_LEAD_HIGH + x for x in range(4)]
            if r.joker_lead_low:
                acts += [A_J_LEAD_LOW + x for x in range(4)]
        return acts
    L = s.lead_suit
    inL = plain & SUIT_MASK[L]
    if inL and r.must_follow:
        if s.demand:
            acts = [inL.bit_length() - 1]  # must give highest card of called suit
        else:
            acts = bit_list(inL)
        can_j = r.joker_any_time
    else:
        can_j = True
        tm = plain & SUIT_MASK[s.trump] if (s.trump >= 0 and r.must_trump) else 0
        if tm and r.must_follow:
            acts = bit_list(tm)
        else:
            acts = bit_list(plain)
    if has_j and can_j:
        acts += [A_J_FOLLOW_HIGH, A_J_FOLLOW_LOW]
    return acts


# ----------------------------------------------------------------------- step
def _trick_winner(s: State) -> int:
    best_i, best_k = 0, -1
    T, L = s.trump, s.lead_suit
    for i in range(len(s.t_players)):
        kind = s.t_kinds[i]
        if kind == K_JHIGH:
            k = 1000 + i
        elif kind == K_JLOW:
            if i == 0:  # led low joker = lowest card of called suit
                k = 100 if L == T else 10
            else:
                k = 0
        else:
            c = s.t_cards[i]
            su = c // NR
            if su == T:
                k = 101 + c % NR
            elif su == L:
                k = 11 + c % NR
            else:
                k = 0
        if k > best_k:
            best_k, best_i = k, i
    return s.t_players[best_i]


def step(s: State, a: int) -> State:
    """Apply action in place (returns s). Caller must pass a legal action."""
    p = s.cur
    n = s.n
    if s.phase == PH_BID:
        b = a - BID_OFFSET
        s.bids[p] = b
        s.bid_history.append((p, b))
        s.n_bids += 1
        if s.n_bids == n:
            s.phase = PH_PLAY
            s.cur = s.leader
        else:
            s.cur = (p + 1) % n
        return s
    r = s.rules
    leading = not s.t_players
    if a < 36:
        c = a
        kind = K_NORMAL
        if leading:
            s.lead_suit = c // NR
            s.demand = False
        else:
            su = c // NR
            if su != s.lead_suit:  # failed to follow -> void in lead suit
                s.voids[p] |= 1 << s.lead_suit
                if s.trump >= 0 and su != s.trump:
                    s.voids[p] |= 1 << s.trump
            elif s.demand:
                pass
    else:
        jm = s.hands[p] & r.joker_mask
        c = (jm & -jm).bit_length() - 1
        if a == A_J_FOLLOW_HIGH:
            kind = K_JHIGH
        elif a == A_J_FOLLOW_LOW:
            # 'as_card': the joker is played as the ordinary card it is printed as (e.g. 7S) and can win normally
            kind = K_NORMAL if r.joker_follow_low == "as_card" else K_JLOW
        elif a < A_J_LEAD_LOW:
            kind = K_JHIGH
            s.lead_suit = a - A_J_LEAD_HIGH
            s.demand = r.joker_lead_demand_highest
        else:
            kind = K_JLOW
            s.lead_suit = a - A_J_LEAD_LOW
            s.demand = False
    s.hands[p] &= ~(1 << c)
    s.played |= 1 << c
    s.t_players.append(p)
    s.t_cards.append(c)
    s.t_kinds.append(kind)
    if len(s.t_players) < n:
        s.cur = (p + 1) % n
        return s
    w = _trick_winner(s)
    s.won[w] += 1
    s.history.append((s.leader, tuple(s.t_cards), tuple(s.t_kinds), s.lead_suit, w))
    s.tricks_done += 1
    s.t_players = []
    s.t_cards = []
    s.t_kinds = []
    s.lead_suit = -1
    s.demand = False
    s.leader = w
    s.cur = w
    if s.tricks_done == s.n_cards:
        s.phase = PH_DONE
    return s


def scores(s: State) -> list:
    r = s.rules
    return [r.score(s.bids[p], s.won[p], s.dark) for p in range(s.n)]


# -------------------------------------------------------------- full game
def game_deals(rules: Rules, seed: int, first_dealer: int = 0):
    """Yield (n_cards, dtype, dealer, deal_seed) for a full game."""
    rng = random.Random(seed)
    for i, (k, t) in enumerate(rules.schedule):
        yield k, t, (first_dealer + i) % rules.n_players, rng.getrandbits(63)
