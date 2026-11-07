"""Player observation (information set view) and reconstruction of a State from it."""
from __future__ import annotations

from .core import PH_BID, State


class Obs:
    """Everything player `me` legitimately knows. No other players' hands."""
    __slots__ = (
        "rules", "me", "n", "dealer", "n_cards", "dtype", "dark", "hand", "trump", "trump_card",
        "bids", "bid_history", "phase", "cur", "t_players", "t_cards", "t_kinds", "lead_suit",
        "demand", "leader", "won", "played", "tricks_done", "history", "voids", "first_seat",
        "hand_sizes",
    )

    def __init__(self, s: State, me: int):
        self.rules = s.rules
        self.me = me
        self.n = s.n
        self.dealer = s.dealer
        self.n_cards = s.n_cards
        self.dtype = s.dtype
        self.dark = s.dark
        hidden = s.dark and s.phase == PH_BID
        self.hand = 0 if hidden else s.hands[me]
        self.trump = s.trump if (not hidden or s.rules.dark_sees_trump) else -2  # -2 = unknown
        self.trump_card = s.trump_card
        self.bids = s.bids[:]
        self.bid_history = s.bid_history[:]
        self.phase = s.phase
        self.cur = s.cur
        self.t_players = s.t_players[:]
        self.t_cards = s.t_cards[:]
        self.t_kinds = s.t_kinds[:]
        self.lead_suit = s.lead_suit
        self.demand = s.demand
        self.leader = s.leader
        self.won = s.won[:]
        self.played = s.played
        self.tricks_done = s.tricks_done
        self.history = s.history[:]
        self.voids = s.voids[:]
        self.first_seat = s.first_seat
        self.hand_sizes = [h.bit_count() for h in s.hands]

    def seat(self, p: int) -> int:
        """Relative seat of p w.r.t. the first bidder (0 = first bidder/leader, n-1 = dealer)."""
        return (p - self.first_seat) % self.n

    @property
    def my_seat(self) -> int:
        return self.seat(self.me)


def state_from_obs(o: Obs, hands: list) -> State:
    """Build a full State consistent with obs, given a hypothesis about all current hands."""
    s = State.__new__(State)
    s.rules = o.rules
    s.n = o.n
    s.dealer = o.dealer
    s.n_cards = o.n_cards
    s.dtype = o.dtype
    s.dark = o.dark
    s.hands = list(hands)
    s.init_hands = None
    s.trump = o.trump if o.trump != -2 else -1
    s.trump_card = o.trump_card
    s.bids = o.bids[:]
    s.n_bids = sum(1 for b in o.bids if b >= 0)
    s.phase = o.phase
    s.cur = o.cur
    s.t_players = o.t_players[:]
    s.t_cards = o.t_cards[:]
    s.t_kinds = o.t_kinds[:]
    s.lead_suit = o.lead_suit
    s.demand = o.demand
    s.leader = o.leader
    s.won = o.won[:]
    s.played = o.played
    s.tricks_done = o.tricks_done
    s.history = o.history[:]
    s.voids = o.voids[:]
    s.first_seat = o.first_seat
    s.bid_history = o.bid_history[:]
    return s
