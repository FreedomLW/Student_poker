"""Rules config: loads rules.yaml into a frozen dataclass with a stable hash."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RULES_PATH = os.path.join(ROOT, "rules.yaml")

SUITS = "SCDH"  # spades, clubs, diamonds, hearts
SUIT_SYM = "♠♣♦♥"
RANKS = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
NR = 9  # ranks per suit
NC = 36


def card_from_str(s: str) -> int:
    s = s.strip().upper()
    suit = SUITS.index(s[-1])
    rank = RANKS.index(s[:-1])
    return suit * NR + rank


def card_str(c: int, unicode: bool = True) -> str:
    return RANKS[c % NR] + (SUIT_SYM if unicode else SUITS)[c // NR]


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


@dataclass(frozen=True)
class Rules:
    raw: dict = field(compare=False, hash=False, repr=False)
    n_players: int
    jokers: tuple
    max_cards: int
    schedule: tuple  # tuple of (n_cards, deal_type) ; deal_type in normal/notrump/dark
    trump_joker_turned: str
    trump_no_cards_left: str
    trump_mode: str          # "fixed" (always the same suit) | "turned" (next card after the deal)
    trump_fixed_suit: int
    joker_follow_low: str    # "as_card" (plays as the ordinary card, e.g. 7S) | "discard" (never wins)
    dealer_hook: bool
    dark_sees_trump: bool
    must_follow: bool
    must_trump: bool
    joker_any_time: bool
    joker_lead_low: bool
    joker_lead_demand_highest: bool
    exact_per_trick: int
    pass_exact: int
    over_per_trick: int
    under_per_trick: int
    under_mode: str
    dark_multiplier: int
    hash: str

    @property
    def joker_mask(self) -> int:
        m = 0
        for j in self.jokers:
            m |= 1 << j
        return m

    def score(self, bid: int, tricks: int, dark: bool = False) -> int:
        if tricks == bid:
            s = self.pass_exact if bid == 0 else self.exact_per_trick * bid
        elif tricks > bid:
            s = self.over_per_trick * (tricks - bid)
        else:
            s = self.under_per_trick * (bid if self.under_mode == "bid" else bid - tricks)
        return s * (self.dark_multiplier if dark else 1)


def build_schedule(n: int, sched: dict) -> tuple:
    mx = NC // n
    out = []
    if sched.get("ascending", True):
        out += [(k, "normal") for k in range(1, mx)]
    reps = sched.get("max_repeats")
    reps = n if reps is None else reps
    out += [(mx, "normal")] * reps
    if sched.get("descending", True):
        out += [(k, "normal") for k in range(mx - 1, 0, -1)]
    for sp in sched.get("special", []) or []:
        cnt = sp.get("count")
        cnt = n if cnt is None else cnt
        nc = sp.get("n_cards") or mx
        out += [(nc, sp["type"])] * cnt
    return tuple(out)


def load_rules(path: str | None = None, overrides: dict | None = None) -> Rules:
    with open(path or DEFAULT_RULES_PATH) as f:
        raw = yaml.safe_load(f)
    if overrides:
        raw = _deep_update(copy.deepcopy(raw), overrides)
    return rules_from_dict(raw)


def rules_from_dict(raw: dict) -> Rules:
    n = int(raw["n_players"])
    assert 3 <= n <= 6, "n_players must be 3..6"
    jokers = tuple(card_from_str(j) for j in raw["deck"].get("jokers", []) or [])
    assert len(jokers) <= 2
    p, b, t, s = raw["play"], raw["bidding"], raw["trump"], raw["scoring"]
    h = hashlib.sha1(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return Rules(
        raw=raw,
        n_players=n,
        jokers=jokers,
        max_cards=NC // n,
        schedule=build_schedule(n, raw.get("schedule", {})),
        trump_joker_turned=t.get("joker_turned", "notrump"),
        trump_no_cards_left=t.get("no_cards_left", "notrump"),
        trump_mode=t.get("mode", "turned"),
        trump_fixed_suit=SUITS.index(str(t.get("fixed_suit", "D")).upper()),
        joker_follow_low=p.get("joker_follow_low", "discard"),
        dealer_hook=bool(b.get("dealer_hook", True)),
        dark_sees_trump=bool(b.get("dark_sees_trump", True)),
        must_follow=bool(p.get("must_follow", True)),
        must_trump=bool(p.get("must_trump", True)),
        joker_any_time=bool(p.get("joker_any_time", True)),
        joker_lead_low=bool(p.get("joker_lead_low", True)),
        joker_lead_demand_highest=bool(p.get("joker_lead_demand_highest", True)),
        exact_per_trick=int(s.get("exact_per_trick", 10)),
        pass_exact=int(s.get("pass_exact", 5)),
        over_per_trick=int(s.get("over_per_trick", 1)),
        under_per_trick=int(s.get("under_per_trick", -10)),
        under_mode=s.get("under_mode", "bid"),
        dark_multiplier=int(s.get("dark_multiplier", 2)),
        hash=h,
    )


_DEFAULT = None


def default_rules() -> Rules:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_rules()
    return _DEFAULT
