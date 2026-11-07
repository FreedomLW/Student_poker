"""Play Odessa poker in the terminal against any registry agents.

python -m play.cli --opponents heuristic pimc:samples=24 store:dmc_v1 [--seat 0] [--seed 1] [--deal 5]
--deal K plays a single K-card deal instead of the full game. Type 'h' for help at a prompt.
"""
from __future__ import annotations

import argparse
import random

from engine import (BID_OFFSET, NR, Obs, SUIT_SYM, action_str, bit_list, card_str, default_rules, game_deals,
                    legal_actions, new_deal, scores, step)
from registry.store import make_agent


def show_hand(h):
    by = {}
    for c in bit_list(h):
        by.setdefault(c // NR, []).append(card_str(c))
    return "  ".join(" ".join(v) for _, v in sorted(by.items()))


class Human:
    name = "you"

    def new_deal(self):
        pass

    def act(self, o, legal):
        tr = SUIT_SYM[o.trump] if o.trump >= 0 else "без козыря"
        print(f"\n--- {o.n_cards} карт, {o.dtype}, козырь: {tr}, сдаёт P{o.dealer}")
        print("Заказы:", " ".join(f"P{p}={b if b >= 0 else '?'}" for p, b in enumerate(o.bids)),
              "| Взятки:", " ".join(f"P{p}={w}" for p, w in enumerate(o.won)))
        if o.t_players:
            print("На столе:", ", ".join(
                f"P{p}:{'JOKER' if (1 << c) & o.rules.joker_mask else card_str(c)}"
                f"{'(ст)' if k == 1 else '(мл)' if k == 2 else ''}" for p, c, k in zip(o.t_players, o.t_cards, o.t_kinds)))
        print("Ваша рука:", show_hand(o.hand) if o.hand else "(тёмная — карты скрыты)")
        for i, a in enumerate(legal):
            print(f"  [{i}] {action_str(a)}")
        while True:
            x = input("Ваш ход (номер): ").strip()
            if x == "h":
                print("Введите номер варианта. Джокер: high=старший, low=младший, lead-high(♠)=заход «по старшим пикам».")
                continue
            if x.isdigit() and int(x) < len(legal):
                return legal[int(x)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponents", nargs="+", default=["heuristic"])
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=random.randrange(10**6))
    ap.add_argument("--deal", type=int, default=0)
    a = ap.parse_args(argv)
    R = default_rules()
    n = R.n_players
    opp = (a.opponents * n)[: n - 1]
    agents = []
    j = 0
    for p in range(n):
        if p == a.seat:
            agents.append(Human())
        else:
            agents.append(make_agent(opp[j], seed=a.seed + p))
            j += 1
    deals = [(a.deal, "normal", 0, a.seed)] if a.deal else list(game_deals(R, a.seed))
    totals = [0] * n
    for k, t, dealer, ds in deals:
        s = new_deal(R, k, dealer, ds, dtype=t)
        for ag in agents:
            ag.new_deal()
        while not s.done:
            p = s.cur
            legal = legal_actions(s)
            act = agents[p].act(Obs(s, p), legal) if len(legal) > 1 else legal[0]
            if p != a.seat:
                print(f"P{p} ({agents[p].name}): {action_str(act)}")
            n_before = s.tricks_done
            step(s, act)
            if s.tricks_done > n_before:
                ld, cards, kinds, ls, w = s.history[-1]
                print(f"   взятку берёт P{w}")
        sc = scores(s)
        for p in range(n):
            totals[p] += sc[p]
        print("\n== Итог раздачи:", " ".join(f"P{p}: заказ {s.bids[p]}, взял {s.won[p]}, {sc[p]:+d}" for p in range(n)))
        print("== Счёт:", totals)


if __name__ == "__main__":
    main()
