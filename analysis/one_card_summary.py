"""Readable summary of the 1-card equilibrium (runs/one_card/sigma.npy) -> markdown."""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

from engine import NR, default_rules

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = default_rules()
J = R.jokers[0]
RN = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']


def hid(h):
    return (1 << len(h)) - 1 + int("".join(map(str, h)) or "0", 2)


def classes(t):
    T = -1 if t == J else t // NR
    out = {}
    for c in range(36):
        if c == t:
            continue
        if c == J:
            k = "джокер"
        elif c // NR == T:
            k = f"козырь {RN[c % NR]}"
        else:
            r = c % NR
            k = f"некозырь {RN[r]}" if r >= 6 else "некозырь ≤J"
        out.setdefault(k, []).append(c)
    return out


def main_fixed():
    sig = np.load(os.path.join(ROOT, "runs", "one_card", "sigma_fixed.npy"))
    hist = json.load(open(os.path.join(ROOT, "runs", "one_card", "history_fixed.json")))
    last = hist[-1]
    T = 2
    cols = [(0, ()), (1, (0,)), (1, (1,)), (2, (0, 0)), (2, (0, 1)), (2, (1, 0)), (2, (1, 1)), (3, (1, 1, 0)), (3, (1, 1, 1))]
    names = ["место 1", "место 2 после 0", "место 2 после 1", "место 3 после 0,0", "место 3 после 0,1",
             "место 3 после 1,0", "место 3 после 1,1", "сдающий после 1,1,0", "сдающий после 1,1,1"]
    print(f"EV/место: {np.round(last['ev'], 2).tolist()} ; exploitability: {np.round(last['exploit'], 4).tolist()}\n")
    print("| карта | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    rows = [("джокер (7♠)", [J])]
    for r in reversed(range(9)):
        rows.append((f"козырь ♦{RN[r]}", [T * NR + r]))
    for r, lab in ((8, "A"), (7, "K"), (6, "Q")):
        rows.append((f"некозырь {lab}", [su * NR + r for su in range(4) if su != T]))
    rows.append(("некозырь ≤J", [su * NR + r for su in range(4) if su != T for r in range(6) if su * NR + r != J]))
    for lab, cs in rows:
        print(f"| {lab} | " + " | ".join(f"{np.mean(sig[seat, cs, 0, hid(h)]):.2f}" for seat, h in cols) + " |")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "fixed":
        return main_fixed()
    sig = np.load(os.path.join(ROOT, "runs", "one_card", "sigma.npy"))
    hist = json.load(open(os.path.join(ROOT, "runs", "one_card", "history.json")))
    last = hist[-1]
    cols = [(0, ()), (1, (0,)), (1, (1,)), (2, (0, 0)), (2, (0, 1)), (2, (1, 0)), (2, (1, 1)), (3, (1, 1, 0)), (3, (1, 1, 1))]
    names = ["место 1 (ходит первым)", "место 2 после 0", "место 2 после 1", "место 3 после 0,0", "место 3 после 0,1",
             "место 3 после 1,0", "место 3 после 1,1", "сдающий после 1,1,0", "сдающий после 1,1,1"]
    order = ["джокер"] + [f"козырь {r}" for r in reversed(RN)] + ["некозырь A", "некозырь K", "некозырь Q", "некозырь ≤J"]
    print(f"EV/место (очки за 1-карточную раздачу): {np.round(last['ev'], 2).tolist()}  ; "
          f"exploitability (выигрыш от best response, очки/раздачу): {np.round(last['exploit'], 4).tolist()}\n")
    for trump_kind in ("козырная раздача",):
        print(f"### P(заказ 1) в эквилибриуме — {trump_kind} (усреднено по картам класса и открытым козырям)\n")
        print("| карта | " + " | ".join(names) + " |")
        print("|---" * (len(names) + 1) + "|")
        acc = {k: [[] for _ in cols] for k in order}
        for t in range(36):
            if t == J:
                continue
            cl = classes(t)
            for k, cs in cl.items():
                for j, (seat, h) in enumerate(cols):
                    acc[k][j].extend(sig[seat, cs, t, hid(h)].tolist())
        for k in order:
            if not acc[k][0]:
                continue
            print(f"| {k} | " + " | ".join(f"{np.mean(v):.2f}" for v in acc[k]) + " |")
    print("\n### Бескозырка (открыт джокер) — P(заказ 1)\n")
    print("| карта | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    cl = classes(J)
    for k in ["некозырь A", "некозырь K", "некозырь Q", "некозырь ≤J"]:
        print(f"| {k} | " + " | ".join(f"{np.mean(sig[seat, cl[k], J, hid(h)]):.2f}" for seat, h in cols) + " |")


if __name__ == "__main__":
    main()
