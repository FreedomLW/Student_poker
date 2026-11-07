"""Insight tables from collected records (analysis.collect).

python -m analysis.tables runs/data/pimc_h2.npz [--ref runs/data/heur.npz] > runs/analysis/tables_pimc.md
"""
from __future__ import annotations

import argparse

import numpy as np

from .calibration import mci, ols
from .collect import load


def diff_ci(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = a.mean() - b.mean()
    h = 1.96 * np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return m, h


def joker_tables(d):
    out = ["### Ценность джокера: E[очки | джокер на руке] − E[очки | нет джокера] (джокер раздаётся случайно ⇒ это причинный эффект)", "",
           "| n_cards | тип | N с джокером | Δ очков | P(выполнил) с джокером | без джокера | Δ взяток |", "|---|---|---|---|---|---|---|"]
    for dt, name in ((0, "обычная"), (1, "бескозырка"), (2, "тёмная")):
        for k in sorted(set(d["n_cards"][d["dtype"] == dt])):
            m = (d["n_cards"] == k) & (d["dtype"] == dt)
            j = m & (d["joker"] == 1)
            nj = m & (d["joker"] == 0)
            if j.sum() < 30:
                continue
            dm, dh = diff_ci(d["score"][j], d["score"][nj])
            tm, th = diff_ci(d["tricks"][j], d["tricks"][nj])
            ej = np.mean(d["tricks"][j] == d["bid"][j])
            en = np.mean(d["tricks"][nj] == d["bid"][nj])
            out.append(f"| {k} | {name} | {j.sum()} | {dm:+.2f}±{dh:.2f} | {ej:.3f} | {en:.3f} | {tm:+.2f}±{th:.2f} |")
    out += ["", "### Как играется джокер (раздачи ≥2 карт)", "",
            "| n_cards | N | заход джокером | старшим | младшим | взял взятку | средний номер взятки (0=первая) | доля «в последней взятке» | заход: вызов козыря (из заходов старшим) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(set(d["n_cards"])):
        if k < 2:
            continue
        m = (d["n_cards"] == k) & (d["j_trick"] >= 0) & (d["dtype"] == 0)
        if m.sum() < 30:
            m = (d["n_cards"] == k) & (d["j_trick"] >= 0)
        if m.sum() < 30:
            continue
        lead = d["j_lead"][m]
        high = d["j_high"][m]
        lh = m & (d["j_lead"] == 1) & (d["j_high"] == 1)
        ct = np.mean(d["j_call_trump"][lh]) if lh.sum() > 10 and (d["trump"][lh] >= 0).any() else float("nan")
        out.append(f"| {k} | {m.sum()} | {lead.mean():.2f} | {high.mean():.2f} | {1 - high.mean():.2f} | "
                   f"{d['j_won'][m].mean():.2f} | {d['j_trick'][m].mean():.2f} | {np.mean(d['j_trick'][m] == k - 1):.2f} | {ct:.2f} |")
    # joker low vs need: low joker means the holder wanted to lose
    return "\n".join(out)


def seat_table(d, label=""):
    out = [f"### EV по местам {label} (очки за раздачу, ±95% CI); место 1 = первый заказывает и ходит, 4 = сдающий", "",
           "| n_cards | тип | место 1 | место 2 | место 3 | сдающий | сдающий − среднее остальных |", "|---|---|---|---|---|---|---|"]
    for dt, name in ((0, "обычная"), (1, "бескозырка"), (2, "тёмная")):
        for k in sorted(set(d["n_cards"][d["dtype"] == dt])):
            m = (d["n_cards"] == k) & (d["dtype"] == dt)
            cells = []
            for s in range(4):
                cells.append("%.2f±%.2f" % mci(d["score"][m & (d["seat"] == s)]))
            # paired difference per deal: dealer - mean(others)
            seeds = d["seed"][m]
            sc = d["score"][m]
            st = d["seat"][m]
            order = np.lexsort((st, seeds))
            sc4 = sc[order].reshape(-1, 4)
            dd = sc4[:, 3] - sc4[:, :3].mean(axis=1)
            out.append(f"| {k} | {name} | " + " | ".join(cells) + " | %+.2f±%.2f |" % mci(dd))
    return "\n".join(out)


def bid_behaviour(d, ref=None, ref_name="heuristic"):
    out = ["### Поведение заказа: частота паса, средний заказ − средние взятки, доля переборов среди промахов", "",
           "| n_cards | P(пас) | P(пас выполнен) | E[взятки − заказ] | промахи: доля переборов | E[очки] |" +
           (f" P(пас) {ref_name} | E[взятки−заказ] {ref_name} | E[очки] {ref_name} |" if ref is not None else ""),
           "|---|---|---|---|---|---|" + ("---|---|---|" if ref is not None else "")]
    for k in sorted(set(d["n_cards"][d["dtype"] == 0])):
        m = (d["n_cards"] == k) & (d["dtype"] == 0)
        z = m & (d["bid"] == 0)
        miss = m & (d["tricks"] != d["bid"])
        row = (f"| {k} | {np.mean(d['bid'][m] == 0):.2f} | {np.mean(d['tricks'][z] == 0):.2f} | "
               f"{np.mean(d['tricks'][m] - d['bid'][m]):+.2f} | {np.mean(d['tricks'][miss] > d['bid'][miss]):.2f} | "
               f"{d['score'][m].mean():.2f} |")
        if ref is not None:
            mr = (ref["n_cards"] == k) & (ref["dtype"] == 0)
            row += (f" {np.mean(ref['bid'][mr] == 0):.2f} | {np.mean(ref['tricks'][mr] - ref['bid'][mr]):+.2f} | "
                    f"{ref['score'][mr].mean():.2f} |")
        out.append(row)
    return "\n".join(out)


def prev_bids_effect(d):
    """T15: bid ~ hand features + previous bids excess (prev_sum - seat*k/n), per size bucket."""
    out = ["### Влияние заказов предыдущих игроков на заказ (OLS: заказ ~ признаки руки + «избыток» предыдущих заказов)", "",
           "| n_cards | коэф. при избытке пред. заказов | ±95% | коэф. при избытке в модели ВЗЯТОК | ±95% | N |", "|---|---|---|---|---|---|"]
    feats = ["joker", "trump_A", "trump_K", "trump_Q", "n_trump", "aces_side", "kings_side", "voids"]
    for k in sorted(set(d["n_cards"][d["dtype"] == 0])):
        m = (d["n_cards"] == k) & (d["dtype"] == 0) & (d["seat"] > 0) & (d["trump"] >= 0)
        if m.sum() < 300:
            continue
        dd = dict(d)
        dd["excess"] = d["prev_bid_sum"] - d["seat"] * k / 4.0
        b, se, _ = ols(dd, m, feats + ["excess"], target="bid")
        b2, se2, _ = ols(dd, m, feats + ["excess"], target="tricks")
        out.append(f"| {k} | {b[-1]:+.3f} | {1.96 * se[-1]:.3f} | {b2[-1]:+.3f} | {1.96 * se2[-1]:.3f} | {m.sum()} |")
    return "\n".join(out)


def dark_table(d):
    out = ["### Тёмная (9 карт): распределение и EV заказов по местам", "",
           "| место | заказ | доля | E[взятки] | P(выполнил) | E[очки] (уже ×2) |", "|---|---|---|---|---|---|"]
    m0 = d["dtype"] == 2
    for s in range(4):
        for b in sorted(set(d["bid"][m0 & (d["seat"] == s)])):
            m = m0 & (d["seat"] == s) & (d["bid"] == b)
            if m.sum() < 20:
                continue
            out.append(f"| {s + 1} | {b} | {m.sum() / (m0 & (d['seat'] == s)).sum():.2f} | {d['tricks'][m].mean():.2f} | "
                       f"{np.mean(d['tricks'][m] == b):.2f} | %.2f±%.2f |" % mci(d["score"][m]))
    m1 = (d["dtype"] == 1)
    out.append("")
    out.append("E[очки] в тёмной (все места) = %.2f±%.2f; в бескозырке того же размера (с просмотром карт, ×1) = %.2f±%.2f" %
               (mci(d["score"][m0]) + mci(d["score"][m1])))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--ref-name", default="heuristic")
    a = ap.parse_args(argv)
    d = load(a.data)
    ref = load(a.ref) if a.ref else None
    print(seat_table(d))
    print()
    print(joker_tables(d))
    print()
    print(bid_behaviour(d, ref, a.ref_name))
    print()
    print(prev_bids_effect(d))
    print()
    print(dark_table(d))


if __name__ == "__main__":
    main()
