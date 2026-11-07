"""Bid calibration analysis from collected self-play records (analysis.collect).

Produces markdown tables: per n_cards/seat stats, E[tricks] by trump count, OLS 'napkin' coefficients.
python -m analysis.calibration runs/data/best.npz > runs/analysis/calibration.md
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .collect import load


def mci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def fmt(m, h, p=2):
    return f"{m:.{p}f}±{h:.{p}f}"


def per_size_table(d, dtype=0):
    lines = ["| n_cards | seat | N | mean bid | mean tricks | P(exact) | P(over) | P(under) | score/deal |",
             "|---|---|---|---|---|---|---|---|---|"]
    sel0 = d["dtype"] == dtype
    for k in sorted(set(d["n_cards"][sel0])):
        for seat in range(4):
            m = sel0 & (d["n_cards"] == k) & (d["seat"] == seat)
            if m.sum() == 0:
                continue
            diff = d["tricks"][m] - d["bid"][m]
            sm, sh = mci(d["score"][m])
            lines.append(f"| {k} | {seat}{' (dealer)' if seat == 3 else ''} | {m.sum()} | {d['bid'][m].mean():.2f} | "
                         f"{d['tricks'][m].mean():.2f} | {np.mean(diff == 0):.3f} | {np.mean(diff > 0):.3f} | "
                         f"{np.mean(diff < 0):.3f} | {fmt(sm, sh)} |")
    return "\n".join(lines)


def tricks_by_trumps(d):
    lines = ["| n_cards | 0 trumps | 1 | 2 | 3 | 4+ | joker (any) |", "|---|---|---|---|---|---|---|"]
    m0 = (d["dtype"] == 0) & (d["trump"] >= 0)
    for k in sorted(set(d["n_cards"][m0])):
        row = [str(k)]
        for nt in range(5):
            m = m0 & (d["n_cards"] == k) & (d["joker"] == 0) & ((d["n_trump"] == nt) if nt < 4 else (d["n_trump"] >= 4))
            row.append(fmt(*mci(d["tricks"][m])) + f" (n={m.sum()})" if m.sum() > 30 else "—")
        m = m0 & (d["n_cards"] == k) & (d["joker"] == 1)
        row.append(fmt(*mci(d["tricks"][m])) if m.sum() > 30 else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


FEATS_T = ["joker", "trump_A", "trump_K", "trump_Q", "n_trump", "aces_side", "kings_side", "voids"]
FEATS_NT = ["joker", "aces_side", "kings_side", "max_len", "hcp"]


def ols(d, mask, feats, target="tricks"):
    X = np.column_stack([np.ones(mask.sum())] + [d[f][mask].astype(float) for f in feats])
    y = d[target][mask].astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res @ res / max(1, len(y) - X.shape[1])
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    ols.last_rmse = float(np.sqrt(res @ res / len(y)))
    return beta, se, r2


def ols_table(d):
    lines = ["| n_cards | " + " | ".join(["const"] + FEATS_T) + " | R² |", "|---" * (len(FEATS_T) + 3) + "|"]
    coefs = {}
    m0 = (d["dtype"] == 0) & (d["trump"] >= 0)
    for k in sorted(set(d["n_cards"][m0])):
        m = m0 & (d["n_cards"] == k)
        if m.sum() < 200:
            continue
        b, se, r2 = ols(d, m, FEATS_T)
        coefs[int(k)] = dict(zip(["const"] + FEATS_T, map(float, b)))
        coefs[int(k)]["_rmse"] = ols.last_rmse
        lines.append(f"| {k} | " + " | ".join(f"{bi:+.2f}" for bi in b) + f" | {r2:.2f} |")
    return "\n".join(lines), coefs


def ols_nt(d):
    m = (d["dtype"] != 2) & (d["trump"] < 0) & (d["n_cards"] == 9)
    if m.sum() < 200:
        return "", {}
    b, se, r2 = ols(d, m, FEATS_NT)
    line = "| 9 (NT) | " + " | ".join(f"{bi:+.2f}±{1.96*s:.2f}" for bi, s in zip(b, se)) + f" | {r2:.2f} |"
    head = "| n_cards | " + " | ".join(["const"] + FEATS_NT) + " | R² |\n" + "|---" * (len(FEATS_NT) + 3) + "|"
    c = dict(zip(["const"] + FEATS_NT, map(float, b)))
    c["_rmse"] = ols.last_rmse
    return head + "\n" + line, c


def main():
    d = load(sys.argv[1])
    print("## Per n_cards and seat (normal deals)\n")
    print(per_size_table(d, 0))
    print("\n## No-trump special deals\n")
    print(per_size_table(d, 1))
    print("\n## Dark deals\n")
    print(per_size_table(d, 2))
    print("\n## E[tricks] by number of trumps (hands without joker), trump deals\n")
    print(tricks_by_trumps(d))
    t, coefs = ols_table(d)
    print("\n## OLS: tricks ~ features, per n_cards (trump deals)\n")
    print(t)
    t2, c2 = ols_nt(d)
    print("\n## OLS no-trump 9-card deals\n")
    print(t2)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            json.dump(dict(trump=coefs, nt=c2), f, indent=1)


if __name__ == "__main__":
    main()
