"""Distil the best agent's BIDDING into a shallow decision tree (human-readable rules).

python -m analysis.distill runs/data/best.npz --depth 4
Saves runs/analysis/tree_d<depth>.pkl (used by wrap mode=tree) and prints rules + accuracy.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

from .collect import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATS = ["n_cards", "seat", "is_trump_deal", "n_trump", "trump_A", "trump_K", "trump_hi", "aces_side", "kings_side",
         "joker", "max_len", "voids", "hcp", "prev_bid_sum", "forbidden_rel"]


def feat_matrix(d):
    cols = []
    for f in FEATS:
        if f == "is_trump_deal":
            cols.append((d["trump"] >= 0).astype(int))
        elif f == "forbidden_rel":  # dealer: forbidden bid (or -1)
            cols.append(d["forbidden"])
        else:
            cols.append(d[f])
    return np.column_stack(cols)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--depth", type=int, nargs="+", default=[3, 4, 6])
    a = ap.parse_args(argv)
    d = load(a.data)
    m = d["dtype"] != 2  # dark bids are blind -> excluded
    X = feat_matrix(d)[m]
    y = d["bid"][m]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, te = idx[: int(0.8 * len(y))], idx[int(0.8 * len(y)):]
    os.makedirs(os.path.join(ROOT, "runs", "analysis"), exist_ok=True)
    for dep in a.depth:
        clf = DecisionTreeClassifier(max_depth=dep, min_samples_leaf=50, random_state=0).fit(X[tr], y[tr])
        acc = (clf.predict(X[te]) == y[te]).mean()
        within1 = (np.abs(clf.predict(X[te]) - y[te]) <= 1).mean()
        print(f"\n### depth {dep}: exact agreement {acc:.3f}, within ±1 {within1:.3f} (test n={len(te)})\n")
        if dep <= 4:
            print("```\n" + export_text(clf, feature_names=FEATS) + "```")
        with open(os.path.join(ROOT, "runs", "analysis", f"tree_d{dep}.pkl"), "wb") as f:
            pickle.dump(dict(clf=clf, feats=FEATS), f)


if __name__ == "__main__":
    main()
