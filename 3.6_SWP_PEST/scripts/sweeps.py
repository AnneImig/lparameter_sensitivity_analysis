"""Shared paths and helpers for the Sect. 3.6 figures (the two 1,000,000-run sweeps)."""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'results')
LY1 = os.path.join(RES, 'ly1_1M')       # Ly1 l-free sweep
LY2 = os.path.join(RES, 'ly2_1M')    # Ly2 l-free sweep (alpha 1e-3-0.5, n 1.05-4)


def best_per_class(d, edges, seed=0):
    """Best and median KGE per l class, the best taken from an equal-size random subsample per class."""
    u = d[d.ok]
    idx = np.clip(np.digitize(u.l1, edges) - 1, 0, len(edges) - 2)
    pools = [u.kge.to_numpy()[idx == i] for i in range(len(edges) - 1)]
    floor = min(len(p) for p in pools)
    rng = np.random.default_rng(seed)
    return ([rng.choice(p, floor, replace=False).max() for p in pools],
            [np.median(p) for p in pools], floor)
