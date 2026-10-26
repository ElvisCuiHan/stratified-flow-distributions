from __future__ import annotations

import numpy as np


def scatter2d(x, info=None, ax=None, *, s=8, alpha=0.65):
    """Quick 2D scatter plot for simulated SFD samples.

    Matplotlib is imported lazily so plotting remains an optional dependency.
    """

    import matplotlib.pyplot as plt

    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("scatter2d expects an array with shape (n, d), d >= 2.")
    if ax is None:
        _, ax = plt.subplots()
    if info is not None and "component_index" in info:
        c = np.asarray(info["component_index"]).reshape(-1)
    else:
        c = None
    ax.scatter(x[:, 0], x[:, 1], c=c, s=s, alpha=alpha)
    ax.set_xlabel("Y1")
    ax.set_ylabel("Y2")
    ax.set_aspect("equal", adjustable="box")
    return ax

