"""Metrics for Subtask 1."""
import math

import numpy as np


def rmse_va(P, rows):
    """Official RMSE_VA = sqrt(mean(dV^2 + dA^2)) (not divided by sqrt(2)), plus RMSE of V and A separately.

    P: array [N, 2] of predictions; rows: eval rows whose last field is the gold [V, A].
    """
    G = np.array([r[-1] for r in rows])
    E = (P - G) ** 2
    return float(math.sqrt(E.sum(1).mean())), float(math.sqrt(E[:, 0].mean())), float(math.sqrt(E[:, 1].mean()))
