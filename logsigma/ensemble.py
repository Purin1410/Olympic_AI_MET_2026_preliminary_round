"""Ensemble = average of the (clipped) predictions of several runs, as in the paper (3 seeds)."""
import os

import numpy as np

from common.scoring import official_score
from logsigma.data import write_predictions, gold_path
from logsigma.metrics import rmse_va


def ensemble(cfg, name, members, rows_by_split, out_dir, scorer):
    """Average pred_{dev,test}.npy of `members`, write pred_<split>_<name>.jsonl, score with the official scorer."""
    result = {"members": members}
    for split, rows in rows_by_split.items():
        P = np.mean([np.load(os.path.join(out_dir, "runs", m, f"pred_{split}.npy")) for m in members], axis=0)
        path = os.path.join(out_dir, f"pred_{split}_{name}.jsonl")
        write_predictions(P, rows, path)
        result[f"{split}_rmse"], result[f"{split}_rmse_v"], result[f"{split}_rmse_a"] = rmse_va(P, rows)
        result[f"official_{split}"] = official_score(scorer, path, gold_path(cfg["data"], split), 1)[1]
    return result
