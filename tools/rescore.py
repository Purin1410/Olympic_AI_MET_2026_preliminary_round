"""Re-score the shipped predictions in reference_outputs/ with the official DimABSA scorer. CPU only, no training.

Run from the repo root:  python -m tools.rescore
Needs numpy, scipy, pyyaml (preinstalled on Colab). Downloads the DimABSA2026 repo (pinned commit) on first run.
"""
import os
import glob
import tempfile

import numpy as np
import yaml

from common.config import resolve
from common.fetch import fetch_repo
from common.scoring import official_score
from logsigma.data import gold_path, load_eval, write_predictions

REF = resolve("reference_outputs")


def main():
    lcfg = yaml.safe_load(open(resolve("logsigma/config.yaml")))["data"]
    scfg = yaml.safe_load(open(resolve("scsg/config.yaml")))["data"]
    fetch_repo(lcfg["repo_url"], lcfg["repo_commit"], resolve(lcfg["repo_dir"]))
    scorer = os.path.join(resolve(lcfg["repo_dir"]), lcfg["scorer"])
    rows = []

    # Subtask 1 (LogSigma): each seed from pred_test.npy, plus the two ensemble files
    gold1 = gold_path(lcfg, "test")
    test_rows = load_eval(lcfg, "test")
    tmp = tempfile.mkdtemp()
    for npy in sorted(glob.glob(os.path.join(REF, "logsigma", "*", "runs", "*", "pred_test.npy"))):
        name = os.path.basename(os.path.dirname(npy))
        path = os.path.join(tmp, f"{name}.jsonl")
        write_predictions(np.load(npy), test_rows, path)
        rows.append(("LogSigma", name, official_score(scorer, path, gold1, 1)[1]))
    for path in sorted(glob.glob(os.path.join(REF, "logsigma", "*", "pred_test_*.jsonl"))):
        name = os.path.basename(path)[len("pred_test_"):-len(".jsonl")]
        rows.append(("LogSigma", name, official_score(scorer, path, gold1, 1)[1]))

    # Subtask 2 (SCSG): every pred_*_test_*.jsonl of every model
    gold3 = os.path.join(resolve(scfg["dimabsa_dir"]), scfg["gold_dir"],
                         f"{scfg['language']}_{scfg['domain']}_test_task{scfg['subtask']}.jsonl")
    for path in sorted(glob.glob(os.path.join(REF, "scsg", "*", "pred_*_test_*.jsonl"))):
        model = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path)[len("pred_"):-len(".jsonl")]
        rows.append(("SCSG " + model, name, official_score(scorer, path, gold3, 3)[1]))

    print(f"\n{'method':18s} {'run':28s} result")
    for method, name, r in rows:
        if "RMSE_VA" in r:
            print(f"{method:18s} {name:28s} RMSE_VA={r['RMSE_VA']:.4f}")
        else:
            print(f"{method:18s} {name:28s} cP={r.get('cPrecision', float('nan')):.4f} "
                  f"cR={r.get('cRecall', float('nan')):.4f} cF1={r.get('cF1', float('nan')):.4f}")
    return rows


if __name__ == "__main__":
    main()
