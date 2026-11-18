"""Run the official DimABSA2026 scorer (evaluation_script/metrics_subtask_1_2_3.py).

Task 1 -> RMSE_VA, PCC_V, PCC_A.   Task 3 -> cPrecision, cRecall, cF1 (continuous F1).
"""
import re
import sys
import subprocess

from common.config import log


def official_score(scorer, pred, gold, task):
    """Return (result line, dict of numbers) for prediction file `pred` against gold file `gold`."""
    r = subprocess.run([sys.executable, scorer, "-p", pred, "-g", gold, "-t", str(task)],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    # scipy may print warnings after the result line, so search for it instead of taking the last line
    hits = [line for line in out.splitlines() if "Final Results" in line]
    line = hits[-1] if hits else (out.splitlines()[-1] if out else "")
    nums = {k: float(v) for k, v in re.findall(r"'(\w+)': (?:np\.float64\()?([-\d.eE]+)", line)}
    if not nums:
        log("WARNING: scorer returned no numbers:\n" + out[-2000:])
    return line, nums
