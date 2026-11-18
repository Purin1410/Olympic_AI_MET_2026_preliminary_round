"""Entry point: python -m logsigma.train [--config logsigma/config.yaml] [--run_name NAME]

Steps: fetch data -> train every enabled run -> ensembles -> official scoring -> results.json.
All parameters are in logsigma/config.yaml.

Output folder outputs/logsigma/<run_name>/:
    config_used.yaml, run.log, env.json, gpu_util.csv
    runs/<run>/...                      see logsigma/trainer.py
    pred_{dev,test}_<ensemble>.jsonl    ensemble predictions in submission format
    results.json                        all runs and ensembles with official test scores
"""
import os
import gc
import json
import time

import torch

from common.config import T0, log, resolve, load_config, setup_run
from common.env import env_info, start_gpu_monitor, pick_device
from common.fetch import fetch_repo
from common.scoring import official_score
from logsigma.data import load_all, gold_path
from logsigma.trainer import train_run
from logsigma.ensemble import ensemble

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def main(argv=None):
    cfg = load_config(DEFAULT_CONFIG, argv)
    out = setup_run(cfg, "logsigma")
    dcfg = cfg["data"]
    fetch_repo(dcfg["repo_url"], dcfg["repo_commit"], dcfg["repo_dir"])
    env_info(cfg["log"]["env_packages"], os.path.join(out, "env.json"))
    gpu_mon = start_gpu_monitor(cfg["log"]["gpu_monitor_interval_s"], os.path.join(out, "gpu_util.csv"))
    device = pick_device(cfg["hardware"]["device"])
    log("device:", device)

    train_rows, dev_rows, test_rows = load_all(dcfg)
    scorer = os.path.join(resolve(dcfg["repo_dir"]), dcfg["scorer"])
    results = {"run_name": cfg["run"]["name"], "device": str(device), "runs": {}, "ensembles": {}}

    def save():
        with open(os.path.join(out, "results.json"), "w") as f:
            json.dump(results, f, indent=1)

    # 1) train each enabled run one after another
    for rcfg in cfg["runs"]:
        if not rcfg["enabled"]:
            continue
        s = train_run(cfg, rcfg, (train_rows, dev_rows, test_rows), out, device)
        s["official_test"] = official_score(scorer, os.path.join(out, "runs", rcfg["name"], "pred_test.jsonl"),
                                            gold_path(dcfg, "test"), 1)[1]
        results["runs"][rcfg["name"]] = s
        save()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()   # free GPU memory before the next run

    # 2) ensembles whose members all finished
    for name, members in cfg["ensembles"].items():
        if all(m in results["runs"] for m in members):
            results["ensembles"][name] = ensemble(cfg, name, members, {"dev": dev_rows, "test": test_rows}, out, scorer)

    results["total_sec"] = time.time() - T0
    save()
    if gpu_mon:
        gpu_mon.terminate()
    log("===== RESULTS (test set, official scorer) =====")
    for k, v in list(results["runs"].items()) + list(results["ensembles"].items()):
        o = v["official_test"]
        log(f"{k:16s} RMSE_VA={o.get('RMSE_VA', float('nan')):.4f}  PCC_V={o.get('PCC_V', float('nan')):.4f}  "
            f"PCC_A={o.get('PCC_A', float('nan')):.4f}")
    log("done, outputs in", out)
    return results


if __name__ == "__main__":
    main()
