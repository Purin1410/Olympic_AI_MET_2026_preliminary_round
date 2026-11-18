"""Entry point: python -m scsg.run [--config scsg/config.yaml] [--run_name NAME]

Steps: fetch data + author's helper.py -> train (QLoRA) -> inference (greedy, self-consistency) -> evaluation.
All parameters are in scsg/config.yaml. One process, one GPU.

Output folder outputs/scsg/<run_name>/:
    config_used.yaml, run.log, env.json, gpu_util.csv
    data_stats.json, train_log_history.json, train_stats.json, adapter/    see scsg/trainer.py
    raw_*.jsonl, infer_stats_*.json                                         see scsg/inference.py
    gold_eval.jsonl, pred_*.jsonl                                           see scsg/evaluate.py
    results.json        official scores per phase / seed / null mode, plus timing and stats
"""
import os
import json
import time

from common.config import T0, log, load_config, setup_run
from common.env import env_info, start_gpu_monitor
from common.fetch import fetch_repo, fetch_file
from scsg.data import load_helper, load_datasets
from scsg.trainer import train
from scsg.inference import run_inference
from scsg.evaluate import evaluate

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def main(argv=None):
    cfg = load_config(DEFAULT_CONFIG, argv)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = cfg["hardware"]["alloc_conf"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    out = setup_run(cfg, "scsg")
    d = cfg["data"]
    fetch_repo(d["dimabsa_repo_url"], d["dimabsa_repo_commit"], d["dimabsa_dir"])
    fetch_file(d["helper_url"], d["helper_path"])
    env_info(cfg["log"]["env_packages"], os.path.join(out, "env.json"))
    gpu_mon = start_gpu_monitor(cfg["log"]["gpu_monitor_interval_s"], os.path.join(out, "gpu_util.csv"))

    H = load_helper(cfg)
    train_examples, eval_examples, split = load_datasets(cfg, H)
    log(f"train: {len(train_examples)} sentences, evaluation on {split}: {len(eval_examples)} sentences")
    timing = {}

    # 1) fine-tune
    adapter = cfg["infer"]["adapter"] or os.path.join(out, "adapter")
    if cfg["train"]["enabled"]:
        t = time.time()
        adapter = train(cfg, H, train_examples, out)
        timing["train_sec"] = time.time() - t

    # 2) generate raw outputs for each phase
    if cfg["infer"]["enabled"]:
        t = time.time()
        run_inference(cfg, H, eval_examples, adapter, out)
        timing["infer_sec"] = time.time() - t

    # 3) parse, vote and score
    results = evaluate(cfg, H, eval_examples, split, out)
    timing["total_sec"] = time.time() - T0

    stats = {}
    for f in sorted(os.listdir(out)):
        if f in ("data_stats.json", "train_stats.json") or f.startswith("infer_stats_"):
            stats[f] = json.load(open(os.path.join(out, f)))
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"run_name": cfg["run"]["name"], "model": cfg["model"]["name"], "timing": timing, "stats": stats,
                   "results": results}, f, ensure_ascii=False, indent=1)
    if gpu_mon:
        gpu_mon.terminate()
    log("===== RESULTS (official scorer) =====")
    for tag, r in results.items():
        if "cF1" in r:
            log(f"{tag:32s} cP={r['cPrecision']:.4f} cR={r['cRecall']:.4f} cF1={r['cF1']:.4f} "
                f"parse_ok={r['parse_ok_rate']:.3f}")
    log("done, outputs in", out)
    return results


if __name__ == "__main__":
    main()
