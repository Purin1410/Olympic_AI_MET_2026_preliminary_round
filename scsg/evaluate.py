"""Turn raw outputs into predictions for each phase and score them with the official scorer (cF1).

Files written to the output folder:
    gold_eval.jsonl                                   gold of the evaluated sentences (respects data.eval_limit)
    pred_<phase>_s<seed>_<split>_null<mode>.jsonl     predictions in submission format
null_mode 'drop' removes tuples with Aspect=NULL (author's setting, the English test set has none); 'keep' keeps them.
"""
import os
import statistics
from collections import defaultdict

from common.config import log, resolve
from common.jsonl import read_jsonl, write_jsonl
from common.scoring import official_score
from scsg.voting import parse_output, majority_vote


def build_predictions(cfg, H, examples, raw_by_id, n_samples, drop_null):
    """Returns (predictions, parse_ok_rate). n_samples == 1 -> the single output; otherwise majority vote."""
    min_votes = cfg["score"]["min_votes"] or n_samples // 2 + 1
    preds, ok, total = [], 0, 0
    for ex in examples:
        parsed = [parse_output(cfg, H, raw, ex, drop_null) for raw in raw_by_id[ex["id"]]]
        ok += sum(p_ok for _, p_ok in parsed)
        total += len(parsed)
        preds.append(parsed[0][0] if n_samples == 1 else majority_vote([p for p, _ in parsed], min_votes))
    return preds, ok / max(1, total)


def evaluate(cfg, H, examples, split, out_dir):
    d = cfg["data"]
    gold_src = os.path.join(resolve(d["dimabsa_dir"]), d["gold_dir"],
                            f"{d['language']}_{d['domain']}_{split}_task{d['subtask']}.jsonl")
    ids = {e["id"] for e in examples}
    gold = os.path.join(out_dir, "gold_eval.jsonl")
    write_jsonl([g for g in read_jsonl(gold_src) if g["ID"] in ids], gold)
    scorer = os.path.join(resolve(d["dimabsa_dir"]), d["scorer"])

    results = {}
    for seed in cfg["infer"]["eval_seeds"]:
        for phase, p in cfg["infer"]["phases"].items():
            raw_path = os.path.join(out_dir, f"raw_{phase}_s{seed}.jsonl")
            if not os.path.exists(raw_path):
                log("missing", raw_path, "-> not scored")
                continue
            raw_by_id = {r["ID"]: r["outputs"] for r in read_jsonl(raw_path)}
            if any(e["id"] not in raw_by_id for e in examples):
                log(raw_path, "does not cover every sentence -> not scored")
                continue
            for mode in cfg["score"]["null_modes"]:
                preds, parse_ok = build_predictions(cfg, H, examples, raw_by_id, p["n"], mode == "drop")
                tag = f"{phase}_s{seed}_{split}_null{mode}"
                pred_path = os.path.join(out_dir, f"pred_{tag}.jsonl")
                write_jsonl(preds, pred_path)
                line, nums = official_score(scorer, pred_path, gold, d["subtask"])
                results[tag] = {"phase": phase, "eval_seed": seed, "null_mode": mode, **nums, "official": line,
                                "parse_ok_rate": parse_ok,
                                "min_votes": (cfg["score"]["min_votes"] or p["n"] // 2 + 1) if p["n"] > 1 else None,
                                "n_pred_tuples": sum(len(q["Quadruplet"]) for q in preds), "n_sent": len(examples)}
                log(tag, {k: results[tag].get(k) for k in ("cPrecision", "cRecall", "cF1", "parse_ok_rate")})

    # mean / std over eval seeds (the author reports seeds 0..4)
    by_phase = defaultdict(list)
    for r in results.values():
        if "cF1" in r:
            by_phase[f"{r['phase']}_null{r['null_mode']}"].append(r["cF1"])
    results["_by_seed"] = {k: {"n_seeds": len(v), "cF1_mean": statistics.mean(v),
                               "cF1_std": statistics.pstdev(v) if len(v) > 1 else 0.0, "cF1_all": v}
                           for k, v in by_phase.items()}
    return results
