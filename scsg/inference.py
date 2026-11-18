"""Run the decoding phases defined in config (greedy and self-consistency) and save raw LLM outputs.

Files written to the output folder, per eval seed and phase:
    raw_<phase>_s<seed>.jsonl    one line per sentence: {idx, ID, text, outputs[n], n_tok[n], finish[n]}
    infer_stats_s<seed>.json     seconds, seconds per sentence, generated tokens, truncated outputs, peak GPU memory
A phase whose raw file already exists is skipped, so a run interrupted on Colab can be resumed with the same run name.
"""
import os
import json
import time

from common.config import log
from common.jsonl import write_jsonl
from scsg.data import build_prompt
from scsg.generation import Generator


def run_inference(cfg, H, examples, adapter, out_dir):
    import torch
    icfg = cfg["infer"]
    prompts = [build_prompt(cfg, H, ex["text"]) for ex in examples]
    gen = None   # loaded lazily, only if some phase still has to run
    for seed in icfg["eval_seeds"]:
        for phase, p in icfg["phases"].items():
            raw_path = os.path.join(out_dir, f"raw_{phase}_s{seed}.jsonl")
            if os.path.exists(raw_path):
                log("found", raw_path, "-> skip")
                continue
            if gen is None:
                gen = Generator(cfg, adapter)
            torch.manual_seed(seed)
            torch.cuda.reset_peak_memory_stats()
            t, last = time.time(), [0.0]

            def progress(k):
                if time.time() - last[0] > 60 or k == len(prompts):
                    last[0] = time.time()
                    log(f"{phase} s{seed}: {k}/{len(prompts)} sentences, {(time.time() - t) / k:.1f} s/sentence")

            outs = gen.generate(prompts, p["temperature"], p["n"], progress)
            sec = time.time() - t
            write_jsonl([{"idx": i, "ID": ex["id"], "text": ex["text"], "outputs": o[0], "n_tok": o[1], "finish": o[2]}
                         for i, (ex, o) in enumerate(zip(examples, outs))], raw_path)

            n_tok = [k for o in outs for k in o[1]]
            stats_path = os.path.join(out_dir, f"infer_stats_s{seed}.json")
            stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
            stats[phase] = {"sec": sec, "sec_per_sent": sec / max(1, len(examples)), "n_sent": len(examples),
                            "n_samples": p["n"], "temperature": p["temperature"], "gen_tok_total": sum(n_tok),
                            "gen_tok_per_s": sum(n_tok) / max(sec, 1e-9), "gen_tok_mean": sum(n_tok) / max(1, len(n_tok)),
                            "gen_tok_max": max(n_tok, default=0),
                            "n_truncated": sum(f == "length" for o in outs for f in o[2]),
                            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=1)
            log(f"{phase} s{seed} done:", stats[phase])
            log("example output:", outs[0][0][:2])
