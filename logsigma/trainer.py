"""Train one LogSigma run (one seed) with early stopping on dev RMSE_VA.

Files written to <output>/runs/<run_name>/:
    history.json     one entry per epoch: train loss, train MSE V/A, dev RMSE (VA, V, A), sigma^2,
                     effective loss weights 0.5/sigma^2, learning rate, seconds, peak GPU memory
    steps.csv        every `log.step_interval` optimizer steps: loss, MSE V/A, sigma^2, lr
    pred_dev.npy, pred_test.npy   unrounded predictions [N, 2] of the best epoch
    pred_test.jsonl  test predictions in submission format
    summary.json     best epoch, best dev RMSE, test RMSE, sigma^2 at init and at the best epoch
    checkpoint_best_fp16.pt       only for runs listed in log.save_checkpoint_runs
"""
import os
import json
import math
import time
import random

import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from common.config import log
from logsigma.data import build_inputs, write_predictions
from logsigma.metrics import rmse_va
from logsigma.model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Encoder:
    """Tokenise (text, aspect) pairs into model inputs on the target device."""

    def __init__(self, cfg, device):
        self.dcfg = cfg["data"]
        self.tok = AutoTokenizer.from_pretrained(cfg["model"]["encoder"])
        self.device = device

    def __call__(self, texts, aspects):
        return self.tok(build_inputs(self.dcfg, texts, aspects), truncation=True, max_length=self.dcfg["max_length"],
                        padding=True, return_tensors="pt").to(self.device)


@torch.no_grad()
def predict(model, encode, rows, cfg, device):
    """Predict [N, 2] for eval rows (ID, text, aspect, VA), clipped to the VA range."""
    model.eval()
    amp = getattr(torch, cfg["hardware"]["amp_dtype"])
    bs = cfg["train"]["eval_batch_size"]
    out = []
    for b in range(0, len(rows), bs):
        chunk = rows[b:b + bs]
        e = encode([r[1] for r in chunk], [r[2] for r in chunk])
        with torch.autocast(device.type, dtype=amp, enabled=device.type == "cuda"):
            y, _, _ = model(e["input_ids"], e["attention_mask"])
        out.append(y.float().cpu().numpy())
    lo, hi = cfg["data"]["va_range"]
    return np.clip(np.concatenate(out), lo, hi)


def make_optimizer(model, tcfg, rcfg, n_train):
    """AdamW with two parameter groups (encoder + head, and log sigma^2) and linear warmup/decay."""
    rest = [p for n, p in model.named_parameters() if not n.startswith("log_var")]
    groups = [{"params": rest, "lr": tcfg["lr_encoder"], "weight_decay": tcfg["weight_decay"]}]
    if rcfg["use_uncertainty"]:
        groups.append({"params": [model.log_var_v, model.log_var_a], "lr": rcfg["lr_uncertainty"], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    steps_per_epoch = math.ceil(n_train / tcfg["batch_size"] / tcfg["grad_accum"])
    total = steps_per_epoch * tcfg["max_epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(tcfg["warmup_ratio"] * total), total)
    return opt, sched


def train_run(cfg, rcfg, data, out_dir, device):
    """Train the run described by `rcfg` (an entry of cfg['runs']). data = (train, dev, test). Returns the summary dict."""
    tcfg, lcfg = cfg["train"], cfg["log"]
    train_rows, dev_rows, test_rows = data
    name = rcfg["name"]
    run_dir = os.path.join(out_dir, "runs", name)
    os.makedirs(run_dir, exist_ok=True)
    use_amp = device.type == "cuda"
    amp = getattr(torch, cfg["hardware"]["amp_dtype"])

    set_seed(rcfg["seed"])
    encode = Encoder(cfg, device)
    model = build_model(cfg["model"], rcfg["use_uncertainty"]).to(device)
    opt, sched = make_optimizer(model, tcfg, rcfg, len(train_rows))
    scaler = torch.amp.GradScaler(enabled=use_amp)
    var_init = model.sigma2()
    log(f"[{name}] seed={rcfg['seed']} uncertainty={rcfg['use_uncertainty']} train={len(train_rows)} "
        f"dev={len(dev_rows)} test={len(test_rows)} sigma2_init V={var_init[0]:.3f} A={var_init[1]:.3f}")

    BS, GA = tcfg["batch_size"], tcfg["grad_accum"]
    steps_f = open(os.path.join(run_dir, "steps.csv"), "w")
    steps_f.write("step,epoch,loss,mse_v,mse_a,var_v,var_a,lr_encoder\n")
    best, best_epoch, best_state, bad, history, step = float("inf"), 0, None, 0, [], 0

    for epoch in range(1, tcfg["max_epochs"] + 1):
        model.train()
        t = time.time()
        sums = np.zeros(3)     # loss, MSE_V, MSE_A over the epoch
        window = np.zeros(4)   # same + count, between two lines of steps.csv
        n_batches = 0
        order = np.random.permutation(len(train_rows))
        opt.zero_grad()
        if use_amp:
            torch.cuda.reset_peak_memory_stats()
        for k, b in enumerate(range(0, len(order), BS)):
            batch = [train_rows[i] for i in order[b:b + BS]]
            e = encode([r[0] for r in batch], [r[1] for r in batch])
            labels = torch.tensor([r[2] for r in batch], device=device)
            with torch.autocast(device.type, dtype=amp, enabled=use_amp):
                _, loss, (mse_v, mse_a) = model(e["input_ids"], e["attention_mask"], labels)
            scaler.scale(loss / GA).backward()
            vals = np.array([loss.item(), mse_v.item(), mse_a.item()])
            sums += vals
            window += np.append(vals, 1)
            n_batches += 1
            # optimizer step every GA batches (gradient accumulation -> effective batch BS * GA)
            if (k + 1) % GA == 0 or b + BS >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
                step += 1
                if step % lcfg["step_interval"] == 0:
                    sv, sa = model.sigma2()
                    l, mv, ma = window[:3] / window[3]
                    steps_f.write(f"{step},{epoch},{l:.5f},{mv:.5f},{ma:.5f},{sv:.5f},{sa:.5f},{sched.get_last_lr()[0]:.3e}\n")
                    steps_f.flush()
                    window[:] = 0

        P_dev = predict(model, encode, dev_rows, cfg, device)
        r, rv, ra = rmse_va(P_dev, dev_rows)
        sv, sa = model.sigma2()
        # effective weight of each MSE term = 0.5 / sigma^2 (1 without uncertainty)
        wv, wa = (0.5 / sv, 0.5 / sa) if rcfg["use_uncertainty"] else (1.0, 1.0)
        l, mv, ma = sums / n_batches
        history.append({"epoch": epoch, "train_loss": l, "train_mse_v": mv, "train_mse_a": ma,
                        "dev_rmse": r, "dev_rmse_v": rv, "dev_rmse_a": ra, "var_v": sv, "var_a": sa,
                        "w_eff_v": wv, "w_eff_a": wa, "lr_encoder": sched.get_last_lr()[0], "sec": time.time() - t,
                        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if use_amp else None})
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
        log(f"[{name}] epoch {epoch} loss={l:.4f} dev RMSE_VA={r:.4f} (V {rv:.3f}, A {ra:.3f}) "
            f"sigma2 V={sv:.3f} A={sa:.3f} ({time.time() - t:.0f}s)")

        # early stopping: keep the best weights on dev, stop after `patience` epochs without improvement
        if r < best:
            best, best_epoch, bad = r, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg["early_stop_patience"]:
                log(f"[{name}] early stop at epoch {epoch}, best epoch {best_epoch}")
                break
    steps_f.close()

    model.load_state_dict(best_state)
    if name in lcfg["save_checkpoint_runs"]:
        path = os.path.join(run_dir, "checkpoint_best_fp16.pt")
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}, path)
        log(f"[{name}] saved {path}")
    P_dev = predict(model, encode, dev_rows, cfg, device)
    P_test = predict(model, encode, test_rows, cfg, device)
    np.save(os.path.join(run_dir, "pred_dev.npy"), P_dev)
    np.save(os.path.join(run_dir, "pred_test.npy"), P_test)
    write_predictions(P_test, test_rows, os.path.join(run_dir, "pred_test.jsonl"))
    te, tv, ta = rmse_va(P_test, test_rows)
    best_h = history[best_epoch - 1]
    summary = {"run": rcfg, "best_epoch": best_epoch, "epochs_run": len(history), "best_dev_rmse": best,
               "test_rmse": te, "test_rmse_v": tv, "test_rmse_a": ta, "var_init": list(var_init),
               "var_at_best": [best_h["var_v"], best_h["var_a"]], "train_sec": sum(h["sec"] for h in history)}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[{name}] done: best dev {best:.4f} (epoch {best_epoch}), test RMSE_VA {te:.4f}")
    return summary
