"""Data loading for Subtask 1 (DimASR).

Each example is one (sentence, aspect) pair with a gold Valence/Arousal score on a 1-9 scale.
    train rows: (text, aspect, [V, A])
    dev / test rows: (sentence ID, text, aspect, [V, A])
"""
import os

from common.config import resolve
from common.jsonl import read_jsonl, write_jsonl


def data_dir(dcfg):
    return os.path.join(resolve(dcfg["repo_dir"]), dcfg["task_dir"])


def load_train(dcfg):
    """The train file has full quadruplets; keep (text, aspect, VA)."""
    rows = []
    for r in read_jsonl(os.path.join(data_dir(dcfg), dcfg["train_file"])):
        for q in r["Quadruplet"]:
            if dcfg["drop_null_aspect"] and q["Aspect"] == "NULL":
                continue
            v, a = map(float, q["VA"].split("#"))
            rows.append((r["Text"], q["Aspect"], [v, a]))
    if dcfg["train_limit"]:
        rows = rows[:dcfg["train_limit"]]
    return rows


def load_eval(dcfg, split):
    """split: 'dev' or 'test'. The eval files list the aspects of each sentence with their gold VA."""
    rows = []
    for r in read_jsonl(gold_path(dcfg, split)):
        for q in r["Aspect_VA"]:
            v, a = map(float, q["VA"].split("#"))
            rows.append((r["ID"], r["Text"], q["Aspect"], [v, a]))
    return rows


def gold_path(dcfg, split):
    return os.path.join(data_dir(dcfg), dcfg[f"{split}_file"])


def load_all(dcfg):
    return load_train(dcfg), load_eval(dcfg, "dev"), load_eval(dcfg, "test")


def build_inputs(dcfg, texts, aspects):
    """Model input = sentence and aspect joined by the template, e.g. "{text} [SEP] {aspect}"."""
    return [dcfg["input_template"].format(text=t, aspect=a) for t, a in zip(texts, aspects)]


def write_predictions(P, rows, path):
    """Submission format: one line per sentence, {"ID", "Aspect_VA": [{"Aspect", "VA": "v#a"}]}."""
    by_id = {}
    for (sid, _, aspect, _), p in zip(rows, P):
        by_id.setdefault(sid, []).append({"Aspect": aspect, "VA": f"{p[0]:.2f}#{p[1]:.2f}"})
    write_jsonl([{"ID": sid, "Aspect_VA": v} for sid, v in by_id.items()], path)
