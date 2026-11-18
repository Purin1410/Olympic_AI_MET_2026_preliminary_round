"""From raw LLM text to quadruples, and self-consistency voting over n samples.

Self-consistency (as merge_predictions in the author's evaluate.py):
    every sample proposes a set of (Aspect, Category, Opinion, V, A) tuples;
    a tuple key (Aspect, Category, Opinion) is kept if it appears in at least `min_votes` samples;
    its V and A are the mean over the samples that proposed it.
Keys are compared case-sensitively, exactly like the author; the official scorer lowercases later.
"""
import statistics
from collections import defaultdict


def parse_output(cfg, H, raw, example, drop_null):
    """Raw text -> prediction {ID, Quadruplet: [...]} using the author's parser. Returns (prediction, parsed_ok).
    A parse failure gives an empty prediction, as in the author's format_predictions."""
    st = cfg["data"]["subtask"]
    try:
        tuples = H.parse_label_string(raw, subtask=st)
        pred = H.convert_tuples_to_output_format(tuples, example["id"], subtask=st,
                                                 disable_null_aspect=drop_null, text=example["text"])
        return pred, len(tuples) > 0
    except Exception:
        return {"ID": example["id"], "Quadruplet": []}, False


def majority_vote(preds, min_votes):
    """Merge the predictions of n samples for one sentence."""
    votes = defaultdict(list)
    for p in preds:
        for q in p["Quadruplet"]:
            v, a = map(float, q["VA"].split("#"))
            votes[(q["Aspect"], q["Category"], q["Opinion"])].append((v, a))
    kept = [{"Aspect": k[0], "Category": k[1], "Opinion": k[2],
             "VA": f"{statistics.mean(x[0] for x in vs):.2f}#{statistics.mean(x[1] for x in vs):.2f}"}
            for k, vs in votes.items() if len(vs) >= min_votes]
    return {"ID": preds[0]["ID"], "Quadruplet": kept}
