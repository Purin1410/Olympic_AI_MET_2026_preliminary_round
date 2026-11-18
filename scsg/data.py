"""Data, prompts and labels for quadruple extraction (DimASQP).

Reading the data, building the prompt and parsing the LLM output are done by the author's
helper.py (downloaded at run time), so prompts and parsing are exactly the same as the original.
Known quirk of that parser: tuples whose opinion contains an apostrophe ("aren't") are dropped,
which loses 48/2129 gold test tuples (cF1 ceiling ~0.989).
"""
import os
import sys
import contextlib

from common.config import resolve

# chat tags per model family (formatting constants, as in the author's llm.py)
CHAT = {
    "gemma":   {"us": "<start_of_turn>user\n", "ue": "<end_of_turn>\n", "ms": "<start_of_turn>model\n",
                "me": "<end_of_turn>", "template": "gemma-3"},
    "mistral": {"us": "[INST] ", "ue": "", "ms": " [/INST]", "me": "</s>", "template": "mistral"},
    "qwen":    {"us": "<|im_start|>user\n", "ue": "<|im_end|>\n", "ms": "<|im_start|>assistant\n",
                "me": "<|im_end|>", "template": "qwen-3"},
}


@contextlib.contextmanager
def working_dir(path):
    """helper.py uses relative paths ('task-dataset/...', 'logs.txt'), so call it from the right folder."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_helper(cfg):
    """Import the author's helper.py (its import creates a logs.txt next to it)."""
    helper_dir = os.path.dirname(resolve(cfg["data"]["helper_path"]))
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)
    with working_dir(helper_dir):
        import helper
    return helper


def load_datasets(cfg, H):
    """Returns (train examples, eval examples, eval split name). Each example: {id, text, label}."""
    d = cfg["data"]
    args = (d["subtask"], d["language"], d["domain"])
    with working_dir(resolve(d["dimabsa_dir"])):
        if d["strategy"] == "test-train_dev":      # final submission setting: train on train+dev, evaluate on test
            train, split = H.get_dataset(*args, split="train") + H.get_dataset(*args, split="dev"), "test"
        elif d["strategy"] == "dev-train":         # development setting: train on train, evaluate on dev
            train, split = H.get_dataset(*args, split="train"), "dev"
        else:
            raise ValueError(d["strategy"])
        ev = H.get_dataset(*args, split=split)
    if d["eval_limit"]:
        ev = ev[:d["eval_limit"]]
    return train, ev, split


def build_prompt(cfg, H, text):
    """User turn with the author's instruction prompt, followed by the start of the model turn."""
    d, c = cfg["data"], CHAT[cfg["model"]["chat_family"]]
    return c["us"] + H.get_prompt(text, d["subtask"], d["language"], d["domain"]) + c["ue"] + c["ms"]


def build_label(cfg, H, example):
    """Training target = str(list of tuples), e.g. "[('food', 'FOOD#QUALITY', 'great', '7.50', '7.25')]"."""
    return str(H.convert_label_objects_to_tuples(example["label"], cfg["data"]["subtask"]))


def end_of_turn(cfg):
    return CHAT[cfg["model"]["chat_family"]]["me"]
