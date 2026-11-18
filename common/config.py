"""Config loading and run folder setup.

All hyperparameters live in one YAML file per method. The command line only accepts
the config path and a run name; everything else is edited in the YAML.
"""
import os
import sys
import time
import argparse

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = time.time()


def log(*args):
    """Print with elapsed seconds since the process started."""
    print(f"[{time.time() - T0:7.0f}s]", *args, flush=True)


def resolve(path):
    """Relative paths in the config are relative to the repo root, not to the current directory."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_config(default_config, argv=None):
    """Parse `--config` and `--run_name`, return the config dict.

    default_config: config.yaml next to the entry point, used when --config is omitted.
    argv: pass a list (e.g. []) to call this from a notebook instead of the command line.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=default_config)
    p.add_argument("--run_name", default=None, help="overrides run.name in the config")
    args = p.parse_args(argv)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.run_name:
        cfg["run"]["name"] = args.run_name
    if not cfg["run"]["name"]:
        cfg["run"]["name"] = time.strftime("%Y%m%d_%H%M%S")
    return cfg


class _Tee:
    """Write to the console and to a file at the same time."""

    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def isatty(self):
        return False


def setup_run(cfg, method):
    """Create outputs/<method>/<run_name>/, mirror stdout/stderr into run.log, save config_used.yaml.

    Returns the output folder path.
    """
    out = os.path.join(resolve(cfg["run"]["output_root"]), method, cfg["run"]["name"])
    os.makedirs(out, exist_ok=True)
    if not isinstance(sys.stdout, _Tee):   # avoid double logging when called twice in a notebook
        f = open(os.path.join(out, "run.log"), "a", encoding="utf-8")
        sys.stdout, sys.stderr = _Tee(sys.stdout, f), _Tee(sys.stderr, f)
    with open(os.path.join(out, "config_used.yaml"), "w", encoding="utf-8") as g:
        yaml.safe_dump(cfg, g, allow_unicode=True, sort_keys=False)
    log("output folder:", out)
    log("config:\n" + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return out
