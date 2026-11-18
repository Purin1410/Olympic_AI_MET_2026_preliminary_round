"""Download external code and data at run time.

The original repositories (DimABSA2026 data, nchellwig helper.py) have no license,
so they are not copied into this repo. They are fetched at a pinned commit instead.
"""
import os
import subprocess
import urllib.request

from common.config import log, resolve


def fetch_repo(url, commit, dst):
    """Shallow-clone exactly one commit into `dst` (skipped if already present). Returns the absolute path."""
    dst = resolve(dst)
    if os.path.isdir(os.path.join(dst, ".git")):
        return dst
    os.makedirs(dst, exist_ok=True)
    for cmd in (["git", "init", "-q"],
                ["git", "remote", "add", "origin", url],
                ["git", "fetch", "-q", "--depth", "1", "origin", commit],
                ["git", "checkout", "-q", "FETCH_HEAD"]):
        subprocess.run(cmd, cwd=dst, check=True)
    log("fetched", url, "@", commit[:7], "->", dst)
    return dst


def fetch_file(url, dst):
    """Download one file to `dst` (skipped if already present). Returns the absolute path."""
    dst = resolve(dst)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        urllib.request.urlretrieve(url, dst)
        log("fetched", url, "->", dst)
    return dst
