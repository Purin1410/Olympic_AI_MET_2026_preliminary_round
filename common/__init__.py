"""Shared utilities for the `logsigma` and `scsg` packages.

    config   -> load the YAML config, create the output folder, mirror stdout into run.log
    env      -> library versions, GPU monitor, device selection
    fetch    -> download the dataset repo / original author code at a pinned commit
    jsonl    -> read / write .jsonl files
    scoring  -> run the official DimABSA2026 scorer
"""
