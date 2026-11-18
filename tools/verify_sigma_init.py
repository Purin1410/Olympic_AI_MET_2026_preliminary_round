"""Print the INITIAL sigma^2 (before any training) that LogSigma's code draws for given seeds.

It replays the random-number order of the LogSigma model: seed -> AutoModel.from_pretrained -> Dropout -> Linear(h, 2)
-> log_var_v, log_var_a = log(uniform(0.2, 1.0)). Compare the output with Table 10 of the LogSigma paper
(Track B, seeds 42/12/73): the "learned" sigma^2 values there match these untrained values to about +-0.001.

Usage:  python tools/verify_sigma_init.py cardiffnlp/twitter-roberta-large-topic-sentiment-latest 42 12 73
"""
import sys
import math
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

model_name, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
for seed in seeds:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    enc = AutoModel.from_pretrained(model_name)
    nn.Dropout(0.1)
    nn.Linear(enc.config.hidden_size, 2)
    lv = torch.log(torch.empty(1).uniform_(0.2, 1.0))
    la = torch.log(torch.empty(1).uniform_(0.2, 1.0))
    print(f"{model_name} seed {seed}: sigma2_V={math.exp(lv):.4f} sigma2_A={math.exp(la):.4f}", flush=True)
