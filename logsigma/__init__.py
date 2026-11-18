"""LogSigma: Valence/Arousal regression per aspect (SemEval-2026 Task 3, Subtask 1 / DimASR).

Pipeline (entry point: `python -m logsigma.train`):

    data.py      load (text, aspect, [V, A]) pairs, write predictions in submission format
    model.py     encoder + 2-dim regression head + uncertainty-weighted loss (learned log sigma^2)
    metrics.py   RMSE_VA as defined by the task
    trainer.py   train one run (one seed), early stopping on dev, save predictions and logs
    ensemble.py  average the predictions of several runs
    train.py     read config -> train every enabled run -> ensemble -> official scoring

Re-implemented from the LogSigma paper (Table 6); the original repo only ships notebooks.
"""
