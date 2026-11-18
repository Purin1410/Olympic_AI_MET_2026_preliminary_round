"""SCSG (nchellwig, 2nd place SemEval-2026 Task 3): quadruple extraction (Aspect, Category, Opinion, V#A)
with a QLoRA fine-tuned LLM and self-consistency.

Pipeline (entry point: `python -m scsg.run`):

    data.py        load data, build prompts and training labels (uses the author's helper.py)
    trainer.py     QLoRA fine-tuning with TRL SFTTrainer (unsloth or plain transformers + peft)
    generation.py  load base model + LoRA adapter, sample n outputs per prompt (OOM-safe batching)
    inference.py   run each decoding phase (greedy, self-consistency) and save raw LLM outputs
    voting.py      parse raw outputs into quadruples, merge n samples by majority vote
    evaluate.py    build predictions per phase and score them with the official scorer
    run.py         read config -> train -> inference -> evaluation -> results.json
"""
