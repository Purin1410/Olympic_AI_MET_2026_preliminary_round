"""QLoRA fine-tuning of the LLM to generate the list of quadruples (hyperparameters from the author's llm.py).

Files written to the output folder:
    data_stats.json          number of training examples, token lengths (mean / p95 / max), examples over max_seq_length
    train_log_history.json   loss, learning rate, grad norm every `train.logging_steps` steps
    train_stats.json         training time, number of steps, final train loss, peak GPU memory
    adapter/                 the LoRA adapter (base model weights are not saved)
"""
import os
import gc
import json
import time
import shutil

from common.config import log
from scsg.data import CHAT, build_prompt, build_label


def _training_rows(cfg, H, examples, backend):
    c = CHAT[cfg["model"]["chat_family"]]
    if backend == "unsloth":   # one text field; the loss is restricted to the answer later by train_on_responses_only
        return [{"text": build_prompt(cfg, H, ex["text"]) + build_label(cfg, H, ex) + c["me"]} for ex in examples]
    # prompt / completion format: TRL computes the loss on the completion only
    return [{"prompt": build_prompt(cfg, H, ex["text"]), "completion": build_label(cfg, H, ex) + c["me"]} for ex in examples]


def _trainer_args(cfg, out_dir):
    tcfg, mcfg = cfg["train"], cfg["model"]
    return dict(per_device_train_batch_size=tcfg["per_device_batch_size"], gradient_accumulation_steps=tcfg["grad_accum"],
                warmup_steps=tcfg["warmup_steps"], num_train_epochs=tcfg["epochs"], learning_rate=tcfg["lr"],
                logging_steps=tcfg["logging_steps"], optim=tcfg["optim"], weight_decay=tcfg["weight_decay"],
                lr_scheduler_type=tcfg["lr_scheduler"], seed=cfg["seed"], report_to="none",
                output_dir=os.path.join(out_dir, "trainer_tmp"), save_strategy="no", max_steps=tcfg["max_steps"] or -1,
                fp16=mcfg["dtype"] == "float16", bf16=mcfg["dtype"] == "bfloat16")


def _build_unsloth(cfg, H, examples, out_dir):
    import torch
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    mcfg, L, c = cfg["model"], cfg["model"]["lora"], CHAT[cfg["model"]["chat_family"]]
    model, tok = FastModel.from_pretrained(model_name=mcfg["name"], max_seq_length=mcfg["max_seq_length"],
                                           full_finetuning=False, dtype=getattr(torch, mcfg["dtype"]))
    # LoRA on attention + MLP of the language part, vision tower untouched (as the author)
    model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                     finetune_attention_modules=True, finetune_mlp_modules=True, r=L["r"],
                                     lora_alpha=L["alpha"], lora_dropout=L["dropout"], bias="none",
                                     random_state=cfg["seed"],
                                     use_gradient_checkpointing="unsloth" if cfg["train"]["gradient_checkpointing"] else False)
    rows = _training_rows(cfg, H, examples, "unsloth")
    tok = get_chat_template(tok, chat_template=c["template"])
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=Dataset.from_list(rows),
                         args=SFTConfig(dataset_text_field="text", **_trainer_args(cfg, out_dir)))
    trainer = train_on_responses_only(trainer, instruction_part=c["us"], response_part=c["ms"])
    return trainer, tok, rows


def _build_hf(cfg, H, examples, out_dir):
    import torch
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, prepare_model_for_kbit_training
    mcfg, L = cfg["model"], cfg["model"]["lora"]
    tok = AutoTokenizer.from_pretrained(mcfg["name"])
    model = AutoModelForCausalLM.from_pretrained(mcfg["name"], torch_dtype=getattr(torch, mcfg["dtype"]), device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_cfg = LoraConfig(r=L["r"], lora_alpha=L["alpha"], lora_dropout=L["dropout"], bias="none",
                          target_modules=L["hf_target_regex"], task_type="CAUSAL_LM")
    rows = _training_rows(cfg, H, examples, "hf")
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=Dataset.from_list(rows), peft_config=peft_cfg,
                         args=SFTConfig(completion_only_loss=True, max_length=mcfg["max_seq_length"],
                                        **_trainer_args(cfg, out_dir)))
    return trainer, tok, rows


def data_stats(cfg, tok, rows):
    """Token length statistics of the training texts (prompt + answer)."""
    tk = getattr(tok, "tokenizer", tok)
    lens = sorted(len(tk("".join(r.values()), add_special_tokens=True)["input_ids"]) for r in rows)
    m, t = cfg["model"], cfg["train"]
    return {"n_train": len(rows), "tok_mean": sum(lens) / len(lens), "tok_p95": lens[int(0.95 * len(lens))],
            "tok_max": lens[-1], "n_over_max_seq": sum(n > m["max_seq_length"] for n in lens),
            "max_seq_length": m["max_seq_length"], "eff_batch": t["per_device_batch_size"] * t["grad_accum"]}


def train(cfg, H, examples, out_dir):
    """Fine-tune and save the adapter to <out_dir>/adapter. Returns the adapter path."""
    backend = cfg["train"]["backend"]
    if backend == "unsloth":
        import unsloth  # noqa: F401  (unsloth must be imported before trl / transformers / peft)
    import torch
    H.set_seed(cfg["seed"])
    trainer, tok, rows = (_build_unsloth if backend == "unsloth" else _build_hf)(cfg, H, examples, out_dir)
    log("example training row:\n" + json.dumps(rows[1], ensure_ascii=False))
    ds = data_stats(cfg, tok, rows)
    with open(os.path.join(out_dir, "data_stats.json"), "w") as f:
        json.dump(ds, f, indent=1)
    log("data_stats", ds)

    t = time.time()
    res = trainer.train()
    stats = {"train_sec": time.time() - t, "steps": res.global_step, "train_loss": res.training_loss,
             "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    with open(os.path.join(out_dir, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    with open(os.path.join(out_dir, "train_log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    adapter = save_adapter(trainer.model, tok, os.path.join(out_dir, "adapter"))
    shutil.rmtree(os.path.join(out_dir, "trainer_tmp"), ignore_errors=True)
    log("training done", stats)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return adapter


def save_adapter(model, tok, path):
    """Save the LoRA adapter. With Gemma-3, unsloth's save_pretrained sometimes writes no weights -> check and save manually."""
    model.save_pretrained(path)
    tok.save_pretrained(path)
    f = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(f) or os.path.getsize(f) < 1_000_000:
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        sd = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()}
        save_file(sd, f)
        log("adapter weights saved manually:", len(sd), "tensors")
    assert os.path.getsize(f) > 1_000_000, "adapter is empty, stopping before spending hours on inference"
    log("adapter", path, round(os.path.getsize(f) / 1e6, 1), "MB")
    return path
