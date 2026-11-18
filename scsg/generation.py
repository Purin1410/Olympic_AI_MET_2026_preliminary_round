"""Text generation with the fine-tuned model (Hugging Face generate; the author used vLLM).

vLLM is not used: with bitsandbytes 4-bit Gemma-3 on a T4 it failed to load in every configuration we tried.
Same temperature and number of samples as the author, but a different sampler, so individual samples differ.
"""
import gc

from common.config import log
from scsg.data import CHAT


def load_model(cfg, adapter):
    """Load the 4-bit base model with the LoRA adapter for inference. Returns (model, tokenizer)."""
    m, icfg = cfg["model"], cfg["infer"]
    if icfg["loader"] == "unsloth":
        import unsloth  # noqa: F401  (must be imported before transformers / peft)
    import torch
    dt = getattr(torch, m["dtype"])
    if icfg["loader"] == "unsloth":
        from unsloth import FastModel
        model, tok = FastModel.from_pretrained(model_name=adapter, max_seq_length=m["max_seq_length"], dtype=dt)
        FastModel.for_inference(model)
        tok = getattr(tok, "tokenizer", tok)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(m["name"])
        model = AutoModelForCausalLM.from_pretrained(m["name"], torch_dtype=dt, device_map={"": 0})
        model = PeftModel.from_pretrained(model, adapter).eval()
    # unsloth's bnb checkpoints declare compute_dtype=bfloat16; a T4 has no bf16 and emulates it ~10x slower
    n4 = 0
    for mod in model.modules():
        if hasattr(mod, "compute_dtype") and type(mod).__name__.startswith("Linear4bit"):
            mod.compute_dtype = dt
            n4 += 1
    log(f"loader={icfg['loader']} adapter={adapter}: set compute_dtype={m['dtype']} on {n4} 4-bit layers")
    tok.padding_side = "left"   # left padding for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


class Generator:
    """generate(prompts, temperature, n) -> for each prompt: (texts[n], n_tokens[n], finish_reasons[n])."""

    def __init__(self, cfg, adapter):
        self.cfg, self.icfg = cfg, cfg["infer"]
        self.model, self.tok = load_model(cfg, adapter)
        self.eos = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids(CHAT[cfg["model"]["chat_family"]]["me"])]

    def _generate(self, prompts, temperature, n):
        import torch
        max_new = self.icfg["max_new_tokens"]
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
        kw = dict(max_new_tokens=max_new, eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id, use_cache=True)
        if temperature > 0:   # pure sampling at the given temperature, no top-p / top-k truncation
            kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0, num_return_sequences=n)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            ids = self.model.generate(**enc, **kw)[:, enc["input_ids"].shape[1]:]
        n_tok = (ids != self.tok.pad_token_id).sum(1).tolist()
        texts = self.tok.batch_decode(ids, skip_special_tokens=True)
        return [(t, k, "length" if k >= max_new else "stop") for t, k in zip(texts, n_tok)]

    def _sample(self, prompts, temperature, n, chunk):
        """len(prompts) x n samples, `chunk` samples per prompt per generate call.
        On OOM: split the prompts in half, then reduce the chunk. The retry is outside the except block,
        because inside it the traceback still holds GPU tensors and empty_cache cannot free them."""
        import torch
        try:
            per_prompt, left = [[] for _ in prompts], n
            while left:
                c = min(chunk, left)
                outs = self._generate(prompts, temperature, c)   # order: prompt 0 x c, prompt 1 x c, ...
                for j in range(len(prompts)):
                    per_prompt[j] += outs[j * c:(j + 1) * c]
                left -= c
            return per_prompt
        except torch.OutOfMemoryError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        if len(prompts) > 1:
            h = len(prompts) // 2
            log(f"OOM with {len(prompts)} prompts -> split {h} + {len(prompts) - h}")
            return self._sample(prompts[:h], temperature, n, chunk) + self._sample(prompts[h:], temperature, n, chunk)
        if chunk > 1:
            log(f"OOM with 1 prompt x {chunk} samples -> {chunk // 2} samples per call")
            return self._sample(prompts, temperature, n, chunk // 2)
        raise torch.OutOfMemoryError("OOM with 1 prompt x 1 sample")

    def generate(self, prompts, temperature, n, on_progress=None):
        greedy = temperature == 0
        batch = self.icfg["greedy_batch"] if greedy else self.icfg["sc_batch_prompts"]
        chunk = 1 if greedy else self.icfg["sc_chunk"]
        results = []
        for b in range(0, len(prompts), batch):
            for samples in self._sample(prompts[b:b + batch], temperature, n, chunk):
                results.append(([s[0] for s in samples], [s[1] for s in samples], [s[2] for s in samples]))
            if on_progress:
                on_progress(len(results))
        return results
