"""Full 43-layer V4-Flash load on GPU + greedy generation from a real prompt.

Uses QuantizedLinear (native FP4/FP8 storage) so the ~149GB checkpoint fits
on a single 189GB HBM GPU. Dequant-to-bf16 happens per-GEMM in forward.

This is the smallest end-to-end proof that the V4 architecture implementation
is correct: given real V4-Flash weights, the model should complete a simple
factual prompt with a sensible continuation.
"""

import json
import os
import time

import torch

from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer
from rtp_llm.models_py.modules.dsv4.weight_loader import load_v4_safetensors


CKPT_DIR = (
    "/home/wangyin.yx/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V4-Flash/snapshots/"
    "6e763230a9d263eca2023f1d4a5ce1bfe126cf48"
)


def build_full_args(cfg):
    return V4Args(
        vocab_size=cfg["vocab_size"], dim=cfg["hidden_size"],
        n_heads=cfg["num_attention_heads"],
        n_layers=cfg["num_hidden_layers"], n_mtp_layers=0,
        q_lora_rank=cfg["q_lora_rank"], head_dim=cfg["head_dim"],
        rope_head_dim=cfg["qk_rope_head_dim"],
        o_groups=cfg["o_groups"], o_lora_rank=cfg["o_lora_rank"],
        window_size=cfg["sliding_window"],
        compress_ratios=cfg["compress_ratios"][:cfg["num_hidden_layers"]],
        rope_theta=cfg["rope_theta"], compress_rope_theta=cfg["compress_rope_theta"],
        rope_factor=cfg["rope_scaling"]["factor"],
        beta_fast=cfg["rope_scaling"]["beta_fast"],
        beta_slow=cfg["rope_scaling"]["beta_slow"],
        original_seq_len=cfg["rope_scaling"]["original_max_position_embeddings"],
        index_n_heads=cfg["index_n_heads"], index_head_dim=cfg["index_head_dim"],
        index_topk=cfg["index_topk"],
        moe_inter_dim=cfg["moe_intermediate_size"], n_routed_experts=256,
        n_shared_experts=cfg["n_shared_experts"],
        n_activated_experts=cfg["num_experts_per_tok"],
        score_func=cfg["scoring_func"], route_scale=cfg["routed_scaling_factor"],
        swiglu_limit=cfg["swiglu_limit"], n_hash_layers=cfg["num_hash_layers"],
        hc_mult=cfg["hc_mult"], hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"],
        hc_eps=cfg["hc_eps"], norm_eps=cfg["rms_norm_eps"],
        max_batch_size=1, max_seq_len=256,
    )


def load_full_model(device: str = "cuda:0"):
    cfg = json.load(open(os.path.join(CKPT_DIR, "config.json")))
    args = build_full_args(cfg)

    print(f"Instantiating full V4-Flash ({args.n_layers} layers, "
          f"{args.n_routed_experts} experts) on meta...", flush=True)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = V4Transformer(args)
    torch.set_default_dtype(torch.float32)

    print(f"Materializing on {device}...", flush=True)
    t0 = time.time()
    model = model.to_empty(device=device)
    # Recompute RoPE cache — `to_empty` leaves meta-constructed complex tensors
    # as uninitialized zeros, which silently breaks position encoding.
    for layer in model.layers:
        layer.attn.reset_rope_cache(device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    print(f"  materialize {time.time() - t0:.1f}s", flush=True)

    print("Loading weights from ckpt (native FP4+FP8+BF16, no dequant)...", flush=True)
    t0 = time.time()
    loaded = load_v4_safetensors(model, CKPT_DIR, dtype=torch.bfloat16, device=device,
                                 strict=False, verbose=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        mem_gb = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  load {time.time() - t0:.1f}s, {len(loaded)} tensors, "
              f"GPU mem: {mem_gb:.1f} GB", flush=True)
    else:
        print(f"  load {time.time() - t0:.1f}s, {len(loaded)} tensors", flush=True)
    return model, cfg


def generate(model: V4Transformer, tokenizer, prompt: str,
             max_new_tokens: int = 20, device: str = "cuda:0") -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    print(f"prompt: {prompt!r}  ids: {input_ids.tolist()[0]}", flush=True)
    generated = []
    start_pos = 0
    cur_ids = input_ids
    with torch.inference_mode():
        # Prefill
        t0 = time.time()
        logits = model(cur_ids, start_pos=start_pos)
        print(f"  prefill ({cur_ids.size(1)} tok): {time.time() - t0:.1f}s", flush=True)
        next_id = logits[0].argmax().item()
        generated.append(next_id)
        start_pos += cur_ids.size(1)

        # Decode
        for step in range(max_new_tokens - 1):
            t0 = time.time()
            cur_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits = model(cur_ids, start_pos=start_pos)
            next_id = logits[0].argmax().item()
            generated.append(next_id)
            start_pos += 1
            if step < 3 or step % 5 == 0:
                print(f"  step {step + 1}: {time.time() - t0:.2f}s  tok={next_id}  "
                      f"text={tokenizer.decode([next_id])!r}", flush=True)
            if next_id == tokenizer.eos_token_id:
                break

    full = tokenizer.decode(generated, skip_special_tokens=True)
    return full


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, cfg = load_full_model(device=device)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)

    prompt = "The capital of France is"
    out = generate(model, tok, prompt, max_new_tokens=20, device=device)
    print("=" * 60)
    print(f"OUTPUT: {prompt}{out}")
    print("=" * 60)
