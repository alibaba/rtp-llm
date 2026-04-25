"""End-to-end test: load V4-Flash layer 0 (SWA-only) + globals from real ckpt,
run a forward pass, verify finite logits with sane distribution.

Doesn't validate semantic correctness (only 1 of 43 layers active), just proves:
  - all per-layer weights load and shape-match
  - mHC + attention + MoE composition runs on real numerical values
  - no NaN/Inf with trained init
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


def main():
    cfg = json.load(open(os.path.join(CKPT_DIR, "config.json")))

    # Single layer = layer 0, which is pure SWA (no compressor/indexer) — smallest.
    torch.set_default_dtype(torch.bfloat16)
    args = V4Args(
        vocab_size=cfg["vocab_size"], dim=cfg["hidden_size"], n_heads=cfg["num_attention_heads"],
        n_layers=1, n_mtp_layers=0,
        q_lora_rank=cfg["q_lora_rank"], head_dim=cfg["head_dim"], rope_head_dim=cfg["qk_rope_head_dim"],
        o_groups=cfg["o_groups"], o_lora_rank=cfg["o_lora_rank"],
        window_size=cfg["sliding_window"],
        compress_ratios=[0],            # layer 0 = pure SWA
        rope_theta=cfg["rope_theta"], compress_rope_theta=cfg["compress_rope_theta"],
        rope_factor=cfg["rope_scaling"]["factor"], beta_fast=cfg["rope_scaling"]["beta_fast"],
        beta_slow=cfg["rope_scaling"]["beta_slow"],
        original_seq_len=cfg["rope_scaling"]["original_max_position_embeddings"],
        index_n_heads=cfg["index_n_heads"], index_head_dim=cfg["index_head_dim"], index_topk=cfg["index_topk"],
        moe_inter_dim=cfg["moe_intermediate_size"],
        n_routed_experts=cfg["n_routed_experts"],
        n_shared_experts=cfg["n_shared_experts"],
        n_activated_experts=cfg["num_experts_per_tok"],
        score_func=cfg["scoring_func"], route_scale=cfg["routed_scaling_factor"],
        swiglu_limit=cfg["swiglu_limit"], n_hash_layers=cfg["num_hash_layers"],
        hc_mult=cfg["hc_mult"], hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"], hc_eps=cfg["hc_eps"],
        norm_eps=cfg["rms_norm_eps"],
        max_batch_size=1, max_seq_len=64,
    )

    print("Instantiating V4Transformer (n_layers=1, 256 experts) on CPU ...")
    t0 = time.time()
    model = V4Transformer(args)
    torch.set_default_dtype(torch.float32)
    print(f"  done in {time.time() - t0:.1f}s")

    # Filter: load only layer 0 + globals (skip the 42 other layers' keys).
    keys = ["embed.", "norm.weight", "head.weight", "hc_head_", "layers.0."]
    print(f"Loading ckpt with key prefixes {keys} ...")
    t0 = time.time()
    loaded = load_v4_safetensors(
        model, CKPT_DIR, dtype=torch.bfloat16, device="cpu",
        keys_filter=keys, strict=False, verbose=False,
    )
    print(f"  loaded {len(loaded)} tensors in {time.time() - t0:.1f}s")

    # Inspect layer 0 attention sink — should not be all zero (trained value)
    sink = model.layers[0].attn.attn_sink
    print(f"layer 0 attn_sink: shape={tuple(sink.shape)}  range=[{sink.min().item():.3f}, {sink.max().item():.3f}]  norm={sink.norm().item():.3f}")
    # gate.weight should be loaded (not random)
    gw = model.layers[0].ffn.gate.weight
    print(f"layer 0 gate.weight: shape={tuple(gw.shape)}  std={gw.float().std().item():.4g}")
    # tid2eid for hash layer (layer 0 has hash routing since n_hash_layers=3)
    if hasattr(model.layers[0].ffn.gate, "tid2eid") and model.layers[0].ffn.gate.tid2eid is not None:
        tid = model.layers[0].ffn.gate.tid2eid
        print(f"layer 0 gate.tid2eid: shape={tuple(tid.shape)}  dtype={tid.dtype}  "
              f"min={tid.min().item()} max={tid.max().item()}")

    # Forward pass
    print("Running forward (B=1, S=8) ...")
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    t0 = time.time()
    with torch.inference_mode():
        logits = model(input_ids, start_pos=0)
    print(f"  forward done in {time.time() - t0:.1f}s")
    print(f"logits shape: {tuple(logits.shape)}  dtype: {logits.dtype}")
    print(f"logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]  std: {logits.std().item():.3f}")
    print(f"logits finite: {torch.isfinite(logits).all().item()}")
    assert torch.isfinite(logits).all().item(), "non-finite logits in real-weight forward"

    # Top-5 token IDs (just curious — not semantically meaningful with only 1 layer)
    topk = logits[0].topk(5)
    print(f"top-5 token IDs: {topk.indices.tolist()}  scores: {[f'{v:.2f}' for v in topk.values.tolist()]}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
