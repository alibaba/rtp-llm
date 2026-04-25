"""Smoke test: load V4-Flash embedding + norm + head + hc_head from real ckpt.

Avoids instantiating all 43 layers × 256 experts (~250GB BF16) by using
n_layers=1, n_routed_experts=4 stub config. Validates only the global
weight loading path.
"""

import json
import os

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
    print(f"Config: {cfg['num_hidden_layers']} layers, {cfg['n_routed_experts']} experts, "
          f"hidden={cfg['hidden_size']}, vocab={cfg['vocab_size']}")

    # Tiny stub config: just enough to instantiate global params + 1 layer + 4 experts.
    torch.set_default_dtype(torch.bfloat16)
    args = V4Args(
        vocab_size=cfg["vocab_size"],
        dim=cfg["hidden_size"],
        n_heads=cfg["num_attention_heads"],
        n_layers=1,
        n_mtp_layers=0,
        q_lora_rank=cfg["q_lora_rank"],
        head_dim=cfg["head_dim"],
        rope_head_dim=cfg["qk_rope_head_dim"],
        o_groups=cfg["o_groups"],
        o_lora_rank=cfg["o_lora_rank"],
        window_size=cfg["sliding_window"],
        compress_ratios=[0],   # layer 0 = pure SWA, smallest
        rope_theta=cfg["rope_theta"],
        compress_rope_theta=cfg["compress_rope_theta"],
        rope_factor=cfg["rope_scaling"]["factor"],
        beta_fast=cfg["rope_scaling"]["beta_fast"],
        beta_slow=cfg["rope_scaling"]["beta_slow"],
        original_seq_len=cfg["rope_scaling"]["original_max_position_embeddings"],
        index_n_heads=cfg["index_n_heads"],
        index_head_dim=cfg["index_head_dim"],
        index_topk=cfg["index_topk"],
        moe_inter_dim=cfg["moe_intermediate_size"],
        n_routed_experts=4,        # stub: only load 4 experts to save time
        n_shared_experts=cfg["n_shared_experts"],
        n_activated_experts=cfg["num_experts_per_tok"],
        score_func=cfg["scoring_func"],
        route_scale=cfg["routed_scaling_factor"],
        swiglu_limit=cfg["swiglu_limit"],
        n_hash_layers=cfg["num_hash_layers"],
        hc_mult=cfg["hc_mult"],
        hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"],
        hc_eps=cfg["hc_eps"],
        norm_eps=cfg["rms_norm_eps"],
        max_batch_size=1,
        max_seq_len=512,
    )
    model = V4Transformer(args)
    torch.set_default_dtype(torch.float32)

    # Only load global params (skip layers + experts which won't fit).
    keys = ["embed.", "norm.", "head.", "hc_head_"]
    print(f"Loading keys with prefix in {keys} ...")
    loaded = load_v4_safetensors(
        model, CKPT_DIR, dtype=torch.bfloat16, device="cpu",
        keys_filter=keys, strict=False, verbose=True,
    )
    print(f"Loaded {len(loaded)} tensors:")
    for k, v in loaded.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}  norm={v.float().norm().item():.4g}")

    # Sanity: embed weight should be non-zero, normal-ish range
    e = model.embed.weight
    assert e.abs().sum().item() > 0, "embed weight is all zeros — loader did not populate!"
    print(f"\nEmbed weight stats: shape={tuple(e.shape)}  mean={e.float().mean().item():.5f}  "
          f"std={e.float().std().item():.5f}  range=[{e.float().min().item():.3f}, {e.float().max().item():.3f}]")

    # And head.weight too
    h = model.head.weight
    assert h.abs().sum().item() > 0, "head weight is all zeros!"
    print(f"Head weight stats:  shape={tuple(h.shape)}  mean={h.float().mean().item():.5f}  "
          f"std={h.float().std().item():.5f}")

    # hc_head params
    print(f"hc_head_fn shape={tuple(model.hc_head_fn.shape)}  dtype={model.hc_head_fn.dtype}  "
          f"norm={model.hc_head_fn.float().norm().item():.4g}")
    print(f"hc_head_base shape={tuple(model.hc_head_base.shape)}  values={model.hc_head_base.float().tolist()}")
    print(f"hc_head_scale shape={tuple(model.hc_head_scale.shape)}  values={model.hc_head_scale.float().tolist()}")
    print(f"final norm.weight shape={tuple(model.norm.weight.shape)}  "
          f"norm={model.norm.weight.float().norm().item():.4g}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
