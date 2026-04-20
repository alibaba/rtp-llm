"""End-to-end correctness test: mega_moe vs standard BF16 on Qwen3.5-35B-A3B.

Loads the full Qwen3.5-35B-A3B model, runs inference with:
  1. Standard BF16 expert GEMM (baseline)
  2. mega_moe (FP8A+FP4W fused kernel) patched into every MoE layer

Compares logits, per-layer output cosine similarity, and generated text.

Usage (single GPU):
  python benchmark/test_mega_moe_e2e_35b.py

Usage (4-GPU EP=4):
  torchrun --nproc_per_node=4 benchmark/test_mega_moe_e2e_35b.py --ep 4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import types
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

MODEL_DIR_BF16 = "/mnt/nas1/hf/Qwen3.5-35B-A3B"
BLOCK_M_ALIGN  = 128

# ── FP4 helpers ───────────────────────────────────────────────────────────────

def cast_to_fp4(w_bf16: torch.Tensor):
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp4
    G, n, k = w_bf16.shape
    packs, sfs = [], []
    for i in range(G):
        p, s = per_token_cast_to_fp4(w_bf16[i].float(), use_ue8m0=True,
                                      gran_k=32, use_packed_ue8m0=False)
        packs.append(p); sfs.append(s)
    packed = torch.stack(packs)
    sf     = torch.stack(sfs)
    sf     = deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), G)
    return packed, sf


def make_mega_weights(w1_bf16: torch.Tensor, w2_bf16: torch.Tensor):
    import deep_gemm
    l1 = cast_to_fp4(w1_bf16)
    l2 = cast_to_fp4(w2_bf16)
    return deep_gemm.transform_weights_for_mega_moe(l1, l2)


# ── mega_moe patch ────────────────────────────────────────────────────────────

class MegaMoeExpertsForward:
    """Callable that replaces Qwen3_5MoeExperts.forward with mega_moe."""

    def __init__(self, l1_w, l2_w, symm_buf):
        self.l1_w     = l1_w
        self.l2_w     = l2_w
        self.symm_buf = symm_buf

    def __call__(self, hidden_states: torch.Tensor,
                 top_k_index: torch.Tensor,
                 top_k_weights: torch.Tensor) -> torch.Tensor:
        import deep_gemm
        from deep_gemm.utils.math import per_token_cast_to_fp8

        T, H = hidden_states.shape
        buf  = self.symm_buf

        x_fp8, x_sf = per_token_cast_to_fp8(
            hidden_states.float(), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        buf.x[:T].copy_(x_fp8)
        buf.x_sf[:T].copy_(x_sf)
        buf.topk_idx[:T].copy_(top_k_index.int())
        buf.topk_weights[:T].copy_(top_k_weights.float())

        out = torch.empty((T, H), dtype=torch.bfloat16, device=hidden_states.device)
        deep_gemm.fp8_fp4_mega_moe(out, self.l1_w, self.l2_w, buf,
                                    activation_clamp=10.0, fast_math=True)
        return out


def patch_model_with_mega_moe(model, num_experts: int, top_k: int,
                               hidden: int, inter: int,
                               ep_rank: int, ep_size: int,
                               device: torch.device):
    """Replace every Qwen3_5MoeExperts.forward with mega_moe."""
    import deep_gemm

    E_local = num_experts // ep_size
    expert_start = ep_rank * E_local

    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )

    # Allocate ONE shared symm buffer reused across all MoE layers
    T_max = 4096
    block_m = deep_gemm._C.get_block_m_for_mega_moe(ep_size, num_experts, T_max, top_k)
    aligned_T = math.ceil(T_max / block_m) * block_m
    shared_buf = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        num_experts=num_experts,
        num_max_tokens_per_rank=aligned_T,
        num_topk=top_k,
        hidden=hidden,
        intermediate_hidden=inter,
        activation="swiglu",
    )

    patched = 0
    for name, module in model.named_modules():
        if not isinstance(module, Qwen3_5MoeSparseMoeBlock):
            continue

        experts = module.experts
        # gate_up_proj: [E_global, 2I, H]  → take our EP slice
        w1_full = experts.gate_up_proj.data   # [E, 2I, H]
        w2_full = experts.down_proj.data       # [E, H, I]
        w1_local = w1_full[expert_start : expert_start + E_local].to(device)
        w2_local = w2_full[expert_start : expert_start + E_local].to(device)

        l1_w, l2_w = make_mega_weights(w1_local, w2_local)

        fwd = MegaMoeExpertsForward(l1_w, l2_w, shared_buf)
        experts.forward = fwd
        patched += 1

    return patched, shared_buf


# ── Layer-level hooks for cosine similarity tracking ─────────────────────────

class LayerOutputTracker:
    def __init__(self):
        self.outputs: List[torch.Tensor] = []
        self._hooks = []

    def register(self, module):
        def hook(mod, inp, out):
            # out may be tuple; grab the hidden states tensor
            x = out[0] if isinstance(out, tuple) else out
            self.outputs.append(x.detach().float().cpu())
        self._hooks.append(module.register_forward_hook(hook))

    def clear(self):
        self.outputs.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    # Set up single-rank distributed (or multi-rank via torchrun)
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        local_rank = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                        if os.environ.get("CUDA_VISIBLE_DEVICES") else "0")
        torch.cuda.set_device(0)  # after CUDA_VISIBLE_DEVICES remapping, 0 is correct
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29502")
        dist.init_process_group(backend="nccl", world_size=1, rank=0)

    ep_size = dist.get_world_size()
    ep_rank = dist.get_rank()
    device  = torch.device("cuda:0")  # always 0 after CUDA_VISIBLE_DEVICES remapping

    def log(msg):
        if ep_rank == 0:
            print(f"[rank0] {msg}", flush=True)

    log(f"ep_size={ep_size}, device={device}")

    # ── Load model ──────────────────────────────────────────────────────────
    log(f"Loading Qwen3.5-35B-A3B BF16 model from {MODEL_DIR_BF16} ...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_BF16, trust_remote_code=True)

    # Eager CPU load via safetensors.torch.load_file() then move to GPU
    # (avoids lazy mmap NFS page faults; load_file() reads full shards sequentially)
    import safetensors.torch as st, json as _json
    log("Pre-loading all shards into CPU RAM (eager, sequential)...")
    idx_path = os.path.join(MODEL_DIR_BF16, "model.safetensors.index.json")
    wmap = _json.load(open(idx_path))["weight_map"]
    shard_files = sorted(set(wmap.values()))
    state_dict = {}
    for sf in shard_files:
        sf_path = os.path.join(MODEL_DIR_BF16, sf)
        log(f"  loading {sf} ...")
        state_dict.update(st.load_file(sf_path, device="cpu"))
    log(f"All shards loaded into CPU RAM ({len(state_dict)} tensors)")

    # The BF16 model is a VL model: text weights have 'model.language_model.' prefix
    # Remap to match Qwen3_5MoeForCausalLM's expected key format
    prefix = "model.language_model."
    text_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            text_sd["model." + k[len(prefix):]] = v.to(torch.bfloat16)
        elif k.startswith("lm_head."):
            text_sd[k] = v.to(torch.bfloat16)
    del state_dict
    log(f"Extracted {len(text_sd)} text-model tensors")

    from transformers import AutoConfig
    cfg_raw = AutoConfig.from_pretrained(MODEL_DIR_BF16, trust_remote_code=True)
    tc = cfg_raw.text_config if hasattr(cfg_raw, "text_config") else cfg_raw
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    log("Creating text model from config...")
    model = Qwen3_5MoeForCausalLM(tc)
    model = model.to(torch.bfloat16)
    log("Loading state dict into model...")
    missing, unexpected = model.load_state_dict(text_sd, strict=False)
    if missing:
        log(f"WARNING: {len(missing)} missing keys (first 3): {missing[:3]}")
    del text_sd
    log("Moving model to GPU...")
    model = model.to(device)
    model.eval()
    log(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

    # Probe model config
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    hidden      = cfg.hidden_size
    inter       = cfg.moe_intermediate_size
    num_experts = cfg.num_experts
    top_k       = cfg.num_experts_per_tok
    log(f"Config: hidden={hidden}, inter={inter}, experts={num_experts}, top_k={top_k}")
    log(f"inter % 512 == {inter % 512}  (must be 0 for mega_moe)")

    # ── Test prompt ─────────────────────────────────────────────────────────
    prompt = "Tell me a short story about a robot."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    log(f"Input: '{prompt}' ({input_ids.shape[1]} tokens)")

    # ── Track per-MoE-layer outputs ─────────────────────────────────────────
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )
    moe_blocks = [m for _, m in model.named_modules()
                  if isinstance(m, Qwen3_5MoeSparseMoeBlock)]
    log(f"MoE blocks found: {len(moe_blocks)}")

    tracker_ref  = LayerOutputTracker()
    tracker_mega = LayerOutputTracker()
    for blk in moe_blocks:
        tracker_ref.register(blk)

    # ── Baseline forward ────────────────────────────────────────────────────
    log("Running baseline forward pass ...")
    with torch.no_grad():
        ref_out = model(**inputs)
    ref_logits = ref_out.logits.detach().float().cpu()
    ref_layer_outs = [t.clone() for t in tracker_ref.outputs]
    tracker_ref.clear()
    tracker_ref.remove()

    ref_next_token = ref_logits[0, -1].argmax().item()
    log(f"Baseline next token: {repr(tokenizer.decode([ref_next_token]))}")

    # ── Generate baseline text ───────────────────────────────────────────────
    log("Generating baseline text (32 new tokens) ...")
    with torch.no_grad():
        ref_gen = model.generate(input_ids, max_new_tokens=32, do_sample=False)
    ref_text = tokenizer.decode(ref_gen[0][input_ids.shape[1]:], skip_special_tokens=True)
    log(f"Baseline output: {repr(ref_text)}")

    # ── Patch with mega_moe ─────────────────────────────────────────────────
    log("Patching MoE layers with mega_moe ...")
    n_patched, shared_buf = patch_model_with_mega_moe(
        model, num_experts, top_k, hidden, inter,
        ep_rank, ep_size, device
    )
    log(f"Patched {n_patched} MoE blocks")

    for blk in moe_blocks:
        tracker_mega.register(blk)

    # ── mega_moe forward ────────────────────────────────────────────────────
    log("Running mega_moe forward pass ...")
    with torch.no_grad():
        mega_out = model(**inputs)
    mega_logits = mega_out.logits.detach().float().cpu()
    mega_layer_outs = [t.clone() for t in tracker_mega.outputs]
    tracker_mega.clear()
    tracker_mega.remove()

    mega_next_token = mega_logits[0, -1].argmax().item()
    log(f"mega_moe next token: {repr(tokenizer.decode([mega_next_token]))}")

    # ── Generate mega_moe text ───────────────────────────────────────────────
    log("Generating mega_moe text (32 new tokens) ...")
    with torch.no_grad():
        mega_gen = model.generate(input_ids, max_new_tokens=32, do_sample=False)
    mega_text = tokenizer.decode(mega_gen[0][input_ids.shape[1]:], skip_special_tokens=True)
    log(f"mega_moe output: {repr(mega_text)}")

    # ── Comparison ──────────────────────────────────────────────────────────
    if ep_rank != 0:
        dist.destroy_process_group()
        return

    print("\n" + "=" * 70)
    print("E2E Correctness: mega_moe vs BF16 baseline (Qwen3.5-35B-A3B)")
    print(f"  Prompt: {repr(prompt)}")
    print(f"  Model:  hidden={hidden}, inter={inter}, experts={num_experts}, top_k={top_k}")
    print("-" * 70)

    # Per-layer cosine similarity
    if ref_layer_outs and mega_layer_outs:
        layer_sims = []
        for i, (r, m) in enumerate(zip(ref_layer_outs, mega_layer_outs)):
            sim = F.cosine_similarity(r.view(-1), m.view(-1), dim=0).item()
            layer_sims.append(sim)
        avg_cos = sum(layer_sims) / len(layer_sims)
        min_cos = min(layer_sims)
        print(f"  Per-MoE-layer cosine similarity:")
        print(f"    avg={avg_cos:.6f}  min={min_cos:.6f}")
        for i, s in enumerate(layer_sims):
            print(f"    Layer {i:2d}: {s:.6f}")

    # Final logits comparison
    logit_cos = F.cosine_similarity(
        ref_logits[0, -1].view(1, -1), mega_logits[0, -1].view(1, -1)
    ).item()
    logit_maxerr = (ref_logits[0, -1] - mega_logits[0, -1]).abs().max().item()
    print(f"\n  Final logits (last token):")
    print(f"    cosine similarity: {logit_cos:.6f}")
    print(f"    max absolute error: {logit_maxerr:.4f}")

    # Generated text
    print(f"\n  Generated text:")
    print(f"    [BF16 baseline] {repr(ref_text)}")
    print(f"    [mega_moe]      {repr(mega_text)}")
    print(f"    Match: {ref_text == mega_text}")

    # Next-token match
    print(f"\n  Next-token prediction:")
    print(f"    BF16: token {ref_next_token} = {repr(tokenizer.decode([ref_next_token]))}")
    print(f"    mega: token {mega_next_token} = {repr(tokenizer.decode([mega_next_token]))}")
    print(f"    Match: {ref_next_token == mega_next_token}")

    print("=" * 70)
    PASS_THRESHOLD = 0.95
    ok = logit_cos >= PASS_THRESHOLD
    print(f"\n[{'PASS' if ok else 'FAIL'}] logit cosine={logit_cos:.4f} "
          f"{'≥' if ok else '<'} {PASS_THRESHOLD}")
    if not ok:
        sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()

    cap = torch.cuda.get_device_capability()
    if cap[0] < 10:
        print(f"ERROR: mega_moe requires SM100+, got SM{cap[0]}{cap[1]}")
        sys.exit(1)

    run(args)
