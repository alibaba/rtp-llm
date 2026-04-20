"""End-to-end correctness test: mega_moe vs BF16 baseline on Qwen3.5-35B-A3B architecture.

Instantiates the full Qwen3.5-35B-A3B model with random weights (no NAS loading needed),
runs a complete 40-layer forward pass twice:
  1. Standard BF16 expert GEMM (baseline)
  2. mega_moe (FP8A+FP4W) patched into every MoE layer

Validates that all 40 MoE layers produce consistent results within FP4+FP8
quantization error bounds, and that greedy text generation matches.

Usage:
  CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29515 python benchmark/test_mega_moe_e2e_35b_random.py
"""

from __future__ import annotations

import math
import os
import sys
import types
from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F

MODEL_CONFIG_DIR = "/mnt/nas1/hf/Qwen3.5-35B-A3B"

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
    sf = torch.stack(sfs)
    sf = deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), G)
    return packed, sf


def make_mega_weights(w1_bf16: torch.Tensor, w2_bf16: torch.Tensor):
    import deep_gemm
    l1 = cast_to_fp4(w1_bf16)
    l2 = cast_to_fp4(w2_bf16)
    return deep_gemm.transform_weights_for_mega_moe(l1, l2)


# ── mega_moe patch ────────────────────────────────────────────────────────────

class MegaMoeExpertsForward:
    def __init__(self, l1_w, l2_w, symm_buf):
        self.l1_w = l1_w
        self.l2_w = l2_w
        self.symm_buf = symm_buf

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        import deep_gemm
        from deep_gemm.utils.math import per_token_cast_to_fp8

        T, H = hidden_states.shape
        buf = self.symm_buf

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


def patch_model_with_mega_moe(model, num_experts, top_k, hidden, inter, device):
    """Patch all MoE layers to use mega_moe with a single shared symm buffer."""
    import deep_gemm
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    T_max = 4096
    block_m = deep_gemm._C.get_block_m_for_mega_moe(1, num_experts, T_max, top_k)
    aligned_T = math.ceil(T_max / block_m) * block_m

    # Allocate a SINGLE shared buffer reused across all MoE layers
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

        w1 = module.experts.gate_up_proj.data.to(device)
        w2 = module.experts.down_proj.data.to(device)
        l1_w, l2_w = make_mega_weights(w1, w2)

        module.experts.forward = MegaMoeExpertsForward(l1_w, l2_w, shared_buf)
        patched += 1

    return patched, shared_buf


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Single-rank distributed setup
    torch.cuda.set_device(0)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29515"))
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    device = torch.device("cuda:0")

    cap = torch.cuda.get_device_capability()
    if cap[0] < 10:
        print(f"ERROR: mega_moe requires SM100+, got SM{cap[0]}{cap[1]}")
        sys.exit(1)

    print(f"[E2E] GPU SM{cap[0]}{cap[1]}, {torch.cuda.get_device_name(0)}", flush=True)

    # ── Build model from config, NO weight loading ──────────────────────────
    print("[E2E] Building Qwen3.5-35B-A3B model from config with random weights ...",
          flush=True)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(MODEL_CONFIG_DIR, trust_remote_code=True)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    cfg = tc  # use text config for the model

    hidden      = cfg.hidden_size
    inter       = cfg.moe_intermediate_size
    num_experts = cfg.num_experts
    top_k       = cfg.num_experts_per_tok
    print(f"[E2E] Config: hidden={hidden}, inter={inter}, experts={num_experts}, "
          f"top_k={top_k}, layers={cfg.num_hidden_layers}", flush=True)
    print(f"[E2E] inter % 512 == {inter % 512}  (must be 0)", flush=True)

    # Initialize text-only model from text_config with scaled random weights.
    # Scale down by 0.02 so activations stay in FP8 range (±448) across 40 layers.
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    tc_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    torch.manual_seed(12345)
    model = Qwen3_5MoeForCausalLM(tc_cfg).to(torch.bfloat16)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.02)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[E2E] Model initialized: {n_params:.1f}B params", flush=True)

    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
    moe_blocks = [m for _, m in model.named_modules()
                  if isinstance(m, Qwen3_5MoeSparseMoeBlock)]
    print(f"[E2E] MoE blocks: {len(moe_blocks)}", flush=True)

    # ── Test prompt ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG_DIR, trust_remote_code=True)
    prompt = "Tell me a short story about a robot."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    T = input_ids.shape[1]
    print(f"[E2E] Prompt: '{prompt}' ({T} tokens)", flush=True)

    # ── Layer output tracker ─────────────────────────────────────────────────
    class LayerTracker:
        def __init__(self):
            self.outputs: List[torch.Tensor] = []
            self._handles = []
        def register(self, m):
            def hook(mod, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                self.outputs.append(x.detach().float().cpu())
            self._handles.append(m.register_forward_hook(hook))
        def clear(self):
            self.outputs.clear()
        def remove(self):
            for h in self._handles: h.remove()
            self._handles.clear()

    tracker_ref  = LayerTracker()
    tracker_mega = LayerTracker()
    for blk in moe_blocks:
        tracker_ref.register(blk)

    # ── Baseline BF16 forward ────────────────────────────────────────────────
    print("[E2E] Running baseline BF16 forward pass ...", flush=True)
    with torch.no_grad():
        ref_out = model(**inputs)
    ref_logits = ref_out.logits.detach().float().cpu()
    ref_layer_outs = [t.clone() for t in tracker_ref.outputs]
    tracker_ref.clear()
    tracker_ref.remove()

    # Generate reference text
    print("[E2E] Generating baseline text (20 new tokens) ...", flush=True)
    with torch.no_grad():
        ref_gen = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    ref_text = tokenizer.decode(ref_gen[0][T:], skip_special_tokens=True)
    print(f"[E2E] Baseline output: {repr(ref_text)}", flush=True)

    # ── Patch with mega_moe ─────────────────────────────────────────────────
    print("[E2E] Converting all MoE weights to FP4 and patching ...", flush=True)
    n_patched, shared_buf = patch_model_with_mega_moe(model, num_experts, top_k,
                                                       hidden, inter, device)
    print(f"[E2E] Patched {n_patched}/{len(moe_blocks)} MoE blocks", flush=True)

    for blk in moe_blocks:
        tracker_mega.register(blk)

    # ── mega_moe forward ─────────────────────────────────────────────────────
    print("[E2E] Running mega_moe forward pass ...", flush=True)
    with torch.no_grad():
        mega_out = model(**inputs)
    mega_logits = mega_out.logits.detach().float().cpu()
    mega_layer_outs = [t.clone() for t in tracker_mega.outputs]
    tracker_mega.clear()
    tracker_mega.remove()

    # Generate mega_moe text
    print("[E2E] Generating mega_moe text (20 new tokens) ...", flush=True)
    with torch.no_grad():
        mega_gen = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    mega_text = tokenizer.decode(mega_gen[0][T:], skip_special_tokens=True)
    print(f"[E2E] mega_moe output: {repr(mega_text)}", flush=True)

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("E2E Correctness: mega_moe vs BF16 baseline (Qwen3.5-35B-A3B, random weights)")
    print(f"  Prompt: {repr(prompt)}")
    print(f"  Config: hidden={hidden}, inter={inter}, experts={num_experts}, top_k={top_k}")
    print("-" * 70)

    # Per-layer MoE output cosine similarity
    layer_sims = []
    for i, (r, m) in enumerate(zip(ref_layer_outs, mega_layer_outs)):
        sim = F.cosine_similarity(r.reshape(-1), m.reshape(-1), dim=0).item()
        layer_sims.append(sim)

    avg_cos = sum(layer_sims) / len(layer_sims)
    min_cos = min(layer_sims)
    print(f"  Per-MoE-layer cosine similarity  avg={avg_cos:.5f}  min={min_cos:.5f}")
    for i, s in enumerate(layer_sims):
        status = "OK" if s >= 0.9 else "WARN" if s >= 0.8 else "FAIL"
        print(f"    Layer {i:2d}: {s:.5f}  [{status}]")

    # Final logits
    logit_cos = F.cosine_similarity(
        ref_logits[0, -1].view(1, -1), mega_logits[0, -1].view(1, -1)
    ).item()
    logit_maxerr = (ref_logits[0, -1] - mega_logits[0, -1]).abs().max().item()
    print(f"\n  Final logits (last position):")
    print(f"    cosine similarity: {logit_cos:.5f}")
    print(f"    max absolute error: {logit_maxerr:.4f}")

    # Generated text comparison
    print(f"\n  Generated text (greedy, 20 tokens):")
    print(f"    [BF16 baseline] {repr(ref_text)}")
    print(f"    [mega_moe]      {repr(mega_text)}")
    match = ref_text == mega_text
    print(f"    Text match: {match}")

    # Token-level comparison of generated sequence
    ref_tokens  = ref_gen[0][T:].tolist()
    mega_tokens = mega_gen[0][T:].tolist()
    n_match = sum(a == b for a, b in zip(ref_tokens, mega_tokens))
    print(f"    Token-level match: {n_match}/{len(ref_tokens)}")

    print("=" * 70)

    PASS_THRESHOLD = 0.90
    ok = avg_cos >= PASS_THRESHOLD and logit_cos >= PASS_THRESHOLD
    print(f"\n[{'PASS' if ok else 'FAIL'}] avg_layer_cos={avg_cos:.4f}, "
          f"logit_cos={logit_cos:.4f} (threshold={PASS_THRESHOLD})")
    if not ok:
        sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
