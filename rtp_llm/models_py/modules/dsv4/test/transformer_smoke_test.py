"""Smoke test for DeepSeek-V4 standalone Transformer.

Uses a tiny random-weight config to verify forward runs end-to-end without
crashing. Validates the module wiring (mHC + Attention + MoE + per-layer
mock KV cache + hc_head + lm_head) for both prefill and decode steps.
"""

import torch

from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer


def test_tiny_v4_forward():
    torch.manual_seed(0)
    # Set default dtype BEFORE construction so Linear layers come out bf16,
    # while explicitly-fp32 nn.Parameter declarations (mHC, RMSNorm, lm_head)
    # stay fp32. This matches the official inference/model.py contract.
    torch.set_default_dtype(torch.bfloat16)
    args = V4Args(
        vocab_size=128,
        dim=64,
        n_heads=4,
        n_layers=4,
        n_mtp_layers=0,           # skip MTP for the smoke
        q_lora_rank=64,
        head_dim=16,
        rope_head_dim=8,
        o_groups=2,
        o_lora_rank=16,
        window_size=8,
        compress_ratios=[0, 4, 128, 0],   # mix of pure-SWA, CSA, HCA, SWA
        rope_factor=1.0,
        original_seq_len=0,       # disable yarn
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        moe_inter_dim=64,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_hash_layers=0,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        max_batch_size=2,
        max_seq_len=64,
    )
    model = V4Transformer(args)
    torch.set_default_dtype(torch.float32)  # restore for the rest of the process
    # Init ALL nn.Parameter that were declared via torch.empty (not auto-initialized).
    # Linear layers got default Kaiming init; mHC, Gate, QuantizedLinear use torch.empty.
    from rtp_llm.models_py.modules.dsv4.moe import Gate
    from rtp_llm.models_py.modules.dsv4.compressor import Compressor
    from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear
    for module in model.modules():
        if isinstance(module, Gate):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, 'tid2eid') and module.tid2eid is not None:
                torch.nn.init.zeros_(module.tid2eid)
        if isinstance(module, Compressor):
            torch.nn.init.normal_(module.ape, std=0.02)
        if isinstance(module, QuantizedLinear):
            # Zero-init quantized weights for smoke test (finite, bounded outputs)
            module.weight.data.zero_()
            if module.scale is not None:
                module.scale.data.zero_()

    # Init mHC params
    for layer in model.layers:
        torch.nn.init.normal_(layer.attn_hc.fn, std=0.02)
        torch.nn.init.normal_(layer.ffn_hc.fn, std=0.02)
        torch.nn.init.zeros_(layer.attn_hc.base)
        torch.nn.init.zeros_(layer.ffn_hc.base)
        torch.nn.init.ones_(layer.attn_hc.scale)
        torch.nn.init.ones_(layer.ffn_hc.scale)
        # attn_sink small negative for tiny init
        torch.nn.init.constant_(layer.attn.attn_sink, -10.0)

    torch.nn.init.normal_(model.head_hc.fn, std=0.02)
    torch.nn.init.zeros_(model.head_hc.base)
    torch.nn.init.ones_(model.head_hc.scale)
    torch.nn.init.normal_(model.head_weight, std=0.02)

    # PREFILL
    B, S = 1, 32
    input_ids = torch.randint(0, args.vocab_size, (B, S))
    logits = model(input_ids, start_pos=0)
    assert logits.shape == (B, args.vocab_size), f"expected {(B, args.vocab_size)}, got {logits.shape}"
    assert torch.isfinite(logits).all(), "non-finite logits in prefill"
    print(f"PREFILL OK: logits shape {logits.shape}, range [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # DECODE: 1 token at a time
    for step in range(S, S + 4):
        next_id = torch.randint(0, args.vocab_size, (B, 1))
        logits = model(next_id, start_pos=step)
        assert logits.shape == (B, args.vocab_size)
        assert torch.isfinite(logits).all(), f"non-finite logits at decode step {step}"
    print(f"DECODE 4 steps OK")


if __name__ == "__main__":
    test_tiny_v4_forward()
    print("ALL TESTS PASSED")
