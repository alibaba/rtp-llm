"""Unit test: compare fused LayerNorm+FP8GroupQuant vs separate operations."""

import torch
import torch.nn.functional as F
import sys
import os

def test_fused_layernorm_fp8_group_quant():
    """Compare fused kernel output with separate LayerNorm + FP8 quant."""

    # Try to import rtp_llm ops
    try:
        from rtp_llm.ops.compute_ops import rtp_llm_ops
    except ImportError:
        print("SKIP: rtp_llm not installed, cannot run test")
        return

    # Check if the fused op exists
    if not hasattr(rtp_llm_ops, 'fused_add_layernorm_fp8_group_quant'):
        print("SKIP: fused_add_layernorm_fp8_group_quant op not available")
        return

    device = 'cuda'
    dtype = torch.bfloat16
    tokens = 128
    hidden_dim = 768
    group_size = 128
    eps = 1e-5

    torch.manual_seed(42)

    # Create inputs
    input_tensor = torch.randn(tokens, hidden_dim, dtype=dtype, device=device)
    residual_tensor = torch.randn(tokens, hidden_dim, dtype=dtype, device=device)
    gamma = torch.randn(hidden_dim, dtype=dtype, device=device)
    beta = torch.randn(hidden_dim, dtype=dtype, device=device)
    bias = torch.empty(0, dtype=dtype, device=device)  # BERT has no bias

    # ========== Reference: separate LayerNorm + FP8 quant ==========
    ref_input = input_tensor.clone()
    ref_residual = residual_tensor.clone()

    # Step 1: fused_add_layernorm (in-place)
    rtp_llm_ops.fused_add_layernorm(ref_input, ref_residual, bias, gamma, beta, eps)
    # After: ref_input = normed output, ref_residual = pre-norm (input + residual)
    ref_normed = ref_input.clone()
    ref_prenorm = ref_residual.clone()

    # Step 2: per_token_group_quant_fp8
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    ref_fp8, ref_scales = sgl_per_token_group_quant_fp8(
        ref_normed,
        group_size=group_size,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=False,
        scale_ue8m0=False,
    )

    # ========== Test: fused kernel ==========
    fused_input = input_tensor.clone()
    fused_residual = residual_tensor.clone()

    fused_fp8, fused_scales = rtp_llm_ops.fused_add_layernorm_fp8_group_quant(
        fused_input, fused_residual, bias, gamma, beta, eps, group_size,
    )
    # After: fused_input = normed output (in-place), fused_residual = pre-norm (in-place)
    fused_normed = fused_input
    fused_prenorm = fused_residual

    # ========== Compare ==========
    print("=== Comparing fused vs separate ===")

    # 1. Pre-norm residual (should match exactly - same computation)
    prenorm_match = torch.allclose(ref_prenorm, fused_prenorm, atol=0, rtol=0)
    prenorm_maxdiff = (ref_prenorm - fused_prenorm).abs().max().item()
    print(f"Pre-norm residual:  exact_match={prenorm_match}, max_diff={prenorm_maxdiff}")

    # 2. Normed BF16 output
    normed_match = torch.allclose(ref_normed, fused_normed, atol=1e-3, rtol=1e-2)
    normed_maxdiff = (ref_normed.float() - fused_normed.float()).abs().max().item()
    print(f"Normed BF16 output: match(atol=1e-3)={normed_match}, max_diff={normed_maxdiff}")

    # 3. FP8 output (compare as float to see magnitude)
    ref_fp8_f = ref_fp8.float()
    fused_fp8_f = fused_fp8.float()
    fp8_match = torch.equal(ref_fp8.view(torch.uint8), fused_fp8.view(torch.uint8))
    fp8_maxdiff = (ref_fp8_f - fused_fp8_f).abs().max().item()
    fp8_num_diff = (ref_fp8.view(torch.uint8) != fused_fp8.view(torch.uint8)).sum().item()
    total_elems = ref_fp8.numel()
    print(f"FP8 output:         exact_match={fp8_match}, max_diff={fp8_maxdiff}, "
          f"diff_elements={fp8_num_diff}/{total_elems} ({100*fp8_num_diff/total_elems:.2f}%)")

    # 4. Scales
    scales_match = torch.allclose(ref_scales, fused_scales, atol=1e-6, rtol=1e-4)
    scales_maxdiff = (ref_scales - fused_scales).abs().max().item()
    print(f"Group scales:       match(atol=1e-6)={scales_match}, max_diff={scales_maxdiff}")

    # 5. Scale layout check
    print(f"\nRef scales shape={ref_scales.shape}, stride={ref_scales.stride()}")
    print(f"Fused scales shape={fused_scales.shape}, stride={fused_scales.stride()}")

    # Summary
    all_pass = prenorm_match and normed_match and scales_match
    if fp8_num_diff / total_elems > 0.01:  # allow up to 1% FP8 rounding differences
        all_pass = False

    print(f"\n{'PASS' if all_pass else 'FAIL'}: fused LayerNorm+FP8GroupQuant test")
    if not all_pass:
        sys.exit(1)


def test_different_shapes():
    """Test with various token counts and hidden dims."""
    try:
        from rtp_llm.ops.compute_ops import rtp_llm_ops
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    except ImportError:
        print("SKIP: rtp_llm not installed")
        return

    if not hasattr(rtp_llm_ops, 'fused_add_layernorm_fp8_group_quant'):
        print("SKIP: fused op not available")
        return

    device = 'cuda'
    dtype = torch.bfloat16
    group_size = 128
    eps = 1e-5

    test_cases = [
        (1, 256),     # single token, small dim
        (1, 768),     # single token, BERT base
        (32, 768),    # small batch
        (128, 768),   # BERT batch
        (256, 768),   # large batch
        (64, 1024),   # power-of-2 dim
        (128, 1280),  # non-standard dim
    ]

    for tokens, hidden_dim in test_cases:
        if hidden_dim % group_size != 0:
            continue

        torch.manual_seed(42)
        input_t = torch.randn(tokens, hidden_dim, dtype=dtype, device=device)
        residual_t = torch.randn(tokens, hidden_dim, dtype=dtype, device=device)
        gamma = torch.randn(hidden_dim, dtype=dtype, device=device)
        beta = torch.randn(hidden_dim, dtype=dtype, device=device)
        bias = torch.empty(0, dtype=dtype, device=device)

        # Reference
        ref_in = input_t.clone()
        ref_res = residual_t.clone()
        rtp_llm_ops.fused_add_layernorm(ref_in, ref_res, bias, gamma, beta, eps)
        ref_fp8, ref_scales = sgl_per_token_group_quant_fp8(
            ref_in, group_size=group_size, eps=1e-4,
            column_major_scales=True, scale_tma_aligned=False, scale_ue8m0=False)

        # Fused
        fused_in = input_t.clone()
        fused_res = residual_t.clone()
        fused_fp8, fused_scales = rtp_llm_ops.fused_add_layernorm_fp8_group_quant(
            fused_in, fused_res, bias, gamma, beta, eps, group_size)

        normed_ok = torch.allclose(ref_in, fused_in, atol=1e-3, rtol=1e-2)
        scales_ok = torch.allclose(ref_scales, fused_scales, atol=1e-5, rtol=1e-3)
        fp8_diff_pct = (ref_fp8.view(torch.uint8) != fused_fp8.view(torch.uint8)).float().mean().item() * 100

        status = "PASS" if (normed_ok and scales_ok and fp8_diff_pct < 1.0) else "FAIL"
        print(f"  [{status}] tokens={tokens:>4}, hidden={hidden_dim:>5}: "
              f"normed_ok={normed_ok}, scales_ok={scales_ok}, fp8_diff={fp8_diff_pct:.2f}%")

        if status == "FAIL":
            sys.exit(1)

    print("\nAll shape tests passed!")


if __name__ == '__main__':
    print("Test 1: Basic correctness")
    test_fused_layernorm_fp8_group_quant()
    print()
    print("Test 2: Different shapes")
    test_different_shapes()
