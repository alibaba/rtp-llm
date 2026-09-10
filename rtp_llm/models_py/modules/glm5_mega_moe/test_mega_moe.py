"""GLM-5 MegaMoE integration test.

Tests the mega kernel module with GLM-5 shapes. Requires:
- SM100 GPU (Blackwell)
- torch.distributed initialized with world_size > 1
- DeepGEMM >= 2.5

Run with torchrun:
    torchrun --nproc_per_node=4 -m rtp_llm.models_py.modules.glm5_mega_moe.test_mega_moe
"""

import argparse
import logging
import os
import sys
import unittest

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# GLM-5 MoE shape constants
GLM5_HIDDEN = 6144
GLM5_MOE_INTER = 2048
GLM5_N_EXPERTS = 256
GLM5_TOP_K = 8


def _check_prerequisites():
    """Return (can_run, reason) tuple."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    cap = torch.cuda.get_device_capability()
    if cap[0] < 10:
        return False, f"SM{cap[0]}{cap[1]} < SM100"
    try:
        import deep_gemm

        if not hasattr(deep_gemm, "fp8_fp4_mega_moe"):
            return False, "deep_gemm.fp8_fp4_mega_moe not found"
    except ImportError:
        return False, "deep_gemm not importable"
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return False, "torch.distributed not initialized"
        if dist.get_world_size() <= 1:
            return False, f"world_size={dist.get_world_size()} <= 1"
    except Exception as e:
        return False, f"dist check failed: {e}"
    return True, ""


def test_mega_moe_bf16_weights():
    """Test mega MoE with synthetic BF16 weights."""
    can_run, reason = _check_prerequisites()
    if not can_run:
        logger.warning("Skipping test: %s", reason)
        return

    import torch.distributed as dist

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ep_size = world_size
    e_local = GLM5_N_EXPERTS // ep_size

    from .mega_moe import GLM5MegaMoE

    moe = GLM5MegaMoE.from_params(
        layer_id=0,
        dim=GLM5_HIDDEN,
        moe_inter_dim=GLM5_MOE_INTER,
        n_routed_experts=GLM5_N_EXPERTS,
        n_activated_experts=GLM5_TOP_K,
        swiglu_limit=0.0,
        ep_size=ep_size,
        ep_rank=rank,
        max_tokens_per_rank=1024,
    )

    # Create synthetic weights
    torch.manual_seed(42 + rank)
    w1 = (
        (torch.randn(e_local, GLM5_MOE_INTER, GLM5_HIDDEN) * 0.01)
        .to(torch.bfloat16)
        .cuda()
    )
    w2 = (
        (torch.randn(e_local, GLM5_HIDDEN, GLM5_MOE_INTER) * 0.01)
        .to(torch.bfloat16)
        .cuda()
    )
    w3 = (
        (torch.randn(e_local, GLM5_MOE_INTER, GLM5_HIDDEN) * 0.01)
        .to(torch.bfloat16)
        .cuda()
    )

    moe.setup_weights_from_bf16(w1, w2, w3)

    # Create synthetic inputs
    num_tokens = 64
    x = torch.randn(num_tokens, GLM5_HIDDEN, dtype=torch.bfloat16, device="cuda")
    topk_weights = (
        torch.ones(num_tokens, GLM5_TOP_K, dtype=torch.float32, device="cuda")
        / GLM5_TOP_K
    )
    topk_ids = torch.randint(
        0, GLM5_N_EXPERTS, (num_tokens, GLM5_TOP_K), dtype=torch.int64, device="cuda"
    )

    # Run forward
    y = moe(x, topk_weights, topk_ids)

    assert y.shape == (
        num_tokens,
        GLM5_HIDDEN,
    ), f"Expected ({num_tokens}, {GLM5_HIDDEN}), got {y.shape}"
    assert y.dtype == torch.bfloat16, f"Expected bfloat16, got {y.dtype}"
    assert not y.isnan().any(), "Output contains NaN"

    if rank == 0:
        logger.info("test_mega_moe_bf16_weights PASSED: output shape=%s", y.shape)
    dist.barrier()


def test_mega_moe_fp8_weights():
    """Test mega MoE with synthetic FP8 per-block weights."""
    can_run, reason = _check_prerequisites()
    if not can_run:
        logger.warning("Skipping test: %s", reason)
        return

    import torch.distributed as dist

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ep_size = world_size
    e_local = GLM5_N_EXPERTS // ep_size

    from .mega_moe import GLM5MegaMoE

    moe = GLM5MegaMoE.from_params(
        layer_id=0,
        dim=GLM5_HIDDEN,
        moe_inter_dim=GLM5_MOE_INTER,
        n_routed_experts=GLM5_N_EXPERTS,
        n_activated_experts=GLM5_TOP_K,
        swiglu_limit=0.0,
        ep_size=ep_size,
        ep_rank=rank,
        max_tokens_per_rank=1024,
    )

    # Create synthetic FP8 weights
    torch.manual_seed(42 + rank)
    fp8_block = 128

    def make_fp8_weight(shape):
        """Create synthetic FP8 per-block quantized weight."""
        N, K = shape
        w_bf16 = (torch.randn(N, K) * 0.01).to(torch.bfloat16).cuda()
        w_view = w_bf16.float().view(N, K // fp8_block, fp8_block)
        scale = w_view.abs().amax(dim=-1).clamp(min=1e-12) / 448.0  # [N, K//128]
        w_quant = (
            (w_view / scale.unsqueeze(-1))
            .clamp(-448, 448)
            .reshape(N, K)
            .to(torch.float8_e4m3fn)
        )
        return w_quant, scale

    w1_fp8, w1_s = make_fp8_weight((e_local * GLM5_MOE_INTER, GLM5_HIDDEN))
    w2_fp8, w2_s = make_fp8_weight((e_local * GLM5_HIDDEN, GLM5_MOE_INTER))
    w3_fp8, w3_s = make_fp8_weight((e_local * GLM5_MOE_INTER, GLM5_HIDDEN))

    # Reshape to [E, N, K]
    w1_fp8 = w1_fp8.view(e_local, GLM5_MOE_INTER, GLM5_HIDDEN)
    w1_s = w1_s.view(e_local, GLM5_MOE_INTER, GLM5_HIDDEN // fp8_block)
    w2_fp8 = w2_fp8.view(e_local, GLM5_HIDDEN, GLM5_MOE_INTER)
    w2_s = w2_s.view(e_local, GLM5_HIDDEN, GLM5_MOE_INTER // fp8_block)
    w3_fp8 = w3_fp8.view(e_local, GLM5_MOE_INTER, GLM5_HIDDEN)
    w3_s = w3_s.view(e_local, GLM5_MOE_INTER, GLM5_HIDDEN // fp8_block)

    moe.setup_weights_from_fp8(w1_fp8, w1_s, w2_fp8, w2_s, w3_fp8, w3_s)

    # Create synthetic inputs
    num_tokens = 32
    x = torch.randn(num_tokens, GLM5_HIDDEN, dtype=torch.bfloat16, device="cuda")
    topk_weights = (
        torch.ones(num_tokens, GLM5_TOP_K, dtype=torch.float32, device="cuda")
        / GLM5_TOP_K
    )
    topk_ids = torch.randint(
        0, GLM5_N_EXPERTS, (num_tokens, GLM5_TOP_K), dtype=torch.int64, device="cuda"
    )

    # Run forward
    y = moe(x, topk_weights, topk_ids)

    assert y.shape == (
        num_tokens,
        GLM5_HIDDEN,
    ), f"Expected ({num_tokens}, {GLM5_HIDDEN}), got {y.shape}"
    assert y.dtype == torch.bfloat16, f"Expected bfloat16, got {y.dtype}"
    assert not y.isnan().any(), "Output contains NaN"

    if rank == 0:
        logger.info("test_mega_moe_fp8_weights PASSED: output shape=%s", y.shape)
    dist.barrier()


def test_input_packer_correctness():
    """Test that fused packer matches torch reference."""
    if not torch.cuda.is_available():
        logger.warning("Skipping: CUDA not available")
        return
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        logger.warning("Skipping: SM%d < SM9", cap[0])
        return

    from .input_packer_triton import fused_pack_mega_moe_inputs
    from .quant_layouts import per_token_cast_to_fp8_packed_ue8m0

    T, D, topk = 64, GLM5_HIDDEN, GLM5_TOP_K
    x = torch.randn(T, D, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(T, topk, dtype=torch.float32, device="cuda")
    indices = torch.randint(
        0, GLM5_N_EXPERTS, (T, topk), dtype=torch.int64, device="cuda"
    )

    # Reference: torch path
    x_fp8_ref, x_sf_ref = per_token_cast_to_fp8_packed_ue8m0(x.contiguous(), gran_k=32)
    weights_ref = weights.clone()
    indices_ref = indices.clone()

    # Fused: triton path
    out_fp8 = torch.empty_like(x_fp8_ref)
    out_sf = torch.empty_like(x_sf_ref)
    out_weights = torch.empty_like(weights)
    out_indices = torch.empty_like(indices)
    fused_pack_mega_moe_inputs(
        x, weights, indices, out_fp8, out_sf, out_indices, out_weights
    )

    # Check router copies are exact
    assert torch.equal(out_weights, weights_ref), "Router weights mismatch"
    assert torch.equal(out_indices, indices_ref), "Router indices mismatch"

    # FP8 values should be close (triton uses slightly different rounding)
    fp8_match = (
        (out_fp8.view(torch.uint8) == x_fp8_ref.view(torch.uint8)).float().mean()
    )
    sf_match = (out_sf == x_sf_ref).float().mean()
    logger.info("Input packer: fp8_match=%.4f sf_match=%.4f", fp8_match, sf_match)
    assert fp8_match > 0.99, f"FP8 match too low: {fp8_match}"
    assert sf_match > 0.99, f"SF match too low: {sf_match}"
    logger.info("test_input_packer_correctness PASSED")


def main():
    """Run tests (requires torchrun for distributed tests)."""
    import torch.distributed as dist

    # Initialize distributed if not already done
    if not dist.is_initialized():
        if "RANK" in os.environ:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        else:
            logger.info(
                "Running non-distributed tests only (use torchrun for full test)"
            )
            test_input_packer_correctness()
            return

    rank = dist.get_rank()
    if rank == 0:
        logger.info("=" * 60)
        logger.info("GLM-5 MegaMoE Test Suite")
        logger.info("  world_size=%d", dist.get_world_size())
        logger.info("  GPU: %s", torch.cuda.get_device_name())
        logger.info("  SM: %s", torch.cuda.get_device_capability())
        logger.info("=" * 60)

    test_input_packer_correctness()
    test_mega_moe_bf16_weights()
    test_mega_moe_fp8_weights()

    if rank == 0:
        logger.info("All tests PASSED!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
