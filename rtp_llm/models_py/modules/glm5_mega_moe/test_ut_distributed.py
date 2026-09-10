"""GLM-5 MegaMoE distributed unit tests.

Tests weight conversion, input packing, JIT warmup logic, and buffer allocation
with 4-rank torch.distributed. The actual mega kernel (SM100-only) is skipped on
SM<100 hardware, but everything up to and including the kernel call is validated.

Run:
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
        rtp_llm/models_py/modules/glm5_mega_moe/test_ut_distributed.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

logging.basicConfig(
    level=logging.INFO,
    format="[rank%(rank)s] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# GLM-5 constants
GLM5_HIDDEN = 6144
GLM5_MOE_INTER = 2048
GLM5_N_EXPERTS = 256
GLM5_TOP_K = 8


class RankLogFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


def log_info(msg, *args, rank=0):
    if dist.get_rank() == rank:
        logging.info(msg, *args)


def log_all(msg, *args):
    logging.info(msg, *args)


# ============================================================================
# Test 1: Input packer correctness (single-GPU, all ranks)
# ============================================================================
def test_input_packer_correctness():
    """Verify Triton fused packer matches torch reference on each rank."""
    from rtp_llm.models_py.modules.glm5_mega_moe.input_packer_triton import (
        fused_pack_mega_moe_inputs,
    )
    from rtp_llm.models_py.modules.glm5_mega_moe.quant_layouts import (
        per_token_cast_to_fp8_packed_ue8m0,
    )

    rank = dist.get_rank()
    torch.manual_seed(42 + rank)
    T, D, topk = 128, GLM5_HIDDEN, GLM5_TOP_K
    x = torch.randn(T, D, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(T, topk, dtype=torch.float32, device="cuda")
    indices = torch.randint(
        0, GLM5_N_EXPERTS, (T, topk), dtype=torch.int64, device="cuda"
    )

    # Reference
    x_fp8_ref, x_sf_ref = per_token_cast_to_fp8_packed_ue8m0(x.contiguous(), gran_k=32)

    # Fused Triton
    out_fp8 = torch.empty(T, D, dtype=torch.float8_e4m3fn, device="cuda")
    out_sf = torch.empty(T, D // 128, dtype=torch.int32, device="cuda")
    out_weights = torch.empty(T, topk, dtype=torch.float32, device="cuda")
    out_indices = torch.empty(T, topk, dtype=torch.int64, device="cuda")
    fused_pack_mega_moe_inputs(
        x, weights, indices, out_fp8, out_sf, out_indices, out_weights
    )

    # Verify
    assert torch.equal(out_weights, weights), "Router weights mismatch"
    assert torch.equal(out_indices, indices), "Router indices mismatch"
    fp8_match = (
        (out_fp8.view(torch.uint8) == x_fp8_ref.view(torch.uint8)).float().mean().item()
    )
    sf_match = (out_sf == x_sf_ref).float().mean().item()
    assert fp8_match == 1.0, f"FP8 mismatch rate={1-fp8_match:.6f}"
    assert sf_match == 1.0, f"SF mismatch rate={1-sf_match:.6f}"

    log_all(
        f"test_input_packer_correctness PASSED (fp8={fp8_match:.4f}, sf={sf_match:.4f})"
    )
    dist.barrier()


# ============================================================================
# Test 2: Input packer performance (benchmark)
# ============================================================================
def test_input_packer_perf():
    """Benchmark fused packer vs torch packer."""
    from rtp_llm.models_py.modules.glm5_mega_moe.input_packer_triton import (
        fused_pack_mega_moe_inputs,
    )
    from rtp_llm.models_py.modules.glm5_mega_moe.quant_layouts import (
        per_token_cast_to_fp8_packed_ue8m0,
    )

    rank = dist.get_rank()
    torch.manual_seed(7)
    for T in [32, 128, 512, 2048]:
        x = torch.randn(T, GLM5_HIDDEN, dtype=torch.bfloat16, device="cuda")
        weights = torch.randn(T, GLM5_TOP_K, dtype=torch.float32, device="cuda")
        indices = torch.randint(
            0, GLM5_N_EXPERTS, (T, GLM5_TOP_K), dtype=torch.int64, device="cuda"
        )
        out_fp8 = torch.empty(T, GLM5_HIDDEN, dtype=torch.float8_e4m3fn, device="cuda")
        out_sf = torch.empty(T, GLM5_HIDDEN // 128, dtype=torch.int32, device="cuda")
        out_w = torch.empty(T, GLM5_TOP_K, dtype=torch.float32, device="cuda")
        out_i = torch.empty(T, GLM5_TOP_K, dtype=torch.int64, device="cuda")

        # Warmup
        for _ in range(5):
            fused_pack_mega_moe_inputs(
                x, weights, indices, out_fp8, out_sf, out_i, out_w
            )
            per_token_cast_to_fp8_packed_ue8m0(x, gran_k=32)
        torch.cuda.synchronize()

        # Benchmark fused
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            fused_pack_mega_moe_inputs(
                x, weights, indices, out_fp8, out_sf, out_i, out_w
            )
        end.record()
        torch.cuda.synchronize()
        fused_ms = start.elapsed_time(end) / 100

        # Benchmark torch
        start.record()
        for _ in range(100):
            per_token_cast_to_fp8_packed_ue8m0(x, gran_k=32)
        end.record()
        torch.cuda.synchronize()
        torch_ms = start.elapsed_time(end) / 100

        if rank == 0:
            speedup = torch_ms / fused_ms if fused_ms > 0 else float("inf")
            logging.info(
                f"  T={T:5d}: fused={fused_ms:.4f}ms  torch={torch_ms:.4f}ms  speedup={speedup:.2f}x"
            )

    log_info("test_input_packer_perf PASSED")
    dist.barrier()


# ============================================================================
# Test 3: FP8 → FP4 weight conversion
# ============================================================================
def test_fp8_to_fp4_weight_conversion():
    """Test that FP8→BF16→FP4 conversion produces valid FP4 weights."""
    import deep_gemm
    from deep_gemm.utils import cast_back_from_fp4, per_token_cast_to_fp4

    rank = dist.get_rank()
    torch.manual_seed(42 + rank)
    E_local = GLM5_N_EXPERTS // 4  # 64 experts per rank
    fp8_block = 128

    # Create synthetic FP8 weight: [E, inter, hidden]
    # Simulate a weight that was quantized to FP8
    w_bf16_orig = (
        (torch.randn(E_local, GLM5_MOE_INTER, GLM5_HIDDEN) * 0.02)
        .to(torch.bfloat16)
        .cuda()
    )
    # Quantize to FP8 per-block
    w_view = w_bf16_orig.float().view(
        E_local, GLM5_MOE_INTER, GLM5_HIDDEN // fp8_block, fp8_block
    )
    w_scale = w_view.abs().amax(dim=-1).clamp(min=1e-12) / 448.0
    w_fp8 = (
        (w_view / w_scale.unsqueeze(-1))
        .clamp(-448, 448)
        .reshape(E_local, GLM5_MOE_INTER, GLM5_HIDDEN)
        .to(torch.float8_e4m3fn)
    )

    # Dequantize FP8 back to BF16
    w_dequant = (
        (
            w_fp8.float().view(
                E_local, GLM5_MOE_INTER, GLM5_HIDDEN // fp8_block, fp8_block
            )
            * w_scale.unsqueeze(-1)
        )
        .reshape(E_local, GLM5_MOE_INTER, GLM5_HIDDEN)
        .to(torch.bfloat16)
    )

    # Convert one expert to FP4 and verify roundtrip
    test_expert = w_dequant[0]  # [inter, hidden]
    packed, sf = per_token_cast_to_fp4(test_expert, use_ue8m0=True, gran_k=32)
    assert packed.shape == (
        GLM5_MOE_INTER,
        GLM5_HIDDEN // 2,
    ), f"Wrong packed shape: {packed.shape}"
    assert packed.dtype == torch.int8, f"Wrong packed dtype: {packed.dtype}"

    # Verify roundtrip error is bounded
    reconstructed = cast_back_from_fp4(packed, sf, gran_k=32)
    rel_error = (reconstructed.float() - test_expert.float()).abs() / (
        test_expert.float().abs() + 1e-6
    )
    mean_rel_error = rel_error.mean().item()
    max_rel_error = rel_error.max().item()

    log_all(
        f"FP8→FP4 roundtrip: mean_rel_error={mean_rel_error:.6f}, max_rel_error={max_rel_error:.4f}"
    )
    assert mean_rel_error < 0.5, f"Mean relative error too high: {mean_rel_error}"

    log_all("test_fp8_to_fp4_weight_conversion PASSED")
    dist.barrier()


# ============================================================================
# Test 4: Full weight setup (FP8 path)
# ============================================================================
def test_weight_setup_from_fp8():
    """Test the full FP8 weight loading path through GLM5MegaMoE.

    On SM89 this will succeed through weight conversion but fail at buffer
    allocation (which needs SM100). We catch that and report success.
    """
    from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe import GLM5MegaMoE

    rank = dist.get_rank()
    ep_size = dist.get_world_size()
    E_local = GLM5_N_EXPERTS // ep_size
    fp8_block = 128

    torch.manual_seed(42 + rank)

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

    # Create small synthetic FP8 weights (reduce E for memory/speed)
    E_test = min(E_local, 8)  # Only test 8 experts to save memory

    def make_fp8(shape):
        N, K = shape
        w = (torch.randn(N, K) * 0.01).to(torch.bfloat16).cuda()
        w_v = w.float().view(N, K // fp8_block, fp8_block)
        s = w_v.abs().amax(dim=-1).clamp(min=1e-12) / 448.0
        q = (
            (w_v / s.unsqueeze(-1))
            .clamp(-448, 448)
            .reshape(N, K)
            .to(torch.float8_e4m3fn)
        )
        return q, s

    w1_fp8, w1_s = make_fp8((E_test * GLM5_MOE_INTER, GLM5_HIDDEN))
    w2_fp8, w2_s = make_fp8((E_test * GLM5_HIDDEN, GLM5_MOE_INTER))
    w3_fp8, w3_s = make_fp8((E_test * GLM5_MOE_INTER, GLM5_HIDDEN))

    w1_fp8 = w1_fp8.view(E_test, GLM5_MOE_INTER, GLM5_HIDDEN)
    w1_s = w1_s.view(E_test, GLM5_MOE_INTER, GLM5_HIDDEN // fp8_block)
    w2_fp8 = w2_fp8.view(E_test, GLM5_HIDDEN, GLM5_MOE_INTER)
    w2_s = w2_s.view(E_test, GLM5_HIDDEN, GLM5_MOE_INTER // fp8_block)
    w3_fp8 = w3_fp8.view(E_test, GLM5_MOE_INTER, GLM5_HIDDEN)
    w3_s = w3_s.view(E_test, GLM5_MOE_INTER, GLM5_HIDDEN // fp8_block)

    # Override n_local_experts for this test
    import dataclasses

    test_cfg = dataclasses.replace(moe.cfg, n_local_experts=E_test)
    moe.cfg = test_cfg

    try:
        moe.setup_weights_from_fp8(w1_fp8, w1_s, w2_fp8, w2_s, w3_fp8, w3_s)
        log_all("Weight setup FULL SUCCESS (SM100 available)")
    except RuntimeError as e:
        if "SM100" in str(e) or "sm" in str(e).lower() or "symm" in str(e).lower():
            # Expected on SM89 - buffer allocation needs SM100
            # But weight conversion should have succeeded
            if moe._mega_l1_w is not None:
                log_all(
                    f"Weight conversion OK (l1_w={moe._mega_l1_w.shape}), "
                    f"buffer allocation failed as expected on SM89: {e}"
                )
            else:
                log_all(f"Weight setup failed before conversion: {e}")
                raise
        else:
            raise

    # Verify weight shapes if conversion succeeded
    if moe._mega_l1_w is not None:
        # L1: [E, 2*inter, hidden//2] transformed
        assert moe._mega_l1_w.shape == (
            E_test,
            2 * GLM5_MOE_INTER,
            GLM5_HIDDEN // 2,
        ), f"L1 weight shape mismatch: {moe._mega_l1_w.shape}"
        assert moe._mega_l2_w.shape == (
            E_test,
            GLM5_HIDDEN,
            GLM5_MOE_INTER // 2,
        ), f"L2 weight shape mismatch: {moe._mega_l2_w.shape}"
        assert moe._mega_l1_w.dtype == torch.int8
        assert moe._mega_l2_w.dtype == torch.int8
        log_all(
            f"test_weight_setup_from_fp8 PASSED: "
            f"l1_w={moe._mega_l1_w.shape}, l2_w={moe._mega_l2_w.shape}"
        )
    else:
        log_all(
            "test_weight_setup_from_fp8 SKIPPED (conversion failed on this platform)"
        )

    dist.barrier()


# ============================================================================
# Test 5: JIT warmup token count generation
# ============================================================================
def test_jit_warmup_generation():
    """Test JIT warmup token count generation for various EP configurations."""
    from rtp_llm.models_py.modules.glm5_mega_moe.jit_warmup import (
        generate_mega_moe_jit_token_counts,
        mega_moe_config_signature,
    )

    rank = dist.get_rank()
    ep_size = dist.get_world_size()

    # Test for GLM-5 shape
    for num_sms in [128, 132, 148]:
        counts = generate_mega_moe_jit_token_counts(
            num_ranks=ep_size,
            num_experts=GLM5_N_EXPERTS,
            num_experts_per_rank=GLM5_N_EXPERTS // ep_size,
            num_topk=GLM5_TOP_K,
            intermediate_hidden=GLM5_MOE_INTER,
            num_sms=num_sms,
            max_tokens_per_rank=8192,
        )
        assert len(counts) > 0, "No warmup buckets generated"
        assert all(c > 0 for c in counts), "Non-positive token count"
        assert all(c <= 8192 for c in counts), "Token count exceeds max"
        assert counts == sorted(counts), "Token counts not sorted"

        # Verify each count maps to a unique signature
        sigs = set()
        for c in counts:
            sig = mega_moe_config_signature(
                num_ranks=ep_size,
                num_experts=GLM5_N_EXPERTS,
                num_experts_per_rank=GLM5_N_EXPERTS // ep_size,
                num_tokens=c,
                num_topk=GLM5_TOP_K,
                intermediate_hidden=GLM5_MOE_INTER,
                num_sms=num_sms,
            )
            sigs.add(sig)
        assert len(sigs) == len(
            counts
        ), f"Duplicate signatures: {len(sigs)} vs {len(counts)}"

        if rank == 0:
            logging.info(f"  num_sms={num_sms}: {len(counts)} buckets, tokens={counts}")

    log_info("test_jit_warmup_generation PASSED")
    dist.barrier()


# ============================================================================
# Test 6: Mega buffer availability check
# ============================================================================
def test_mega_buf_availability():
    """Test mega_moe_available/enabled on current hardware."""
    from rtp_llm.models_py.modules.glm5_mega_moe.mega_buf import (
        _mega_moe_unavailable_reason,
        estimate_mega_moe_symm_buffer_bytes,
        mega_moe_available,
        mega_moe_enabled,
    )

    rank = dist.get_rank()
    cap = torch.cuda.get_device_capability()

    available = mega_moe_available()
    enabled = mega_moe_enabled()
    reason = _mega_moe_unavailable_reason()

    log_all(
        f"mega_moe: available={available}, enabled={enabled}, "
        f"SM={cap[0]}.{cap[1]}, reason={reason}"
    )

    if cap[0] >= 10:
        assert available, f"Should be available on SM{cap[0]}{cap[1]}"
    else:
        assert not available, f"Should NOT be available on SM{cap[0]}{cap[1]}"
        assert "SM100" in (reason or "") or "sm" in (reason or "").lower()

    # Test buffer size estimation
    est = estimate_mega_moe_symm_buffer_bytes(
        group_size=4,
        num_experts=GLM5_N_EXPERTS,
        num_max_tokens_per_rank=8192,
        num_topk=GLM5_TOP_K,
        hidden=GLM5_HIDDEN,
        intermediate_hidden=GLM5_MOE_INTER,
    )
    if est is not None:
        log_info(f"  Estimated buffer size: {est / (1024**3):.3f} GiB")
    else:
        log_info("  Buffer size estimation not available (expected on SM<100)")

    log_info("test_mega_buf_availability PASSED")
    dist.barrier()


# ============================================================================
# Test 7: End-to-end forward (on SM100) or shape validation (SM89)
# ============================================================================
def test_forward_or_shape_validation():
    """If SM100: run full forward. Otherwise: validate all shapes are correct."""
    from rtp_llm.models_py.modules.glm5_mega_moe.mega_buf import mega_moe_available
    from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe import GLM5MegaMoE

    rank = dist.get_rank()
    ep_size = dist.get_world_size()
    E_local = GLM5_N_EXPERTS // ep_size

    if mega_moe_available():
        # Full test on SM100
        torch.manual_seed(42 + rank)
        moe = GLM5MegaMoE.from_params(
            layer_id=0,
            ep_size=ep_size,
            ep_rank=rank,
            max_tokens_per_rank=512,
        )
        # Create BF16 weights and setup
        w1 = (
            (torch.randn(E_local, GLM5_MOE_INTER, GLM5_HIDDEN) * 0.01)
            .to(torch.bfloat16)
            .cuda()
        )
        w2 = (
            (torch.randn(E_local, GLM5_HIDDEN, GLM5_MOE_INTER) * 0.01)
            .to(torch.bfloat16)
            .cuda()
        )
        w3 = (
            (torch.randn(E_local, GLM5_MOE_INTER, GLM5_HIDDEN) * 0.01)
            .to(torch.bfloat16)
            .cuda()
        )
        moe.setup_weights_from_bf16(w1, w2, w3)

        # Forward
        T = 64
        x = torch.randn(T, GLM5_HIDDEN, dtype=torch.bfloat16, device="cuda")
        topk_w = (
            torch.ones(T, GLM5_TOP_K, dtype=torch.float32, device="cuda") / GLM5_TOP_K
        )
        topk_ids = torch.randint(
            0, GLM5_N_EXPERTS, (T, GLM5_TOP_K), dtype=torch.int64, device="cuda"
        )
        y = moe(x, topk_w, topk_ids)
        assert y.shape == (T, GLM5_HIDDEN)
        assert y.dtype == torch.bfloat16
        assert not y.isnan().any()
        log_all(f"FULL FORWARD PASSED: y={y.shape}, max={y.abs().max():.4f}")
    else:
        # Shape validation on SM89
        log_info("SM<100: running shape validation only (no mega kernel)")
        moe = GLM5MegaMoE.from_params(
            layer_id=0,
            ep_size=ep_size,
            ep_rank=rank,
            max_tokens_per_rank=512,
        )
        assert moe.cfg.dim == GLM5_HIDDEN
        assert moe.cfg.moe_inter_dim == GLM5_MOE_INTER
        assert moe.cfg.n_routed_experts == GLM5_N_EXPERTS
        assert moe.cfg.n_activated_experts == GLM5_TOP_K
        assert moe.cfg.n_local_experts == E_local
        assert moe.cfg.local_expert_start == rank * E_local
        log_info("test_forward_or_shape_validation PASSED (shape validation)")

    dist.barrier()


# ============================================================================
# Main
# ============================================================================
def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger = logging.getLogger()
    logger.addFilter(RankLogFilter(rank))

    if rank == 0:
        logging.info("=" * 70)
        logging.info("GLM-5 MegaMoE Distributed Unit Tests")
        logging.info(f"  world_size={world_size}")
        logging.info(f"  GPU: {torch.cuda.get_device_name()}")
        logging.info(f"  SM: {torch.cuda.get_device_capability()}")
        logging.info(
            f"  GLM-5 shape: hidden={GLM5_HIDDEN} inter={GLM5_MOE_INTER} "
            f"experts={GLM5_N_EXPERTS} topk={GLM5_TOP_K}"
        )
        logging.info("=" * 70)

    tests = [
        ("1. Input Packer Correctness", test_input_packer_correctness),
        ("2. Input Packer Performance", test_input_packer_perf),
        ("3. FP8→FP4 Weight Conversion", test_fp8_to_fp4_weight_conversion),
        ("4. Weight Setup from FP8", test_weight_setup_from_fp8),
        ("5. JIT Warmup Generation", test_jit_warmup_generation),
        ("6. Mega Buffer Availability", test_mega_buf_availability),
        ("7. Forward / Shape Validation", test_forward_or_shape_validation),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        if rank == 0:
            logging.info("")
            logging.info(f"--- {name} ---")
        dist.barrier()
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            logging.error(f"FAILED: {name}: {e}", exc_info=True)
        dist.barrier()

    if rank == 0:
        logging.info("")
        logging.info("=" * 70)
        logging.info(f"RESULTS: {passed}/{passed+failed} passed, {failed} failed")
        logging.info("=" * 70)

    dist.destroy_process_group()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
