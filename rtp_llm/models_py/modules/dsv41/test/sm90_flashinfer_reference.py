"""Independent source-math diagnostic for FI's SM90 W4A8 execution order.

This is NOT a native WGMMA accumulation emulator or an acceptance tolerance.
Source contract: computeSafeFP8QuantScale; BF16 FC1 output; FP32 clamped
SwiGLU -> row FP8 directly (no hidden BF16 cast); BF16 FC2 output; original
top-k slot-order FP32 weighted reduction -> BF16. The mixed-input SM90 path
disables fused finalize even when use_fused_finalize=True is requested.
The public FI API exposes no internal quant tensors to directly compare.
"""

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.test.test_sm90_flashinfer_local import _decode_fp4


def row_fp8_roundtrip(values):
    """FP32 448/amax dynamic row quantization, E4M3 RNE+satfinite.

    The zero/nonpositive-amax branch chooses quant multiplier 1. The tiny
    positive denominator floor is 448/FLT_MAX, not an arbitrary epsilon.
    Return the diagnostic payload, inverse multiplier, and reconstructed FP32.
    """
    values = values.float()
    maximum = values.abs().amax(dim=-1, keepdim=True)
    denominator = maximum.clamp_min(448.0 / torch.finfo(torch.float32).max)
    multiplier = torch.where(maximum > 0, 448.0 / denominator, 1.0)
    # torch's bare E4M3 conversion does not implement satfinite for overflow.
    # Explicit clamp preserves NaNs while implementing the finite saturation.
    payload = (values * multiplier).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    inverse = multiplier.reciprocal()
    return payload, inverse, payload.float() * inverse


@torch.no_grad()
def flashinfer_row_fp8_reference(case, x, route_weights, indices):
    """Independent decoded-checkpoint math in FI's activation/routing order.

    Expert residual compensation is algebraically absorbed into the decoded
    original MXFP4 weights. Internal weight preprocessing / WGMMA instruction
    rounding and compiler FMA selection are not represented as bit-exact.
    Invalid rows/slots are skipped, including poisoned padding values.
    """
    from rtp_llm.utils.model_weight import W

    assert x.dtype == torch.bfloat16
    assert route_weights.dtype == torch.float32 and indices.dtype == torch.int64
    assert route_weights.shape == indices.shape
    assert x.shape[0] == indices.shape[0]
    assert not torch.backends.cuda.matmul.allow_tf32, "reference requires FP32 matmul"
    host_indices = indices.cpu().tolist()
    experts = case.model.num_experts
    used = sorted({e for row in host_indices for e in row if 0 <= e < experts})
    route_outputs = torch.zeros(
        (*indices.shape, x.shape[1]), device=x.device, dtype=torch.float32
    )
    for expert in used:
        matches = [
            (row, slot)
            for row, routes in enumerate(host_indices)
            for slot, value in enumerate(routes)
            if value == expert
        ]
        rows = torch.tensor([r for r, _ in matches], device=x.device, dtype=torch.int64)
        slots = torch.tensor(
            [s for _, s in matches], device=x.device, dtype=torch.int64
        )
        decoded = {}
        for name, wk, sk in (
            ("gate", W.v4_routed_w1_w, W.v4_routed_w1_s),
            ("up", W.v4_routed_w3_w, W.v4_routed_w3_s),
            ("down", W.v4_routed_w2_w, W.v4_routed_w2_s),
        ):
            decoded[name] = _decode_fp4(case.raw[wk][expert], case.raw[sk][expert])
        _, _, activation = row_fp8_roundtrip(x[rows])
        gate = (activation @ decoded["gate"].T).bfloat16().float()
        up = (activation @ decoded["up"].T).bfloat16().float()
        gate = gate.clamp_max(case.cfg.swiglu_limit)
        up = up.clamp(-case.cfg.swiglu_limit, case.cfg.swiglu_limit)
        # No route factor and no BF16 rounding before down-input row quant.
        hidden = F.silu(gate) * up
        _, _, down_input = row_fp8_roundtrip(hidden)
        down = (down_input @ decoded["down"].T).bfloat16().float()
        route_outputs[rows, slots] = down
    valid = (indices >= 0) & (indices < experts)
    clean_weights = torch.where(valid, route_weights, 0.0)
    result = torch.zeros_like(x, dtype=torch.float32)
    # Distinct from expert-order accumulation: follow original top-k slots.
    for slot in range(indices.shape[1]):
        result = result + route_outputs[:, slot] * clean_weights[:, slot, None]
    return result.bfloat16().float()


def diagnostic_metrics(actual, expected):
    """Report the existing .02/.05 threshold without using it as a new gate."""
    delta = (actual.float() - expected.float()).abs()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    outside = (~finite) | (delta > (0.05 + 0.02 * expected.abs()))
    coordinates = outside.nonzero()
    first = coordinates[0].tolist() if coordinates.numel() else None
    return dict(
        max_abs=float(delta.max()),
        rms_abs=float(delta.square().mean().sqrt()),
        outside_original_gate=int(outside.sum()),
        all_finite=bool(finite.all()),
        first_outside=first,
        first_actual=float(actual[tuple(first)]) if first is not None else None,
        first_reference=float(expected[tuple(first)]) if first is not None else None,
        rtol=0.02,
        atol=0.05,
        scope="source-math diagnostic only; not WGMMA bitwise oracle or tolerance waiver",
    )


def _normalized_fp8_weight(raw, scales, base_exponent):
    """Decode raw checkpoint bytes, independently of FI's interleaved reader.

    On the supported lossless range delta=0, offset=s-base+1, and the FP4
    pre-MMA conversion represents E2M1(code)*2**(offset-6). The API residual
    is 2**(base-128)*64 = 2**(base-122), not a FP16 folded block scale.
    """
    original = _decode_fp4(raw, scales)
    residual = torch.exp2(base_exponent.float() - 122.0)
    normalized = original / residual
    # The production payload rewrite canonicalizes code8 (-0) to +0.
    normalized = torch.where(normalized == 0, 0.0, normalized)
    fp8 = normalized.to(torch.float8_e4m3fn)
    torch.testing.assert_close(fp8.float(), normalized, rtol=0, atol=0)
    return fp8, residual


@torch.no_grad()
def flashinfer_normalized_a8_reference(case, x, route_weights, indices):
    """Intended A8 arithmetic with normalization BEFORE and scale AFTER GEMM.

    Unlike the earlier algebraically dequantized diagnostic, matrix operands
    here remain the actual FP8 values. The activation inverse scale times
    expert residual is rounded in FP32 BEFORE scaling the GEMM accumulator,
    followed by BF16 storage. This specifies the intended numerical path;
    it still does not emulate reduced-precision native WGMMA accumulation.
    """
    from rtp_llm.utils.model_weight import W

    assert x.dtype == torch.bfloat16 and indices.dtype == torch.int64
    assert route_weights.dtype == torch.float32
    assert route_weights.shape == indices.shape
    assert x.shape[0] == indices.shape[0]
    assert not torch.backends.cuda.matmul.allow_tf32
    experts = case.model.num_experts
    host = indices.cpu().tolist()
    used = sorted({e for row in host for e in row if 0 <= e < experts})
    route_outputs = torch.zeros(
        (*indices.shape, x.shape[1]), device=x.device, dtype=torch.float32
    )
    for expert in used:
        matches = [
            (row, slot)
            for row, routes in enumerate(host)
            for slot, value in enumerate(routes)
            if value == expert
        ]
        rows = torch.tensor([r for r, _ in matches], device=x.device, dtype=torch.int64)
        slots = torch.tensor(
            [s for _, s in matches], device=x.device, dtype=torch.int64
        )
        s1, s3, s2 = (
            case.raw[key][expert]
            for key in (W.v4_routed_w1_s, W.v4_routed_w3_s, W.v4_routed_w2_s)
        )
        # FC1 preprocessing is JOINT up+gate per expert, not two independent
        # projection minima; FC2 gets its own expert-local residual.
        s31 = torch.cat((s3.view(torch.uint8), s1.view(torch.uint8)), dim=0).int()
        s2_int = s2.view(torch.uint8).int()
        for scales in (s31, s2_int):
            assert bool(((scales >= 1) & (scales <= 252)).all())
            assert int(scales.max() - scales.min()) <= 11
        base31 = torch.maximum(s31.min(), s31.max() - 11)
        base2 = torch.maximum(s2_int.min(), s2_int.max() - 11)
        gate_w, residual31 = _normalized_fp8_weight(
            case.raw[W.v4_routed_w1_w][expert], s1, base31
        )
        up_w, _ = _normalized_fp8_weight(case.raw[W.v4_routed_w3_w][expert], s3, base31)
        down_w, residual2 = _normalized_fp8_weight(
            case.raw[W.v4_routed_w2_w][expert], s2, base2
        )
        payload, inverse, _ = row_fp8_roundtrip(x[rows])
        token_scale31 = inverse * residual31
        accum31 = payload.float() @ torch.cat((up_w.float(), gate_w.float()), dim=0).T
        fc1 = (accum31 * token_scale31).bfloat16().float()
        up, gate = fc1.chunk(2, dim=-1)
        gate = gate.clamp_max(case.cfg.swiglu_limit)
        up = up.clamp(-case.cfg.swiglu_limit, case.cfg.swiglu_limit)
        # Match SwigluBiasAdaptor's g*sigmoid(g*alpha)*(up+beta), alpha1/beta0.
        hidden = (gate * torch.sigmoid(gate)) * up
        down_payload, down_inverse, _ = row_fp8_roundtrip(hidden)
        token_scale2 = down_inverse * residual2
        accum2 = down_payload.float() @ down_w.float().T
        down = (accum2 * token_scale2).bfloat16().float()
        route_outputs[rows, slots] = down
    valid = (indices >= 0) & (indices < experts)
    weights = torch.where(valid, route_weights, 0.0)
    result = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(indices.shape[1]):
        result = result + route_outputs[:, slot] * weights[:, slot, None]
    return result.bfloat16().float()
