"""Diagnostic SM90 routed GEMM candidates, never a production precision switch.

The independent oracle and existing rtol=.02/atol=.05 gate are fixed. Candidate
rejection is recorded explicitly; a diagnostic test passing does NOT promote a
rejected arithmetic path. Select shapes with pytest -k, not permanent env flags.

API provenance: installed humming-kernels 0.1.16, upstream
https://github.com/vllm-project/humming (runtime source hashes are printed).
https://github.com/vllm-project/humming/issues/38 documents that smaller
activation scale groups are not a free performance/precision solution.
"""

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_linear import (
    SM90HummingGroupedLinear,
    decode_quantized_activation,
    unpack_activation_scales,
)


def _quantize(x):
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

    return sgl_per_token_group_quant_fp8(
        x.to(torch.bfloat16).contiguous(), group_size=128, eps=1e-4,
        column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True,
    )


def _independent_quantized_input(x):
    blocks = x.float().reshape(x.shape[0], -1, 128)
    scales = torch.pow(
        2.0, torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    payload = (blocks / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return payload.reshape_as(x), scales


def _independent_weight(weight, scales):
    lut = torch.tensor(
        [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32, device=weight.device,
    )
    packed = weight.to(torch.int16) & 255
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    # Only bounded synthetic checkpoint-range exponents are generated below.
    sf = torch.pow(2., scales.view(torch.uint8).float() - 127)
    return lut[codes.long()] * sf.repeat_interleave(32, dim=-1)


def _metrics(actual, reference):
    a, b = actual.float(), reference.float()
    delta = (a - b).abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    outside = (~finite) | (delta > .05 + .02 * b.abs())
    return {
        "elements": a.numel(), "unequal": int((a != b).sum()),
        "outside_existing_tolerance": int(outside.sum()),
        "max_abs": float(delta.max()),
        "rms_abs": float(delta.square().mean().sqrt()),
        "finite": bool(finite.all()),
    }


class _Candidate:
    """Thin adapter over upstream API; quantizer always remains RTP g128."""

    def __init__(self, linear, *, activation_bits=16, batch_invariant=True, tuning=None):
        self.linear = linear
        self.activation_bits = activation_bits
        self.compute = dict(
            gemm_type="grouped_contiguous", use_f16_accum=False,
            use_batch_invariant=batch_invariant,
        )
        self.tuning = tuning

    def prepare_input(self, payload, packed):
        if self.activation_bits == 16:
            return decode_quantized_activation(payload, packed), None
        return payload, unpack_activation_scales(packed, payload.shape[1] // 128)

    def prepared_gemm(self, inputs, scale, offsets):
        from humming.forward import humming_forward

        linear = self.linear
        return humming_forward(
            linear.config, inputs=inputs, input_scale=scale,
            weight=linear.weight, weight_scale=linear.weight_scale,
            weight_scale_2=getattr(linear, "weight_scale_2", None),
            expert_layout=offsets, valid_shape_m=inputs.shape[0],
            locks=linear.locks, compute_config=self.compute, tuning_config=self.tuning,
        )

    def __call__(self, x, offsets):
        payload, packed = _quantize(x)
        inputs, scale = self.prepare_input(payload, packed)
        return self.prepared_gemm(inputs, scale, offsets)


def _w4a8_linear(weight, scales):
    from humming.schema import HummingInputSchema, HummingWeightSchema
    from humming.transform import prepare_layer_config, transform_humming_tensors

    experts, n, half_k = weight.shape
    linear = torch.nn.Module()
    linear.config = prepare_layer_config(
        shape_n=n, shape_k=half_k * 2, num_experts=experts,
        weight_schema=HummingWeightSchema(
            b_dtype="float4e2m1", bs_dtype="float8e8m0",
            weight_scale_group_size=32, weight_scale_type="group",
        ),
        input_schema=HummingInputSchema(
            a_dtype="float8e4m3", input_scale_dtype="float32",
            input_scale_group_size=128,
        ),
        torch_dtype=torch.bfloat16, device=weight.device,
    )
    # Fused E8M0 normalization may modify weights when span > 11. Refuse that
    # separate lossy case; the full checkpoint audit found [117,127].
    exponents = scales.view(torch.uint8).flatten(1).to(torch.int16)
    assert bool(((exponents.amax(1) - exponents.amin(1)) <= 11).all())
    tensors = transform_humming_tensors(linear.config, {
        "weight": weight.contiguous().clone().view(torch.int32),
        "weight_scale": scales.contiguous().clone(),
    })
    for name, tensor in tensors.items():
        linear.register_buffer(name, tensor)
    linear.register_buffer("locks", torch.zeros(1024, device=weight.device, dtype=torch.int32))
    return linear


def _source_metadata():
    from humming.forward import humming_forward
    from humming.transform import prepare_layer_config
    from humming.tune.sm90_h20 import Sm90H20Heuristics

    sources = {}
    for obj in (humming_forward, prepare_layer_config, Sm90H20Heuristics):
        path = Path(inspect.getfile(obj))
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"humming_version": importlib.metadata.version("humming-kernels"),
            "source_sha256": sources, "torch": torch.__version__}


def _k32_pipeline_candidates(linear, m):
    """Only scheduling changes: retain the accepted FP32/K32 accumulation.

    Humming's invariant validation requires no Stream-K and equal block/warp K;
    TMA and warp specialization are not forbidden by its Python/C++ contracts.
    Their equivalence still requires strict bitwise comparison, not inference.
    """
    from humming.tune import get_heuristics_config

    base = get_heuristics_config(
        linear.config, shape_m=m, use_f16_accum=False,
        use_batch_invariant=True, gemm_type="grouped_contiguous", device=linear.weight.device,
    )
    assert base["block_shape"][2] == base["warp_shape"][2] == 32
    tma = dict(
        base, use_stream_k=False, use_tma=True, use_mbarrier=True,
        use_warp_spec=False, use_tma_a=True, use_tma_b=True,
        # Preserve the existing epilogue and scale-load path initially. Enabling
        # TMA globally otherwise defaults these extra paths to True as well.
        use_tma_c=False, use_tma_bs=False, use_tma_bs2=False,
        use_tma_as=False, use_tma_as2=False, use_tma_bzp=False, use_tma_bias=False,
    )
    candidates = {
        "w4a16_baseline": _Candidate(linear),
        "w4a16_k32_tma_only": _Candidate(linear, tuning=tma),
        "w4a16_k32_tma_warp_spec": _Candidate(linear, tuning=dict(tma, use_warp_spec=True)),
    }
    # Four math warps (one WGMMA warpgroup) plus four producer warps. These
    # dimensions satisfy existing Humming shape constraints; actual compilation
    # and register/shared-memory resource feasibility remain measured gates.
    for tile_m, tile_n, stages, ctas in (
        (32, 128, 3, 2), (64, 128, 3, 2),
        (64, 256, 3, 2), (128, 128, 3, 1),
    ):
        config = dict(
            tma, block_shape=(tile_m, tile_n, 32),
            warp_shape=(tile_m, tile_n // 4, 32),
            use_warp_spec=True, num_stages=stages, num_ctas_per_sm=ctas,
        )
        name = f"w4a16_k32_tma_ws_m{tile_m}_n{tile_n}_s{stages}_c{ctas}"
        candidates[name] = _Candidate(linear, tuning=config)
    return candidates


def _run_diagnostic(n, k, m, layout, *, experts=48, bench_repeats=3, suite="arithmetic"):
    import triton.testing
    from humming.tune import get_heuristics_config

    torch.manual_seed(20261015)
    weight = torch.randint(-128, 128, (experts, n, k // 2), device="cuda", dtype=torch.int8)
    scale_bytes = torch.randint(117, 128, (experts, n, k // 32), device="cuda", dtype=torch.uint8)
    scales = scale_bytes.view(torch.float8_e8m0fnu)
    x = (torch.randn(m, k, device="cuda") * 12).bfloat16()
    if layout == "hot":
        # 47 empty experts and one genuinely large expert, not 48 tiny GEMMs.
        counts = [0] * (experts - 1) + [m]
    else:
        counts = [m // experts + int(i < m % experts) for i in range(experts)]
    host_offsets = [0]
    for count in counts:
        host_offsets.append(host_offsets[-1] + count)
    offsets = torch.tensor(host_offsets, device="cuda", dtype=torch.int32)

    q_ref, s_ref = _independent_quantized_input(x)
    q, packed = _quantize(x)
    torch.testing.assert_close(q.view(torch.uint8), q_ref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(unpack_activation_scales(packed, k // 128), s_ref, rtol=0, atol=0)
    qx = (q_ref.float().reshape(m, k // 128, 128) * s_ref.unsqueeze(-1)).reshape(m, k)
    reference = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for expert, count in enumerate(counts):
        if count:
            begin, end = host_offsets[expert:expert + 2]
            w = _independent_weight(weight[expert], scales[expert])
            reference[begin:end] = (qx[begin:end] @ w.T).bfloat16()
    del qx, q_ref, s_ref
    baseline_linear = SM90HummingGroupedLinear(weight, scales)
    if suite == "k32_pipeline":
        candidates = _k32_pipeline_candidates(baseline_linear, m)
    else:
        fp8_linear = _w4a8_linear(weight, scales)
        # Negative control retained for provenance of run 01. H20 heuristics
        # already divide shape_m by E: passing local M double-divides and makes
        # tiles too small. This is explicitly REJECTED, not a local-M proposal.
        local_tuning = get_heuristics_config(
            baseline_linear.config, shape_m=max(counts), use_f16_accum=False,
            use_batch_invariant=True, gemm_type="grouped_contiguous", device=x.device,
        )
        candidates = {
            "w4a16_baseline": _Candidate(baseline_linear),
            "w4a16_rejected_double_divide_m": _Candidate(baseline_linear, tuning=local_tuning),
            "w4a16_noninvariant": _Candidate(baseline_linear, batch_invariant=False),
            "w4a8_g128_invariant": _Candidate(fp8_linear, activation_bits=8),
            "w4a8_g128_noninvariant": _Candidate(fp8_linear, activation_bits=8, batch_invariant=False),
        }
    baseline_output = baseline_linear(x, offsets)
    baseline_metrics = _metrics(baseline_output, reference)
    records = []
    for name, candidate in candidates.items():
        result = candidate(x, offsets)
        error = _metrics(result, reference)
        if name == "w4a16_baseline":
            torch.testing.assert_close(result, baseline_output, rtol=0, atol=0)
        prepared, prepared_scale = candidate.prepare_input(q, packed)
        timings = {"quantize_prepare_gemm_ms": [], "prepared_gemm_ms": []}
        for repeat in range(bench_repeats):
            stages = [
                ("quantize_prepare_gemm_ms", lambda: candidate(x, offsets)),
                ("prepared_gemm_ms", lambda: candidate.prepared_gemm(prepared, prepared_scale, offsets)),
            ]
            if repeat % 2:
                stages.reverse()
            for stage, fn in stages:
                timings[stage].append(triton.testing.do_bench(
                    fn, warmup=20, rep=100, return_mode="median",
                ))
        repeatable = True
        for _ in range(3):
            repeatable &= bool(torch.equal(candidate(x, offsets), result))
        bitwise_baseline = bool(torch.equal(result.view(torch.int16), baseline_output.view(torch.int16)))
        eligible = error["outside_existing_tolerance"] == 0 and repeatable
        if suite == "k32_pipeline":
            eligible &= bitwise_baseline
        if name == "w4a16_rejected_double_divide_m":
            eligible = False
        record = {
            "candidate": name, "suite": suite, "E": experts, "M": m, "N": n, "K": k,
            "layout": layout, "expert_counts": counts,
            "quantizer": "original RTP g128 eps=1e-4; payload/scales bitwise checked",
            "error_vs_independent_reference": error,
            "error_vs_w4a16": _metrics(result, baseline_output),
            "repeatable": repeatable,
            "bitwise_equal_to_baseline": bitwise_baseline,
            "requires_bitwise_baseline": suite == "k32_pipeline",
            "decision": "BOUNDED_SUPPORT" if eligible else "REJECT",
            "scope": "component only; no routed MoE or model correctness acceptance",
            "compute_config": candidate.compute, "tuning_config": candidate.tuning,
            "layer_config": candidate.linear.config.to_str(), **timings,
        }
        print("SM90_MOE_GEMM_CANDIDATE " + json.dumps(record), flush=True)
        records.append(record)
    print("SM90_MOE_GEMM_SOURCE " + json.dumps(_source_metadata()), flush=True)
    # Fail if the accepted baseline regresses. Rejected experimental candidates
    # remain clearly marked diagnostic evidence, never silently promoted.
    assert baseline_metrics["outside_existing_tolerance"] == 0, baseline_metrics
    return records


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires SM90",
)
@pytest.mark.parametrize("projection,n,k", [("w13", 4608, 5120), ("down", 5120, 2304)], ids=["w13", "down"])
@pytest.mark.parametrize("m,layout", [(1, "hot"), (4096, "balanced"), (32768, "hot"), (32768, "balanced")], ids=["decode", "prefill4096", "hot32768", "balanced32768"])
def test_sm90_moe_gemm_candidates(projection, n, k, m, layout):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _run_diagnostic(n, k, m, layout)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires SM90",
)
def test_small_k128_w4a8_rejection_diagnostic():
    """Keep the previously problematic short-K arithmetic visible as a gate."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _run_diagnostic(128, 128, 4, "balanced", experts=8, bench_repeats=1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires SM90",
)
@pytest.mark.parametrize("projection,n,k", [("w13", 4608, 5120), ("down", 5120, 2304)], ids=["w13", "down"])
@pytest.mark.parametrize("m,layout", [(4096, "balanced"), (32768, "hot")], ids=["prefill4096", "hot32768"])
def test_sm90_moe_k32_pipeline_candidates(projection, n, k, m, layout):
    """Strict scheduling-only exploration; no W4A8 or noninvariant candidates."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _run_diagnostic(n, k, m, layout, suite="k32_pipeline")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
