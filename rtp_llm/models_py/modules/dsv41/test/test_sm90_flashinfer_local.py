"""Independent SM90 FlashInfer-open-source local-MoE acceptance gates.

The existing .02/.05 numerical gate is not relaxed. FlashInfer applies route
weights at GEMM2/finalization, unlike RTP's pre-down-quant weighting; numerical
acceptance is therefore not replaced by a claim of bitwise equivalence to RTP.
Public fused APIs expose no intermediate activation quantization tensors. We
report that boundary as NOT_OBSERVABLE instead of fabricating a first diff.
Parent owns GPU/Bazel. No independent Humming backend is called by this test.
"""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    aa = torch.empty(a.numel(), dtype=a.dtype, device=a.device)
    bb = torch.empty(b.numel(), dtype=b.dtype, device=b.device)
    aa.copy_(a.reshape(-1))
    bb.copy_(b.reshape(-1))
    torch.testing.assert_close(
        aa.view(torch.uint8), bb.view(torch.uint8), rtol=0, atol=0
    )


def _sha(value):
    return hashlib.sha256(
        value.contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


@pytest.fixture(scope="module", params=["small", "real"])
def model_case(request):
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    import flashinfer
    import flashinfer_opensource

    from rtp_llm.models_py.modules.dsv41.moe._sm90_flashinfer_moe import (
        Sm90FlashInferLocalExperts,
    )
    from rtp_llm.utils.model_weight import W

    assert flashinfer is not flashinfer_opensource
    assert flashinfer.__version__ == "0.6.9+8c4f4dcf"
    assert flashinfer_opensource.__version__ == "0.7.1"
    small = request.param == "small"
    experts, dim, inter = (8, 128, 128) if small else (48, 5120, 2304)
    cfg = SimpleNamespace(
        n_routed_experts=experts * 8,
        ep_size=8,
        dim=dim,
        moe_inter_dim=inter,
        swiglu_limit=10.0,
    )
    torch.manual_seed(26092)
    raw = {}
    for wk, sk, n, k in (
        (W.v4_routed_w1_w, W.v4_routed_w1_s, inter, dim),
        (W.v4_routed_w3_w, W.v4_routed_w3_s, inter, dim),
        (W.v4_routed_w2_w, W.v4_routed_w2_s, dim, inter),
    ):
        raw[wk] = torch.randint(
            0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
        ).view(torch.int8)
        raw[sk] = torch.randint(
            121, 125, (experts, n, k // 32), device="cuda", dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    hashes = {str(key): _sha(value) for key, value in raw.items()}
    model = Sm90FlashInferLocalExperts(cfg)
    consumed = dict(raw)
    model.setup_weights(consumed)
    assert not consumed
    assert model.native_output_dtype == torch.bfloat16
    assert model.route_weight_position == "gemm2_finalize_after_activation_quantization"
    assert model.activation_quantization == "flashinfer_sm90_humming_dynamic_row_fp8"
    assert hashes == {str(key): _sha(value) for key, value in raw.items()}
    for rows in (1, 4, 8):
        model.prepare(rows, 8)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield SimpleNamespace(
        model=model, raw=raw, cfg=cfg, label=request.param, hashes=hashes
    )
    assert hashes == {str(key): _sha(value) for key, value in raw.items()}
    torch.backends.cuda.matmul.allow_tf32 = previous


def _decode_fp4(raw, scales):
    # Independent nibble/codebook and UE8M0 exponent decode, not FI/Humming
    # preprocessing or either backend's weight reader.
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=raw.device,
    )
    bits = raw.view(torch.uint8).to(torch.int16)
    codes = torch.stack((bits & 15, bits >> 4), dim=-1).flatten(-2)
    scale = torch.exp2(scales.view(torch.uint8).float() - 127)
    return lut[codes.long()] * scale.repeat_interleave(32, dim=-1)


def _g128_roundtrip(x):
    grouped = x.float().reshape(x.shape[0], -1, 128)
    scale = torch.pow(
        2.0, torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    quant = (grouped / scale[..., None]).to(torch.float8_e4m3fn).float()
    return (quant * scale[..., None]).reshape_as(x)


def _reference(case, x, route_weights, indices, *, rtp_quant_order=False):
    """BF16 expert math; optional original RTP G128/order diagnostic only.

    Route weighting is after GEMM2 in the BF16 mathematical reference. The
    RTP diagnostic explicitly weights hidden BF16 before G128 down quant.
    Neither reconstructs unexposed FI dynamic-row internal quantization.
    """
    from rtp_llm.utils.model_weight import W

    total = torch.zeros_like(x, dtype=torch.float32)
    experts = case.model.num_experts
    host_indices = indices.cpu().tolist()
    used = sorted(
        {expert for row in host_indices for expert in row if 0 <= expert < experts}
    )
    for expert in used:
        matches = [
            (row, slot)
            for row, slots in enumerate(host_indices)
            for slot, value in enumerate(slots)
            if value == expert
        ]
        rows = torch.tensor(
            [row for row, _ in matches], device=x.device, dtype=torch.int64
        )
        slots = torch.tensor(
            [slot for _, slot in matches], device=x.device, dtype=torch.int64
        )
        selected = x[rows]
        w1 = _decode_fp4(
            case.raw[W.v4_routed_w1_w][expert], case.raw[W.v4_routed_w1_s][expert]
        )
        w3 = _decode_fp4(
            case.raw[W.v4_routed_w3_w][expert], case.raw[W.v4_routed_w3_s][expert]
        )
        w2 = _decode_fp4(
            case.raw[W.v4_routed_w2_w][expert], case.raw[W.v4_routed_w2_s][expert]
        )
        if rtp_quant_order:
            act = _g128_roundtrip(selected)
            gate = (act @ w1.T).bfloat16().float()
            up = (act @ w3.T).bfloat16().float()
        else:
            gate = F.linear(selected, w1.bfloat16()).float()
            up = F.linear(selected, w3.bfloat16()).float()
        gate = gate.clamp_max(case.cfg.swiglu_limit)
        up = up.clamp(-case.cfg.swiglu_limit, case.cfg.swiglu_limit)
        hidden = F.silu(gate) * up
        factors = route_weights[rows, slots, None]
        if rtp_quant_order:
            hidden = (hidden * factors).bfloat16()
            contribution = (_g128_roundtrip(hidden) @ w2.T).bfloat16().float()
        else:
            contribution = F.linear(hidden.bfloat16(), w2.bfloat16()).float() * factors
        # Explicit token/route accumulation, no production reducer. This is a
        # mathematical tolerance oracle, not a claimed FP32 reduction schedule.
        for i, (row, _) in enumerate(matches):
            total[row] += contribution[i]
    return total.bfloat16().float()


def _inputs(case, rows, phase):
    torch.manual_seed(26093 + rows * 7 + phase)
    x = torch.randn(rows, case.cfg.dim, device="cuda", dtype=torch.bfloat16) * (
        0.125 if phase % 2 == 0 else 1.0
    )
    weights = torch.rand(rows, 8, device="cuda", dtype=torch.float32)
    weights /= weights.sum(-1, keepdim=True)
    indices = (
        torch.arange(rows * 8, device="cuda").reshape(rows, 8) + phase
    ) % case.model.num_experts
    if phase == 1:
        indices.fill_(-1)
        indices[:, 0] = 0
        indices[:, 1] = case.model.num_experts - 1
    elif phase == 2:
        indices[:, -2:] = -1
        if rows > 1:
            indices[-1].fill_(case.model.num_experts + 1)
    elif phase == 3:
        indices.fill_(-1)
    valid = (indices >= 0) & (indices < case.model.num_experts)
    weights.masked_fill_(~valid, float("nan"))
    x.masked_fill_(~valid.any(-1)[:, None], float("nan"))
    return x, weights, indices.long()


def _report_and_check(actual, expected, *, label, rtp_diagnostic=None):
    finite = bool(torch.isfinite(actual).all())
    delta = (actual.float() - expected.float()).abs()
    outside = delta > (0.05 + 0.02 * expected.abs())
    positions = outside.nonzero()
    report = dict(
        label=label,
        rtol=0.02,
        atol=0.05,
        finite=finite,
        outside_gate=int(outside.sum()),
        max_abs=float(delta.max()),
        first_output_difference=positions[0].tolist() if positions.numel() else None,
        internal_activation_quant_first_difference="NOT_OBSERVABLE: public fused API has no intermediate outputs",
        numerical_gate="PASS" if finite and not positions.numel() else "FAIL",
    )
    if rtp_diagnostic is not None:
        report["vs_original_rtp_quant_order_max_abs"] = float(
            (actual - rtp_diagnostic).abs().max()
        )
        report["rtp_diagnostic_scope"] = (
            "independent G128 and weighting order; not native-Humming accumulation emulation"
        )
    print("FLASHINFER_MOE_NUMERICS " + json.dumps(report), flush=True)
    assert finite
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)


@torch.no_grad()
def test_weight_preparation_wiring_and_source_immutability(model_case):
    if model_case.label != "small":
        pytest.skip("same adapter preprocessing wiring covered on small shape")
    from flashinfer_opensource.fused_moe import (
        preprocess_moe_weights_for_sm90_mixed_gemm_humming as prep,
    )

    from rtp_llm.utils.model_weight import W

    c, m = model_case.raw, model_case.model
    fc1 = prep(
        torch.cat((c[W.v4_routed_w3_w], c[W.v4_routed_w1_w]), 1)
        .contiguous()
        .view(torch.uint8),
        torch.cat(
            (
                c[W.v4_routed_w3_s].view(torch.uint8),
                c[W.v4_routed_w1_s].view(torch.uint8),
            ),
            1,
        ).contiguous(),
    )
    fc2 = prep(
        c[W.v4_routed_w2_w].contiguous().view(torch.uint8),
        c[W.v4_routed_w2_s].view(torch.uint8).contiguous(),
    )
    for name, result in (("fc1", fc1), ("fc2", fc2)):
        _bits(getattr(m, name + "_weight"), result[0])
        _bits(getattr(m, name + "_scale"), result[1].view(torch.int32))
        _bits(getattr(m, name + "_residual"), result[2] * 64)
    assert model_case.hashes == {str(key): _sha(value) for key, value in c.items()}


@pytest.mark.parametrize("rows", [1, 8])
@torch.no_grad()
def test_independent_bf16_math_and_original_quant_order(model_case, rows):
    x, weights, indices = _inputs(model_case, rows, 0)
    saved = tuple(_sha(t) for t in (x, weights, indices))
    actual = model_case.model.forward_local(x, weights, indices)
    expected = _reference(model_case, x, weights, indices)
    rtp = _reference(model_case, x, weights, indices, rtp_quant_order=True)
    _report_and_check(
        actual, expected, label=f"{model_case.label}_m{rows}", rtp_diagnostic=rtp
    )
    _bits(actual, actual.bfloat16().float())
    _bits(
        model_case.model.forward_local(
            x, weights, indices, output_dtype=torch.bfloat16
        ),
        actual.bfloat16(),
    )
    assert saved == tuple(_sha(t) for t in (x, weights, indices))


@torch.no_grad()
def test_graph_changed_x_routes_routeweights_and_poison_padding(model_case):
    model = model_case.model
    x, weights, indices = _inputs(model_case, 4, 0)
    for _ in range(3):
        model.forward_local(x, weights, indices)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = model.forward_local(x, weights, indices)
    for phase in (1, 2, 3, 0):
        new = _inputs(model_case, 4, phase)
        for destination, source in zip((x, weights, indices), new):
            destination.copy_(source)
        saved = tuple(_sha(t) for t in (x, weights, indices))
        eager = model.forward_local(x, weights, indices)
        graph.replay()
        torch.cuda.synchronize()
        _bits(captured, eager)
        if phase == 3:
            _bits(captured, torch.zeros_like(captured))
        else:
            _report_and_check(
                captured,
                _reference(model_case, x, weights, indices),
                label=f"{model_case.label}_graph_phase{phase}",
            )
        assert saved == tuple(_sha(t) for t in (x, weights, indices))


@torch.no_grad()
def test_input_contract_empty_and_unprepared_shape(model_case):
    model = model_case.model
    x, weights, indices = _inputs(model_case, 1, 0)
    result = model.forward_local(x[:0], weights[:0], indices[:0])
    assert result.shape == (0, model_case.cfg.dim) and result.dtype == torch.float32
    x3, w3, i3 = _inputs(model_case, 3, 0)
    with pytest.raises(RuntimeError, match="prepare"):
        model.forward_local(x3, w3, i3)
    for bad_x, bad_w, bad_i in (
        (x.float(), weights, indices),
        (x, weights.bfloat16(), indices),
        (x, weights, indices.int()),
    ):
        with pytest.raises(TypeError):
            model.forward_local(bad_x, bad_w, bad_i)


@torch.no_grad()
def test_small_phase1_row_fp8_diagnostic_original_gate(model_case):
    """Reproduce the unchanged failing M4/phase1 inputs, never soften its gate."""
    if model_case.label != "small":
        pytest.skip(
            "exact small02 M4 phase1 diagnostic; real checkpoint has own target"
        )
    from rtp_llm.models_py.modules.dsv41.test.sm90_flashinfer_reference import (
        diagnostic_metrics,
        flashinfer_row_fp8_reference,
    )

    inputs = _inputs(model_case, 4, 1)
    saved = tuple(_sha(t) for t in inputs)
    actual = model_case.model.forward_local(*inputs)
    bf16_math = _reference(model_case, *inputs)
    row_math = flashinfer_row_fp8_reference(model_case, *inputs)
    print(
        "FLASHINFER_SMALL_PHASE1_DIAGNOSTIC "
        + json.dumps(
            dict(
                evidence="RTP_MEASURED",
                fixture="unchanged small02: seed26122 M4 E8 K128 I128 phase1 amplitude1 expert0/7",
                fi_vs_row_fp8_source_math=diagnostic_metrics(actual, row_math),
                fi_vs_original_bf16_gate=diagnostic_metrics(actual, bf16_math),
                row_fp8_vs_bf16_math=diagnostic_metrics(row_math, bf16_math),
                internal_quant_comparison="NOT_OBSERVABLE: public FI fused API",
                gate_scope="Original .02/.05 BF16 mathematical acceptance, not waived by row-reference agreement",
            )
        ),
        flush=True,
    )
    assert saved == tuple(_sha(t) for t in inputs)
    _report_and_check(actual, bf16_math, label="small_phase1_original_gate_diagnostic")
