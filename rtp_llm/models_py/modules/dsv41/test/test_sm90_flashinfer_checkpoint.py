"""Actual layer-6 checkpoint weights, synthetic activations, isolated FI gate.

Set DSV41_FLASHINFER_CHECKPOINT=/path/to/DeepSeek-V4.1-Flash.
Only experts [0,48) are read; no model or full checkpoint is loaded. Ordinary
N(0,1) and independently labelled 16x stress inputs retain the existing
rtol=.02/atol=.05 BF16 mathematical gate. Neither is a captured model input.
Graph/eager equality is a separate exact contract, not a replacement for
numerical acceptance. Public FI does not expose activation quant intermediates.
GPU execution belongs to the parent agent.
"""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.test.test_sm90_flashinfer_local import (
    _bits,
    _reference,
    _report_and_check,
    _sha,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("DSV41_FLASHINFER_CHECKPOINT"),
    reason="explicit actual-checkpoint local MoE qualification",
)


def _record(name, payload):
    print(name + " " + json.dumps(payload), flush=True)
    root = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
    root.mkdir(parents=True, exist_ok=True)
    (root / (name.lower() + ".json")).write_text(json.dumps(payload, indent=2) + "\n")


@pytest.fixture(scope="module")
def checkpoint_case():
    from safetensors import safe_open
    from rtp_llm.models_py.modules.dsv41.moe._sm90_flashinfer_moe import (
        Sm90FlashInferLocalExperts,
    )
    from rtp_llm.utils.model_weight import W

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    root = Path(os.environ["DSV41_FLASHINFER_CHECKPOINT"]).resolve(strict=True)
    index_bytes = (root / "model.safetensors.index.json").read_bytes()
    weight_map = json.loads(index_bytes)["weight_map"]
    specs = {
        "w1": (W.v4_routed_w1_w, W.v4_routed_w1_s, 2304, 5120),
        "w3": (W.v4_routed_w3_w, W.v4_routed_w3_s, 2304, 5120),
        "w2": (W.v4_routed_w2_w, W.v4_routed_w2_s, 5120, 2304),
    }
    cpu = {}
    selected = {}
    for part, (wk, sk, n, k) in specs.items():
        cpu[wk] = torch.empty((48, n, k // 2), dtype=torch.int8)
        cpu[sk] = torch.empty((48, n, k // 32), dtype=torch.uint8)
        for expert in range(48):
            for kind, key in (("weight", wk), ("scale", sk)):
                source = f"layers.6.ffn.experts.{expert}.{part}.{kind}"
                assert source in weight_map, f"Missing exact checkpoint key: {source}"
                selected[source] = (key, expert, kind)
    shards = sorted({weight_map[key] for key in selected})
    records = []
    for shard in shards:
        with safe_open(str(root / shard), framework="pt", device="cpu") as reader:
            for source, (key, expert, kind) in selected.items():
                if weight_map[source] != shard:
                    continue
                tensor = reader.get_tensor(source)
                dtype = torch.int8 if kind == "weight" else torch.float8_e8m0fnu
                assert tensor.dtype == dtype, (source, tensor.dtype, dtype)
                assert tensor.shape == cpu[key][expert].shape, (source, tensor.shape)
                cpu[key][expert].copy_(
                    tensor if kind == "weight" else tensor.view(torch.uint8)
                )
                records.append(
                    dict(
                        key=source,
                        shard=shard,
                        shape=list(tensor.shape),
                        dtype=str(dtype),
                    )
                )
    assert len(records) == 288
    ranges = {}
    for part, (_, sk, _, _) in specs.items():
        flat = cpu[sk].flatten(1).to(torch.int16)
        ranges[part] = dict(
            global_min=int(flat.min()),
            global_max=int(flat.max()),
            per_expert_min=flat.amin(1).tolist(),
            per_expert_max=flat.amax(1).tolist(),
        )
        cpu[sk] = cpu[sk].view(torch.float8_e8m0fnu)
    source_hashes = {str(key): _sha(value) for key, value in cpu.items()}
    manifest = dict(
        evidence="SOURCE_FACT: exact safetensors tensors read at execution",
        checkpoint=str(root),
        index_sha256=hashlib.sha256(index_bytes).hexdigest(),
        layer=6,
        expert_range=[0, 48],
        selected_tensors=records,
        stacked_tensor_sha256=source_hashes,
        scale_byte_ranges=ranges,
        activation_scope="synthetic BF16 normal and 16x stress; NOT captured model inputs",
        precision="checkpoint I8 packed MXFP4 + E8M0; FI dynamic row FP8 activation",
    )
    # Emit before setup so a lossless-scale contract failure is diagnosable.
    _record("FLASHINFER_CHECKPOINT_MANIFEST", manifest)
    raw = {key: value.cuda() for key, value in cpu.items()}
    del cpu
    cfg = SimpleNamespace(
        n_routed_experts=384,
        ep_size=8,
        dim=5120,
        moe_inter_dim=2304,
        swiglu_limit=10.0,
    )
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        model = Sm90FlashInferLocalExperts(cfg)
        consumed = dict(raw)
        model.setup_weights(consumed)
        assert not consumed
        assert model.native_output_dtype == torch.bfloat16
        assert (
            model.route_weight_position
            == "gemm2_finalize_after_activation_quantization"
        )
        assert (
            model.activation_quantization == "flashinfer_sm90_humming_dynamic_row_fp8"
        )
        assert source_hashes == {str(key): _sha(value) for key, value in raw.items()}
        model.prepare(8, 8)
        yield SimpleNamespace(model=model, cfg=cfg, raw=raw, manifest=manifest)
        assert source_hashes == {str(key): _sha(value) for key, value in raw.items()}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _inputs(amplitude, phase):
    torch.manual_seed(262610 + phase)
    x = (torch.randn(8, 5120, device="cuda") * amplitude).bfloat16()
    weights = torch.rand(8, 8, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    indices = torch.full((8, 8), -1, device="cuda", dtype=torch.int64)
    if phase == 0:
        # Six total local routes, striped over six receive rows.
        indices[:6, 0] = torch.tensor([0, 47, 7, 39, 15, 31], device="cuda")
    elif phase == 1:
        indices[:, 0] = 47
        indices[:, 1] = 0
    elif phase == 2:
        indices[:, 2] = torch.arange(8, device="cuda") * 5
        indices[-1, 2] = 48  # Positive invalid ID, not just -1 padding.
    elif phase != 3:
        raise ValueError(phase)
    valid = (indices >= 0) & (indices < 48)
    weights.masked_fill_(~valid, float("nan"))
    x.masked_fill_(~valid.any(-1)[:, None], float("nan"))
    return x, weights, indices


def _diff(a, b):
    aa = a.contiguous().view(torch.uint8)
    bb = b.contiguous().view(torch.uint8)
    mismatches = (aa != bb).nonzero()
    return dict(
        differing_bytes=int((aa != bb).sum()),
        first_byte=mismatches[0].tolist() if mismatches.numel() else None,
        max_abs=float((a.float() - b.float()).abs().max()),
        all_finite=bool(torch.isfinite(a).all()),
    )


@pytest.mark.parametrize("amplitude", [1.0, 16.0], ids=["normal", "amplified16"])
@torch.no_grad()
def test_checkpoint_independent_bf16_gate(checkpoint_case, amplitude):
    case = checkpoint_case
    inputs = _inputs(amplitude, 0)
    hashes = tuple(_sha(t) for t in inputs)
    actual = case.model.forward_local(*inputs)
    expected = _reference(case, *inputs)
    original_order = _reference(case, *inputs, rtp_quant_order=True)
    _record(
        f"FLASHINFER_CHECKPOINT_EAGER_A{int(amplitude)}",
        dict(
            evidence="RTP_MEASURED",
            actual_checkpoint_weights=True,
            activation=f"synthetic N(0,1)*{amplitude}; NOT captured model activation",
            output_math_diff=_diff(actual, expected),
            original_rtp_g128_order_diff=_diff(actual, original_order),
            intermediate_quant="NOT_OBSERVABLE: public FI fused API",
            rtol=0.02,
            atol=0.05,
        ),
    )
    assert hashes == tuple(_sha(t) for t in inputs)
    _bits(actual, actual.bfloat16().float())
    _report_and_check(
        actual,
        expected,
        label=f"checkpoint_layer6_normal_or_stress_a{amplitude}",
        rtp_diagnostic=original_order,
    )


@pytest.mark.parametrize("amplitude", [1.0, 16.0], ids=["normal", "amplified16"])
@pytest.mark.parametrize(
    "phase", [1, 2, 3], ids=["boundary_experts", "padding", "empty"]
)
@torch.no_grad()
def test_checkpoint_graph_live_inputs(checkpoint_case, amplitude, phase):
    case = checkpoint_case
    live = _inputs(amplitude, 0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            case.model.forward_local(*live)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = case.model.forward_local(*live)
    for destination, source in zip(live, _inputs(amplitude, phase)):
        destination.copy_(source)
    hashes = tuple(_sha(t) for t in live)
    eager1 = case.model.forward_local(*live)
    eager2 = case.model.forward_local(*live)
    graph.replay()
    first_replay = output.clone()
    graph.replay()
    torch.cuda.synchronize()
    # Record eager-repeat as well as graph-repeat to distinguish native
    # finalize nondeterminism from stale captured inputs; neither is waived.
    _record(
        f"FLASHINFER_CHECKPOINT_GRAPH_A{int(amplitude)}_PHASE{phase}",
        dict(
            evidence="RTP_MEASURED",
            amplitude=amplitude,
            phase=phase,
            eager_repeat=_diff(eager1, eager2),
            graph_repeat=_diff(first_replay, output),
            graph_vs_eager=_diff(output, eager1),
            active_inputs="x, route weights, expert IDs changed in-place",
        ),
    )
    assert hashes == tuple(_sha(t) for t in live)
    _bits(eager1, eager2)
    _bits(first_replay, output)
    _bits(output, eager1)
    if phase == 3:
        _bits(output, torch.zeros_like(output))
    else:
        _report_and_check(
            output,
            _reference(case, *live),
            label=f"checkpoint_layer6_graph_a{amplitude}_phase{phase}",
        )


@torch.no_grad()
def test_checkpoint_row_fp8_source_math_diagnostic(checkpoint_case):
    """Report both amplitudes before enforcing the unchanged BF16 gate."""
    from rtp_llm.models_py.modules.dsv41.test.sm90_flashinfer_reference import (
        diagnostic_metrics,
        flashinfer_row_fp8_reference,
    )

    pending = []
    for amplitude in (1.0, 16.0):
        inputs = _inputs(amplitude, 0)
        actual = checkpoint_case.model.forward_local(*inputs)
        bf16_math = _reference(checkpoint_case, *inputs)
        row_math = flashinfer_row_fp8_reference(checkpoint_case, *inputs)
        _record(
            f"FLASHINFER_CHECKPOINT_ROW_REFERENCE_A{int(amplitude)}",
            dict(
                evidence="RTP_MEASURED",
                synthetic_activation_amplitude=amplitude,
                fi_vs_row_fp8_source_math=diagnostic_metrics(actual, row_math),
                fi_vs_original_bf16_gate=diagnostic_metrics(actual, bf16_math),
                row_fp8_vs_bf16_math=diagnostic_metrics(row_math, bf16_math),
                finalize="SM90 mixed-input disables fused finalize; BF16 FC2 then slot-order FP32 weighted sum then BF16",
                internal_quant_comparison="NOT_OBSERVABLE: no internal FI activation tensors exposed",
            ),
        )
        pending.append((amplitude, actual, bf16_math))
    # Diagnostic agreement never supersedes the original mathematical gate.
    for amplitude, actual, bf16_math in pending:
        _report_and_check(
            actual,
            bf16_math,
            label=f"checkpoint_row_diagnostic_original_gate_a{amplitude}",
        )


@pytest.mark.parametrize("amplitude", [1.0, 16.0], ids=["normal", "amplified16"])
@pytest.mark.parametrize("phase", [0, 1, 2], ids=["six_routes", "boundary", "padding"])
@torch.no_grad()
def test_checkpoint_intended_normalized_a8_gate(checkpoint_case, amplitude, phase):
    """Additional intended-A8 gate, not replacement/waiver of BF16 failures."""
    from rtp_llm.models_py.modules.dsv41.test.sm90_flashinfer_reference import (
        diagnostic_metrics,
        flashinfer_normalized_a8_reference,
        flashinfer_row_fp8_reference,
    )

    inputs = _inputs(amplitude, phase)
    saved = tuple(_sha(t) for t in inputs)
    actual = checkpoint_case.model.forward_local(*inputs)
    normalized = flashinfer_normalized_a8_reference(checkpoint_case, *inputs)
    prior_row_math = flashinfer_row_fp8_reference(checkpoint_case, *inputs)
    original_bf16 = _reference(checkpoint_case, *inputs)
    _record(
        f"FLASHINFER_CHECKPOINT_INTENDED_A8_A{int(amplitude)}_PHASE{phase}",
        dict(
            evidence="RTP_MEASURED",
            amplitude=amplitude,
            phase=phase,
            fi_vs_normalized_intended_a8=diagnostic_metrics(actual, normalized),
            fi_vs_previous_row_math=diagnostic_metrics(actual, prior_row_math),
            fi_vs_original_bf16=diagnostic_metrics(actual, original_bf16),
            reference_boundary="FP8 payloads and independently normalized FP8 weights -> FP32 matmul -> FP32(inverse activation scale * expert residual) -> BF16",
            scope="Additional intended A8 gate .02/.05; original BF16 failure tests untouched; not end-to-end model acceptance",
        ),
    )
    assert saved == tuple(_sha(t) for t in inputs)
    _report_and_check(
        actual,
        normalized,
        label=f"checkpoint_intended_normalized_a8_a{amplitude}_phase{phase}",
    )
