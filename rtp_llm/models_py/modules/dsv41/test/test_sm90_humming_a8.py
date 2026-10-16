"""A8 wiring/roundtrip gates, explicitly separate from arithmetic acceptance.

Root schedules this target through Bazel and gpu_lock. Native WGMMA numerical
errors against the unchanged .02/.05 gate are recorded, not renamed PASS.
These tests do not authorize production promotion or loosen historical tests.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_a8 import (
    SM90HummingGroupedA8Linear,
    validate_lossless_a8_scales,
)


def test_native_mode_defaults_to_k32_promotion(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import _GroupedLocalExperts
    from rtp_llm.utils.model_weight import W

    monkeypatch.setenv("DSV41_SM90_NATIVE_A8", "1")
    monkeypatch.delenv("DSV41_SM90_A8_PROMOTION_GROUP", raising=False)
    adapter = Mock()
    monkeypatch.setattr(
        "rtp_llm.models_py.modules.dsv41.moe._sm90_humming_a8.SM90HummingGroupedA8Linear",
        adapter,
    )
    weights = {}
    for key in (W.v4_routed_w1_w, W.v4_routed_w3_w, W.v4_routed_w2_w):
        weights[key] = torch.zeros((1, 128, 64), dtype=torch.int8)
    for key in (W.v4_routed_w1_s, W.v4_routed_w3_s, W.v4_routed_w2_s):
        weights[key] = torch.full((1, 128, 4), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    local = _GroupedLocalExperts(SimpleNamespace())
    local.setup_weights(weights)
    assert local.native_a8 and not weights
    assert adapter.call_count == 2
    assert all(call.kwargs == {"promotion_group_size": 32} for call in adapter.call_args_list)


def _promotion_group_size():
    group = int(os.environ.get("DSV41_SM90_A8_PROMOTION_GROUP", "128"))
    assert group in (32, 128), "Only g128/g32 promotion metadata is supported; RTP quantization stays g128"
    return group


def test_a8_scale_guard():
    valid = torch.tensor([[[117, 127]], [[120, 131]]], dtype=torch.uint8)
    validate_lossless_a8_scales(valid.view(torch.float8_e8m0fnu))
    for invalid in ([116, 128], [0, 1], [252, 253], [254, 255]):
        with pytest.raises(ValueError):
            validate_lossless_a8_scales(torch.tensor([[invalid]], dtype=torch.uint8).view(torch.float8_e8m0fnu))


@pytest.fixture
def sm90():
    assert torch.cuda.is_available(), "requires parent-locked CUDA GPU"
    assert torch.cuda.get_device_capability() == (9, 0), "requires SM90"
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def _weights(e=2, n=128, k=512):
    weight = torch.randint(-128, 128, (e, n, k // 2), device="cuda", dtype=torch.int8)
    raw = torch.randint(117, 128, (e, n, k // 32), device="cuda", dtype=torch.uint8)
    raw[..., 0], raw[..., 1] = 117, 127
    return weight, raw.view(torch.float8_e8m0fnu)


def _weight_reference(weight, scales):
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                       device=weight.device, dtype=torch.float32)
    raw = weight.to(torch.int16) & 255
    nibble = torch.stack((raw & 15, raw >> 4), -1).flatten(-2)
    return lut[nibble.long()] * scales.float().repeat_interleave(32, -1)


def _activation_reference(x):
    blocks = x.float().reshape(x.shape[0], -1, 128)
    scale = torch.pow(2., torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448)))
    q = (blocks / scale[..., None]).to(torch.float8_e4m3fn).reshape_as(x)
    return q, scale, (q.float().reshape_as(blocks) * scale[..., None]).reshape_as(x)


def _oracle(x, weight, scales, offsets):
    a = _activation_reference(x)[2]
    w = _weight_reference(weight, scales)
    result = torch.empty((x.shape[0], weight.shape[1]), device=x.device, dtype=torch.bfloat16)
    boundaries = offsets.cpu().tolist()
    for e, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        result[start:end] = (a[start:end] @ w[e].T).bfloat16()
    return result


def _record_error(label, actual, reference):
    error = (actual.float() - reference.float()).abs()
    count = int((error > (.05 + .02 * reference.float().abs())).sum())
    finite = bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all())
    record = dict(label=label, rtol=.02, atol=.05,
                  outside_existing_tolerance=count, elements=actual.numel(),
                  max_abs=float(error.max()), finite=finite,
                  arithmetic_gate="FAIL_NONFINITE" if not finite else ("FAIL" if count else "PASS_THIS_FIXTURE_ONLY"),
                  scope="A8 wiring diagnostic, not model or full-shape numerical acceptance")
    print("SM90_HUMMING_A8_DIAGNOSTIC " + json.dumps(record), flush=True)
    directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if directory:
        (Path(directory) / (label + ".json")).write_text(json.dumps(record, indent=2))
    assert finite, "Nonfinite output/oracle is a hard failure, not an arithmetic diagnostic waiver"


def _bits(left, right):
    assert bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())
    torch.testing.assert_close(left.view(torch.int16), right.view(torch.int16), rtol=0, atol=0)


def _actual_selected_binary(layer, rows, directory):
    from humming.kernel.humming import HummingKernel

    # Audit the same interval table consumed by launch_kernel, not a separately
    # hand-picked or reconstructed tuning configuration.
    table = HummingKernel.prepare_kernels(
        layer.config.to_str(), json.dumps(layer.compute_config), None, layer.weight.device,
    ).reshape(-1, 4).tolist()
    selected = [entry for entry in table if entry[0] < rows <= entry[1]]
    assert len(selected) == 1, (rows, table)
    low, high, kernel_id, num_sms = selected[0]
    kernel = HummingKernel._id2kernel[kernel_id]
    source = Path(kernel.kernel_filename)
    exported = directory / "actual_selected_humming.cubin"
    shutil.copy2(source, exported)
    generated = source.with_name("kernel.cu")
    if generated.is_file():
        shutil.copy2(generated, directory / "actual_selected_humming.cu")
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    result = subprocess.run([cuobjdump, "--dump-sass", str(exported)],
                            capture_output=True, text=True, check=True, timeout=120)
    (directory / "actual_selected_humming.sass").write_text(result.stdout)
    counts = {name: len(re.findall(r"\b" + name + r"\.", result.stdout))
              for name in ("QGMMA", "HGMMA", "HMMA")}
    instructions = list(dict.fromkeys(re.findall(r"\bQGMMA\.[^;\n]+", result.stdout)))
    record = dict(selected_interval=[low, high], kernel_id=kernel_id, num_sms=num_sms,
                  registered_kernel_name=kernel.kernel_name,
                  original_cubin=str(source), exported_cubin=str(exported),
                  cubin_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),
                  sass_counts=counts, qgmma_instructions=instructions,
                  provenance="Grouped A8 adapter actual production registered interval table",
                  layer_config=json.loads(layer.config.to_str()), compute_config=layer.compute_config)
    (directory / "actual_selected_binary.json").write_text(json.dumps(record, indent=2))
    assert counts["QGMMA"] > 0
    assert counts["HGMMA"] == counts["HMMA"] == 0
    assert all(".F32.E4M3.E4M3 " in instruction for instruction in instructions)
    return record


def test_a8_weight_basis_roundtrip(sm90):
    torch.manual_seed(5141)
    weight, scales = _weights(k=128)
    saved_w, saved_s = weight.clone(), scales.view(torch.uint8).clone()
    layer = SM90HummingGroupedA8Linear(weight, scales, promotion_group_size=_promotion_group_size())
    torch.testing.assert_close(weight, saved_w, rtol=0, atol=0)
    torch.testing.assert_close(scales.view(torch.uint8), saved_s, rtol=0, atol=0)
    x = torch.eye(128, device="cuda", dtype=torch.bfloat16).repeat(2, 1)
    offsets = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
    reference = _weight_reference(weight, scales).transpose(1, 2).reshape(256, 128).bfloat16()
    # Exactly one nonzero product per output: no reduction-precision waiver.
    torch.testing.assert_close(layer(x, offsets), reference, rtol=0, atol=0)


@pytest.mark.parametrize("rows,n,k", [
    (17, 128, 512),
    *((rows, n, k) for rows in (1, 16, 512)
      for n, k in ((4608, 5120), (5120, 2304))),
])
def test_a8_payload_contract_repeat_graph_diagnostic(sm90, monkeypatch, rows, n, k):
    from humming.config import MmaType
    from humming.forward import humming_forward
    from humming.tune import get_heuristics_config

    torch.manual_seed(5141)
    output_root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    assert output_root, "Bazel undeclared outputs are mandatory"
    promotion_group = _promotion_group_size()
    label = f"moe_native_a8_promote{promotion_group}_m{rows}_n{n}_k{k}_e2"
    directory = Path(output_root) / label
    directory.mkdir(parents=True, exist_ok=False)
    weight, scales = _weights(n=n, k=k)
    saved_w, saved_s = weight.clone(), scales.view(torch.uint8).clone()
    layer = SM90HummingGroupedA8Linear(weight, scales, promotion_group_size=promotion_group)
    assert layer.config.mma_type == MmaType.WGMMA
    assert str(layer.config.a_dtype) == "float8e4m3"
    assert layer.config.input_scale_group_size == promotion_group
    assert layer.compute_config["use_batch_invariant"]
    assert not layer.compute_config["use_f16_accum"]
    tuning = get_heuristics_config(layer.config, shape_m=rows, device=weight.device,
                                   **layer.compute_config)
    assert not tuning["use_stream_k"]
    x = (torch.randn(rows, k, device="cuda") * 12).bfloat16()
    offsets = torch.tensor([0, rows // 2, rows], device="cuda", dtype=torch.int32)
    saved_x, saved_offsets = x.clone(), offsets.clone()
    q, sf = layer.quantize_input(x)
    qref, sfref, _ = _activation_reference(x)
    torch.testing.assert_close(q.view(torch.uint8), qref.view(torch.uint8), rtol=0, atol=0)
    expected_sf = sfref.repeat_interleave(4, dim=-1) if promotion_group == 32 else sfref
    torch.testing.assert_close(sf, expected_sf, rtol=0, atol=0)

    calls = []
    def observe(config, *args, **kwargs):
        assert config.mma_type == MmaType.WGMMA
        assert kwargs["inputs"].dtype == torch.float8_e4m3fn
        assert kwargs["input_scale"].dtype == torch.float32
        calls.append(True)
        return humming_forward(config, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr("humming.forward.humming_forward", observe)
        first = layer(x, offsets)
    assert calls
    _record_error(label + "_initial", first, _oracle(x, weight, scales, offsets))
    _actual_selected_binary(layer, rows, directory)
    for _ in range(3):
        _bits(layer(x, offsets), first)
    _bits(layer.prepared(q, sf, offsets), first)
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

    original_q, original_packed = sgl_per_token_group_quant_fp8(
        x, group_size=128, eps=1e-4, column_major_scales=True,
        scale_tma_aligned=True, scale_ue8m0=True,
    )
    torch.testing.assert_close(original_q.view(torch.uint8), q.view(torch.uint8), rtol=0, atol=0)
    _bits(layer.forward_quantized(original_q, original_packed, offsets), first)
    torch.testing.assert_close(x, saved_x, rtol=0, atol=0)
    torch.testing.assert_close(offsets, saved_offsets, rtol=0, atol=0)
    # Warm on a side stream and join before capture, as required by Torch.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(x, offsets)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer(x, offsets)
    for index, factor in enumerate((16., .03125)):
        torch.manual_seed(514100 + index)
        x.copy_((torch.randn_like(x) * factor).bfloat16())
        # Include empty-expert layouts and genuinely change metadata even for
        # M=1: route that row to the other expert, then restore its ownership.
        split = (0 if rows // 2 else rows) if index == 0 else (rows if rows // 2 else 0)
        offsets.copy_(torch.tensor([0, split, rows], device="cuda", dtype=torch.int32))
        eager = layer(x, offsets)
        graph.replay()
        torch.cuda.synchronize()
        _bits(captured, eager)
        assert not torch.equal(captured.view(torch.int16), first.view(torch.int16))
        _record_error(f"{label}_changed_{index}", captured, _oracle(x, weight, scales, offsets))
    torch.testing.assert_close(weight, saved_w, rtol=0, atol=0)
    torch.testing.assert_close(scales.view(torch.uint8), saved_s, rtol=0, atol=0)
    print("A8_WIRING_PASS: original arithmetic gate results are separate diagnostics; no promotion", flush=True)
