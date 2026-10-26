"""Native G32 DeepGEMM acceptance; retain the independent FP32 tolerance."""

import gc
import json
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from rtp_llm.models_py.modules.dsv41.sm90_deepgemm_dense import (
    Sm90DeepGEMMMXFP8Linear,
    select_sm90_dense_backend,
)
from rtp_llm.models_py.modules.dsv41.sm90_dense import Sm90MXFP8Linear
from rtp_llm.models_py.modules.dsv41.test.sm90_dense_reference import (
    SHAPES,
    _metrics,
    _oracle,
)
from rtp_llm.models_py.modules.dsv41.utils import (
    V41MXFP8Linear,
    release_prepared_sm90_checkpoint_weights,
)


@pytest.fixture(autouse=True)
def _native_backend(monkeypatch):
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    monkeypatch.delenv("DSV41_SM90_DENSE_BACKEND", raising=False)
    monkeypatch.delenv("DSV41_SM90_DENSE_BACKEND_MAP", raising=False)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def test_default_backend_is_native_deepgemm():
    assert select_sm90_dense_backend(5120, 8192) == "deepgemm"


def _linear(n, k):
    torch.manual_seed(5141)
    w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    s = torch.pow(
        2.0, torch.randint(-3, 3, ((n + 31) // 32, k // 32), device="cuda").float()
    )
    s = s.to(torch.float8_e8m0fnu)
    return V41MXFP8Linear(w, s), w, s


def test_policy_is_static_per_layer_without_humming_repack(monkeypatch):
    monkeypatch.setenv("DSV41_SM90_DENSE_BACKEND_MAP", '{"256x512":"deepgemm"}')
    with patch("rtp_llm.models_py.modules.dsv41.sm90_dense.Sm90MXFP8Linear") as humming:
        linear, w, s = _linear(256, 512)
        humming.assert_not_called()
    assert type(linear._sm90_linear) is Sm90DeepGEMMMXFP8Linear
    assert linear.weight.data_ptr() == w.data_ptr()
    assert linear._sm90_linear.weight_scale.shape == (8, 16)
    assert linear._sm90_linear.weight_scale.dtype == torch.float32
    torch.testing.assert_close(
        linear._sm90_linear.weight_scale, s.float(), rtol=0, atol=0
    )
    loader = SimpleNamespace(weights=[{"used": w}], global_weights={"scale": s})
    assert release_prepared_sm90_checkpoint_weights(linear, loader) == 0
    assert loader.weights[0]["used"] is w
    monkeypatch.setenv("DSV41_SM90_DENSE_BACKEND", "humming")
    monkeypatch.delenv("DSV41_SM90_DENSE_BACKEND_MAP")
    assert select_sm90_dense_backend(256, 512) == "humming"
    assert linear._sm90_linear.backend == "deepgemm"


def test_noncontiguous_checkpoint_owner_releases_unused_source():
    w = torch.randn(512, 256, device="cuda").to(torch.float8_e4m3fn).T
    s = torch.ones(8, 16, device="cuda").to(torch.float8_e8m0fnu)
    source = weakref.ref(w)
    linear = V41MXFP8Linear(w, s)
    assert linear.weight.data_ptr() != w.data_ptr()
    assert not linear._sm90_linear.retains_checkpoint_weight
    loader = SimpleNamespace(weights=[{"used": w}], global_weights={"scale": s})
    del w
    assert release_prepared_sm90_checkpoint_weights(linear, loader) == 256 * 512
    gc.collect()
    assert source() is None


@pytest.mark.parametrize(
    "policy",
    ["[]", '{"x":"deepgemm"}', '{"0x32":"deepgemm"}', '{"256x512":"other"}', "bad"],
)
def test_invalid_override_rejected(monkeypatch, policy):
    monkeypatch.setenv("DSV41_SM90_DENSE_BACKEND_MAP", policy)
    with pytest.raises(ValueError):
        select_sm90_dense_backend(256, 512)


@pytest.mark.parametrize(
    "shape", ["merged_qkv", "cp8_q_b", "cp8_o_b", "shared_w13", "shared_w2"]
)
@pytest.mark.parametrize("rows", [1, 128, 512])
def test_actual_shapes_are_bitwise_equal_to_humming(shape, rows):
    _run_actual_shape(shape, rows, strict_reference=False)


@pytest.mark.parametrize(
    "shape", ["merged_qkv", "cp8_q_b", "cp8_o_b", "shared_w13", "shared_w2"]
)
@pytest.mark.parametrize("rows", [1, 128])
def test_actual_shapes_match_independent_reference(shape, rows):
    _run_actual_shape(shape, rows, strict_reference=True)


def _run_actual_shape(shape, rows, *, strict_reference):
    n, k = SHAPES[shape]
    linear, w, s = _linear(n, k)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    q, packed = linear._quantize_input(x)
    q_saved, packed_saved = q.view(torch.uint8).clone(), packed.clone()
    out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    assert linear.forward_quantized(q, packed, out=out) is out
    baseline = Sm90MXFP8Linear(w, s).forward_quantized(q, packed)
    torch.testing.assert_close(out, baseline, rtol=0, atol=0)
    print(
        "NATIVE_G32_BASELINE "
        + json.dumps(
            dict(
                shape=shape,
                rows=rows,
                bitwise_baseline=True,
                oracle=_metrics(out, _oracle(q, packed, w, s)),
            )
        ),
        flush=True,
    )
    if strict_reference:
        torch.testing.assert_close(out, _oracle(q, packed, w, s), rtol=0.015, atol=0.02)
    torch.testing.assert_close(linear(x), out, rtol=0, atol=0)
    torch.testing.assert_close(q.view(torch.uint8), q_saved, rtol=0, atol=0)
    torch.testing.assert_close(packed, packed_saved, rtol=0, atol=0)
    del linear, w, s, x, q, packed, out
    gc.collect()


@pytest.mark.parametrize(
    "rows,n,k,leading",
    [
        (0, 256, 128, (0,)),
        (1, 256, 128, (1,)),
        (7, 256, 512, (7,)),
        (15, 320, 640, (3, 5)),
        (129, 256, 512, (129,)),
    ],
)
def test_output_shape_empty_rows_and_graph_changes(rows, n, k, leading):
    _run_graph_case(rows, n, k, leading, strict_reference=rows <= 7)


def _run_graph_case(rows, n, k, leading, *, strict_reference):
    linear, w, s = _linear(n, k)
    x = torch.randn(*leading, k, device="cuda", dtype=torch.bfloat16)
    if rows:
        q, packed = linear._quantize_input(x)
    else:
        q = torch.empty(*leading, k, device="cuda", dtype=torch.float8_e4m3fn)
        packed = torch.empty(rows, (k // 32 + 3) // 4, device="cuda", dtype=torch.int32)
    out = torch.empty(*leading, n, device="cuda", dtype=torch.bfloat16)
    if rows == 0:
        with patch.object(linear._sm90_linear, "_forward") as launch:
            assert linear.forward_quantized(q, packed, out=out) is out
            launch.assert_not_called()
        return
    for _ in range(3):
        linear.forward_quantized(q, packed, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        linear.forward_quantized(q, packed, out=out)
    baseline = Sm90MXFP8Linear(w, s)
    observations = []
    for multiplier in (16.0, 0.03125):
        new_q, new_packed = linear._quantize_input(torch.randn_like(x) * multiplier)
        q.copy_(new_q)
        packed.copy_(new_packed)
        graph.replay()
        torch.testing.assert_close(
            out, baseline.forward_quantized(q, packed), rtol=0, atol=0
        )
        torch.testing.assert_close(
            out, linear.forward_quantized(q, packed), rtol=0, atol=0
        )
        observations.append(
            dict(
                multiplier=multiplier,
                baseline_bitwise=True,
                graph_eager_bitwise=True,
                **_metrics(out, _oracle(q, packed, w, s))
            )
        )
    print(
        "NATIVE_G32_GRAPH "
        + json.dumps(dict(rows=rows, n=n, k=k, observations=observations)),
        flush=True,
    )
    assert all(o["finite"] for o in observations)
    if strict_reference:
        assert all(not o["tolerance_violations"] for o in observations), json.dumps(
            observations
        )


def test_single_group_output_projection_uses_native_weight():
    from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_output_projection import (
        Sm90GroupedOutputProjection,
    )

    _, w, s = _linear(1024, 512)
    layer = Sm90GroupedOutputProjection(w.unsqueeze(0), s.unsqueeze(0))
    assert layer.backend == "deepgemm"
    assert layer.weight.data_ptr() == w.data_ptr()
    x = torch.randn(17, 512, device="cuda", dtype=torch.bfloat16)
    linear = V41MXFP8Linear(w, s)
    q, packed = linear._quantize_input(x)
    out = layer.forward_quantized(q.unsqueeze(1), packed.unsqueeze(1))
    torch.testing.assert_close(out, _oracle(q, packed, w, s), rtol=0.015, atol=0.02)
