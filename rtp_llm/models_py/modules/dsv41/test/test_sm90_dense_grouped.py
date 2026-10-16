"""Parent-locked SM90 WO_A grouped gates, with optional complete-chain timing.

No performance ranking after a failed numerical/graph/arithmetic gate. The
FP64 oracle independently decodes checkpoint and activation exponent bytes.
The measured chain starts at the original post-RoPE BF16 intermediate, and
ends at BF16 [1,8192]; RoPE, WO_B and model serving are NOT measured here.
"""

import hashlib
import json
import os
import statistics
from pathlib import Path

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.sm90_dense_grouped import Sm90GroupedWOA
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


@pytest.fixture(autouse=True)
def _locked_sm90():
    assert torch.cuda.is_available(), "Parent-locked SM90 GPU required"
    assert torch.cuda.get_device_capability() == (9, 0), "SM90 required"


def _hash(tensor):
    canonical = torch.empty(tensor.numel(), dtype=tensor.dtype, device=tensor.device)
    canonical.copy_(tensor.reshape(-1))
    return hashlib.sha256(canonical.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _decode_packed(packed):
    words = packed.cpu().to(torch.int64)
    shifts = torch.arange(4, dtype=torch.int64) * 8
    return ((words[..., None] >> shifts) & 255).reshape(words.shape[0], -1)


def _pow2(exponents):
    result = torch.ldexp(torch.ones_like(exponents, dtype=torch.float64), exponents.int() - 127)
    return torch.where(exponents == 255, float("nan"), result)


def _oracle(q, packed, weight, scales):
    # CPU FP64, one group at a time to bound memory. Never use production
    # unpack/layout helpers or Humming packed weights as the oracle.
    a = q.cpu().double().reshape(8, 128, 32) * _pow2(_decode_packed(packed))[..., None]
    b = weight.cpu()
    exponents = scales.contiguous().view(torch.uint8).cpu()
    result = []
    for group in range(8):
        bs = _pow2(exponents[group]).repeat_interleave(32, 0).repeat_interleave(32, 1)
        result.append((b[group].double() * bs) @ a[group].reshape(4096))
    return torch.stack(result).reshape(1, 8192)


def _metrics(actual, reference):
    actual = actual.detach().cpu().double()
    error = (actual - reference).abs()
    return dict(max_abs=float(error.max()), rms_abs=float(error.square().mean().sqrt()),
                violations=int((~torch.isfinite(actual) | ~torch.isfinite(reference)
                                | (error > .02 + .015 * reference.abs())).sum()))


def _old_chain(linears, x):
    return torch.stack([linear(x[:, g].contiguous()) for g, linear in enumerate(linears)], dim=1).flatten(1)


def _assert_quant_exact(adapter, old, x):
    q, packed = adapter.quantize(x)
    expected_q, expected_exponents = [], []
    for g, linear in enumerate(old):
        old_q, old_packed = linear._quantize_input(x[:, g].contiguous())
        expected_q.append(old_q.view(torch.uint8).cpu())
        expected_exponents.append(_decode_packed(old_packed))
    assert torch.equal(q.view(torch.uint8).cpu(), torch.cat(expected_q))
    assert torch.equal(_decode_packed(packed), torch.cat(expected_exponents))
    return q, packed


def _inputs():
    torch.manual_seed(1841)
    w = torch.randn(8, 1024, 4096, device="cuda").to(torch.float8_e4m3fn)
    s = torch.exp2(torch.randint(-3, 3, (8, 32, 128), device="cuda").float()).to(torch.float8_e8m0fnu)
    x = torch.randn(1, 8, 4096, device="cuda", dtype=torch.bfloat16)
    # Distinct groups make group-index errors visible.
    x.mul_(torch.arange(1, 9, device="cuda", dtype=torch.bfloat16).view(1, 8, 1) / 4)
    return w, s, x


def _run(weight, scales, x, directory, *, benchmark):
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(decision="NOT_RUN", scope="M1 G8 WO_A post-RoPE BF16 to BF16; not serving TPOT",
                  tolerance=dict(rtol=.015, atol=.02), oracle="Independent CPU FP64 decoded g32",
                  performance="NOT_RUN", numerical=[], quant_payload_scale_exact="NOT_RUN")
    originals = [(tensor, _hash(tensor)) for tensor in (weight, scales, x)]
    prepared = []
    adapter = None
    try:
        adapter = Sm90GroupedWOA(weight, scales)
        old = [V41MXFP8Linear(weight[g], scales[g]) for g in range(8)]
        # Explicitly prove this is the same legacy native-A8 Dense arm.
        from rtp_llm.models_py.modules.dsv41.sm90_dense import Sm90MXFP8Linear
        assert all(type(linear._sm90_linear) is Sm90MXFP8Linear for linear in old)
        q, packed = _assert_quant_exact(adapter, old, x)
        prepared = [(tensor, _hash(tensor)) for tensor in
                    (q, packed, adapter.weight, adapter.weight_scale, adapter.expert_offsets)]
        record["quant_payload_scale_exact"] = "PASS"
        reference = _oracle(q, packed, weight, scales)
        out = torch.full((1, 8192), float("nan"), dtype=torch.bfloat16, device=x.device)
        result = adapter.forward_quantized(q, packed, out=out)
        assert result is out
        baseline = _old_chain(old, x)
        for name, value in (("candidate_fp64", result), ("old_fp64", baseline)):
            metric = _metrics(value, reference)
            record["numerical"].append(dict(arm=name, **metric))
        assert all(metric["violations"] == 0 for metric in record["numerical"]), record["numerical"]
        torch.testing.assert_close(result, baseline, rtol=.015, atol=.02)
        assert result.dtype == torch.bfloat16 and result.shape == (1, 8192)
        for _ in range(3):
            torch.testing.assert_close(adapter(x), result, rtol=0, atol=0)

        # Freeze weights/metadata; graph must quantize the new BF16 inputs.
        static = x.clone()
        for _ in range(3):
            adapter(static)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = adapter(static)
        initial_q, initial_packed = adapter.quantize(static)
        for factor in (-.125, 8.):
            static.copy_(torch.randn_like(static) * factor)
            changed_q, changed_packed = _assert_quant_exact(adapter, old, static)
            assert not torch.equal(initial_q.view(torch.uint8), changed_q.view(torch.uint8))
            assert not torch.equal(_decode_packed(initial_packed), _decode_packed(changed_packed))
            graph.replay()
            ref = _oracle(changed_q, changed_packed, weight, scales)
            metric = _metrics(graph_output, ref)
            record["numerical"].append(dict(arm=f"graph_{factor}", **metric))
            assert metric["violations"] == 0, metric
            torch.testing.assert_close(graph_output, adapter(static), rtol=0, atol=0)
            torch.testing.assert_close(graph_output, _old_chain(old, static), rtol=.015, atol=.02)
        record["changed_input_graph"] = "PASS"

        # Export the actual registered kernel/cubin selected by the production
        # launcher. This helper inspects SASS, not merely config/operand dtype.
        from rtp_llm.models_py.modules.dsv41.test.test_sm90_dense_native_a8 import _actual_selected_binary
        arithmetic = _actual_selected_binary(adapter, 8, directory)
        record["selected_binary"] = arithmetic
        assert arithmetic["qgmma_instructions"], arithmetic
        assert all(".F32.E4M3.E4M3 " in instruction for instruction in arithmetic["qgmma_instructions"]), arithmetic
        assert arithmetic["sass_counts"]["HMMA"] == arithmetic["sass_counts"]["HGMMA"] == 0, arithmetic
        assert torch.equal(adapter.expert_offsets.cpu(), torch.arange(9, dtype=torch.int32))
        for tensor, digest in originals + prepared:
            assert _hash(tensor) == digest, "Input/weight mutated"
        record["decision"] = "PASS"
        if benchmark:
            from rtp_llm.models_py.modules.dsv41.test.test_sm90_dense_deepgemm import _measure
            flush = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
            # Time both eager orchestration and the production-style captured
            # complete chain. Neither arm excludes quantization/scale/copies.
            old_graph, grouped_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(old_graph):
                old_graph_output = _old_chain(old, x)
            with torch.cuda.graph(grouped_graph):
                grouped_graph_output = adapter(x)
            old_graph.replay()
            grouped_graph.replay()
            torch.testing.assert_close(old_graph_output, baseline, rtol=0, atol=0)
            torch.testing.assert_close(grouped_graph_output, result, rtol=0, atol=0)
            arms = [("eight_dense_eager", lambda: _old_chain(old, x)),
                    ("grouped_eager", lambda: adapter(x)),
                    ("eight_dense_graph", old_graph.replay),
                    ("grouped_graph", grouped_graph.replay)]
            results = {}
            for mode in ("hot", "cold"):
                samples = {name: [] for name, _ in arms}
                for repeat in range(3):
                    for name, call in (arms if repeat % 2 == 0 else arms[::-1]):
                        measured = _measure(call, flush if mode == "cold" else None,
                                            directory / f"{name}_{mode}_{repeat}.json", prepared=False)
                        count = sum(k["count"] for k in measured["profile"]["kernels"] if "void humming<" in k["name"])
                        assert count == (8 if name.startswith("eight_dense") else 1), measured
                        samples[name].append(measured)
                results[mode] = dict(samples=samples, kernel_sum_us={
                    name: statistics.median(s["profile"]["kernel_sum_us_per_iter"] for s in values)
                    for name, values in samples.items()})
            record["performance"] = results
    except BaseException as error:
        record["decision"] = "REJECT"
        record["error"] = repr(error)
        raise
    finally:
        record["immutable_inputs"] = all(_hash(t) == digest for t, digest in originals + prepared)
        if not record["immutable_inputs"]:
            record["decision"] = "REJECT"
            record["performance"] = "INVALIDATED_BY_INPUT_MUTATION"
        (directory / "result.json").write_text(json.dumps(record, indent=2))
        assert record["immutable_inputs"], "Input mutation invalidates candidate"


@pytest.mark.parametrize("quant_backend", ["auto", "legacy", "v2"])
@torch.inference_mode()
def test_grouped_wo_a(quant_backend, monkeypatch):
    monkeypatch.setenv("DSV4_FP8_QUANT_KERNEL", quant_backend)
    root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    assert root, "Persist acceptance and failed records in Bazel outputs"
    weight, scales, x = _inputs()
    _run(weight, scales, x, Path(root) / f"grouped_{quant_backend}",
         benchmark=quant_backend == "auto" and os.environ.get("PERF_GROUPED_WOA") == "1")


@torch.inference_mode()
def test_grouped_wo_a_checkpoint(monkeypatch):
    checkpoint = os.environ.get("DSV41_GROUPED_WOA_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set DSV41_GROUPED_WOA_CHECKPOINT for real-weight acceptance")
    from safetensors import safe_open

    monkeypatch.setenv("DSV4_FP8_QUANT_KERNEL", "auto")
    root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    assert root, "Persist acceptance and failed records in Bazel outputs"
    checkpoint = Path(checkpoint)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    layer = int(os.environ.get("DSV41_GROUPED_WOA_LAYER", "0"))
    tensors = []
    for suffix, shape, dtype in (("weight", (8, 1024, 4096), torch.float8_e4m3fn),
                                 ("scale", (8, 32, 128), torch.float8_e8m0fnu)):
        key = f"layers.{layer}.attn.wo_a.{suffix}"
        with safe_open(str(checkpoint / index[key]), framework="pt", device="cpu") as source:
            tensor = source.get_tensor(key)
        assert tensor.dtype == dtype, (key, tensor.dtype)
        tensors.append(tensor.reshape(shape).cuda())
    torch.manual_seed(1841)
    x = torch.randn(1, 8, 4096, device="cuda", dtype=torch.bfloat16)
    _run(*tensors, x, Path(root) / f"grouped_checkpoint_layer{layer}",
         benchmark=os.environ.get("PERF_GROUPED_WOA") == "1")


@torch.inference_mode()
def test_bounded_api_rejects_other_rows_and_aliases():
    weight, scales, x = _inputs()
    adapter = Sm90GroupedWOA(weight, scales)
    assert adapter.supports(x)
    for bad in (x.expand(2, -1, -1), x[:, :4], x.float(), x.cpu(), x.transpose(1, 2)):
        assert not adapter.supports(bad)
        with pytest.raises(ValueError):
            adapter(bad)
    q, packed = adapter.quantize(x)
    with pytest.raises(ValueError):
        adapter.forward_quantized(q[:, :2048], packed)
    with pytest.raises(ValueError):
        adapter.forward_quantized(q, packed, out=torch.empty(1, 8192, device=x.device, dtype=torch.float32))
    # Correctly shaped BF16 view sharing q storage must also be rejected.
    alias = q.view(torch.bfloat16).view(1, -1)[:, :8192]
    with pytest.raises(ValueError, match="alias"):
        adapter.forward_quantized(q, packed, out=alias)
