"""Native-A8 arithmetic-contract checks, NOT FP32-numerical acceptance.

Real V41 factory, actual selected Humming cubin/SASS, immutable operands,
repeatability and changed-input CUDA graph replay are asserted. Independent
FP32-oracle errors retain the original .015/.02 thresholds in artifacts and
are reported separately: arithmetic PASS never converts numerical REJECT to
PASS. No timing/ranking is performed by this diagnostic target.
"""

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.sm90_dense import Sm90MXFP8Linear
from rtp_llm.models_py.modules.dsv41.test.test_sm90_dense_tuning import (
    SHAPES,
    _hash,
    _metrics,
    _oracle,
    _selected_default,
)
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


@pytest.fixture(autouse=True)
def _require_locked_sm90():
    assert torch.cuda.is_available(), "Requires parent-locked CUDA GPU"
    assert torch.cuda.get_device_capability() == (9, 0), "Requires SM90"
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def _same_bits(actual, expected):
    assert torch.equal(actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16))


def _actual_selected_binary(adapter, rows, directory):
    from humming.kernel.humming import HummingKernel

    # Same registered interval table consumed by production launch_kernel;
    # do not create a parallel hand-selected proxy kernel for this audit.
    table = HummingKernel.prepare_kernels(
        adapter.config.to_str(), adapter.compute_config, None, adapter.weight.device
    ).reshape(-1, 4).tolist()
    selected = [entry for entry in table if entry[0] < rows <= entry[1]]
    assert len(selected) == 1
    low, high, kernel_id, num_sms = selected[0]
    kernel = HummingKernel._id2kernel[kernel_id]
    source = Path(kernel.kernel_filename)
    exported = directory / "actual_selected_humming.cubin"
    shutil.copy2(source, exported)
    generated = source.with_name("kernel.cu")
    if generated.is_file():
        shutil.copy2(generated, directory / "actual_selected_humming.cu")
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    result = subprocess.run(
        [cuobjdump, "--dump-sass", str(exported)], capture_output=True, text=True,
        check=True, timeout=120,
    )
    (directory / "actual_selected_humming.sass").write_text(result.stdout)
    sass = result.stdout
    qgmma = re.findall(r"\bQGMMA\.[^;\n]+", sass)
    counts = {name: len(re.findall(r"\b" + name + r"\.", sass))
              for name in ("QGMMA", "HGMMA", "HMMA")}
    return dict(
        selected_interval=[low, high], kernel_id=kernel_id, num_sms=num_sms,
        registered_kernel_name=kernel.kernel_name, original_cubin=str(source),
        exported_cubin=str(exported), cubin_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),
        sass_counts=counts, qgmma_instructions=list(dict.fromkeys(qgmma)),
        provenance="Actual V41 factory adapter's production registered interval table",
    )


def _run_arithmetic_contract(rows, shape_name, n, k, leading=None):
    assert importlib.metadata.version("humming-kernels") == "0.1.16"
    output_root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    assert output_root, "Bazel undeclared outputs are mandatory"
    directory = Path(output_root) / f"native_a8_contract_{shape_name}_m{rows}_n{n}_k{k}"
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(
        M=rows, N=n, K=k, shape_name=shape_name, seed=5141,
        scope="Dense native-A8 arithmetic contract, not FP32 numerical acceptance or performance",
        arithmetic_contract="NOT_RUN", numerical_gate="NOT_RUN",
        tolerance=dict(rtol=.015, atol=.02), numerical_observations=[],
        performance="NOT_RUN", model_correctness="NOT_RUN",
    )
    immutable = []
    try:
        torch.manual_seed(5141)
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.pow(
            2.0, torch.randint(-3, 3, (n // 32, k // 32), device="cuda").float()
        ).to(torch.float8_e8m0fnu)
        reference_weight, reference_scales = weight.clone(), scales.clone()
        immutable.extend((value, _hash(value)) for value in (weight, scales))
        with pytest.MonkeyPatch.context() as patch:
            # Deliberately conflict with the former selected-HMMA option to
            # prove native-A8 precedence through the real factory.
            patch.setenv("DSV41_SM90_NATIVE_A8", "1")
            patch.setenv("DSV41_SM90_DENSE_G32_SELECTED", "1")
            linear = V41MXFP8Linear(weight, scales)
        adapter = linear._sm90_linear
        assert type(adapter) is Sm90MXFP8Linear
        config = json.loads(adapter.config.to_str())
        assert config["mma_type"] == "wgmma"
        assert config["a_dtype"] == config["b_dtype"] == "float8e4m3"
        assert config["input_scale_group_size"] == config["weight_scale_group_size"] == 32
        assert not config["use_fused_e8m0_scale"] and not config["use_int_weight_scale"]
        compute = json.loads(adapter.compute_config)
        assert compute["use_batch_invariant"] and not compute["use_f16_accum"]
        interval, tuning = _selected_default(adapter, rows)
        assert tuning.get("mma_type", config["mma_type"]) == "wgmma"
        assert not tuning.get("use_stream_k", False)
        record.update(factory=type(linear).__name__, adapter=type(adapter).__name__,
                      layer_config=config, compute_config=compute,
                      actual_selected_interval=interval, actual_selected_tuning=tuning)
        x = torch.randn(*(leading or (rows,)), k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty((*x.shape[:-1], n), device="cuda", dtype=torch.bfloat16)
        q, packed = linear._quantize_input(x)
        immutable.extend((value, _hash(value)) for value in (q, packed))
        expected = _oracle(q, packed, reference_weight, reference_scales)
        linear.forward_quantized(q, packed, out=out)
        saved = out.clone()
        record["numerical_observations"].append(dict(phase="original", **_metrics(saved, expected)))
        for _ in range(3):
            linear.forward_quantized(q, packed, out=out)
            _same_bits(out, saved)
        linear(x, out=out)
        _same_bits(out, saved)
        torch.cuda.synchronize()
        binary = _actual_selected_binary(adapter, rows, directory)
        record["binary_evidence"] = binary
        assert binary["sass_counts"]["QGMMA"] > 0
        assert binary["sass_counts"]["HGMMA"] == binary["sass_counts"]["HMMA"] == 0
        assert all(".F32.E4M3.E4M3 " in instruction for instruction in binary["qgmma_instructions"])

        # Actual forward trace corroborates that this test launched Humming;
        # instruction dtype is established by the selected cubin, not its name.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            linear(x, out=out)
        timeline = directory / "actual_factory_forward.json"
        profiler.export_chrome_trace(str(timeline))
        events = json.loads(timeline.read_text())["traceEvents"]
        names = [event.get("name", "") for event in events if event.get("cat") == "kernel"]
        record["actual_forward_kernel_names"] = names
        assert sum("humming<" in name for name in names) == 1

        for _ in range(3):
            linear(x, out=out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            linear(x, out=out)
        record["changed_input_graph"] = []
        torch.manual_seed(514100 + rows)
        for multiplier in (16.0, 0.03125):
            x.copy_(torch.randn_like(x) * multiplier)
            changed_q, changed_packed = linear._quantize_input(x)
            payload_changes = int((changed_q.view(torch.uint8) != q.view(torch.uint8)).sum())
            scale_changes = int((changed_packed != packed).sum())
            assert payload_changes and scale_changes
            graph.replay()
            graph_result = out.clone()
            reference = _oracle(changed_q, changed_packed, reference_weight, reference_scales)
            record["numerical_observations"].append(
                dict(phase=f"changed_input_{multiplier}", **_metrics(graph_result, reference))
            )
            linear(x, out=out)
            _same_bits(out, graph_result)
            for _ in range(3):
                graph.replay()
                _same_bits(out, graph_result)
            record["changed_input_graph"].append(dict(
                multiplier=multiplier, payload_bytes_changed=payload_changes,
                scale_words_changed=scale_changes, graph_vs_eager_bitwise=True,
                repeated_graph_bitwise=True,
            ))
        record["numerical_gate"] = (
            "REJECT" if any(item["tolerance_violations"] for item in record["numerical_observations"])
            else "PASS_BOUNDED_ORIGINAL_TOLERANCE"
        )
        assert all(item["finite"] for item in record["numerical_observations"])
        assert all(_hash(value) == digest for value, digest in immutable)
        record["immutable_operands"] = "PASS"
        record["arithmetic_contract"] = "PASS_NATIVE_E4M3_QGMMA"
        print(f"NATIVE_A8_ARITHMETIC_CONTRACT {shape_name} M{rows}: "
              f"{record['arithmetic_contract']}; FP32_NUMERICAL_GATE={record['numerical_gate']}", flush=True)
    except Exception as exc:
        record["arithmetic_contract"] = "FAIL"
        record["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if immutable:
            record["immutable_hashes_unchanged"] = [_hash(value) == digest for value, digest in immutable]
        (directory / "arithmetic_contract.json").write_text(json.dumps(record, indent=2))


@pytest.mark.parametrize("rows,n,k,leading", [(7, 256, 512, None), (15, 320, 640, (3, 5))])
def test_native_a8_arithmetic_contract_fixture(rows, n, k, leading):
    _run_arithmetic_contract(rows, "fixture", n, k, leading)


@pytest.mark.parametrize("rows", [1, 16, 512])
@pytest.mark.parametrize("shape_name", os.environ.get(
    "DSV41_NATIVE_A8_DENSE_SHAPES", "shared_w13,shared_w2"
).split(","))
def test_native_a8_arithmetic_contract_actual(rows, shape_name):
    n, k = SHAPES[shape_name]
    _run_arithmetic_contract(rows, shape_name, n, k)
