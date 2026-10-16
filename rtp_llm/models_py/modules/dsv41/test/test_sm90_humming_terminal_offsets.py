"""Isolated wheel-0.1.16 vs terminal-offset backport correctness gates.

Each Humming arm runs in a fresh interpreter and separate JIT cache; the parent
compares CPU BF16 bits. No installed package or production selector is edited.
RTP G128 activation quantization and native W4A8/K32 promotion stay unchanged.
This tests valid rows and skipped stores, not a new full-FP32 accumulation claim.
Parent owns Bazel/GPU locking; subprocesses inherit its CUDA_VISIBLE_DEVICES.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import pytest


def _activate_arm():
    helper = Path(os.environ["DSV41_HUMMING_TERMINAL_HELPER"]).resolve(strict=True)
    if any(name == "humming" or name.startswith("humming.") for name in sys.modules):
        raise AssertionError("Worker must select its source before importing Humming")
    spec = importlib.util.spec_from_file_location("terminal_source_selection", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provenance = module.activate_before_import()
    return helper, provenance


def _hash_tensor(tensor):
    # CPU serialization is outside both capture and benchmark timing.
    return hashlib.sha256(
        tensor.detach()
        .contiguous()
        .view(-1)
        .view(__import__("torch").uint8)
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()


def _offsets(experts, valid, layout):
    counts = [0] * experts
    if layout == "first":
        counts[0] = valid
    elif layout == "last":
        counts[-1] = valid
    elif layout == "ends":
        counts[0], counts[-1] = valid // 2, valid - valid // 2
    elif layout == "striped":
        for row in range(valid):
            counts[(row * 7) % experts] += 1
    else:
        assert valid == 0 and layout == "empty"
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return offsets


def _worker(args):
    helper, activation = _activate_arm()
    import torch
    import humming
    from humming.config import MmaType
    from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_a8 import (
        SM90HummingGroupedA8Linear,
    )

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    assert str(humming.__version__) == "0.1.16"
    assert (
        Path(humming.__file__).resolve().parent
        == Path(activation["selected_site"]) / "humming"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    arm = os.environ["DSV41_HUMMING_TERMINAL_ARM"]
    assert arm in ("baseline", "candidate")
    experts, n, k = args.experts, args.n, args.k
    torch.manual_seed(252516)
    weight = torch.randint(
        -128, 128, (experts, n, k // 2), device="cuda", dtype=torch.int8
    )
    raw_scales = torch.randint(
        117, 128, (experts, n, k // 32), device="cuda", dtype=torch.uint8
    )
    scales = raw_scales.view(torch.float8_e8m0fnu)
    source_hashes = _hash_tensor(weight), _hash_tensor(raw_scales)
    layer = SM90HummingGroupedA8Linear(weight, scales, promotion_group_size=32)
    assert layer.config.mma_type == MmaType.WGMMA
    assert str(layer.config.a_dtype) == "float8e4m3"
    assert layer.config.input_scale_group_size == 32
    assert layer.compute_config["use_batch_invariant"]
    assert not layer.compute_config["use_f16_accum"]
    records, timing_records = {}, []

    def exact(a, b):
        assert a.dtype == b.dtype and a.shape == b.shape
        torch.testing.assert_close(
            a.contiguous().view(torch.uint8),
            b.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )

    def quant_reference(x):
        blocks = x.float().reshape(x.shape[0], -1, 128)
        scale = torch.pow(
            2.0, torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
        )
        q = (blocks / scale[..., None]).to(torch.float8_e4m3fn).reshape_as(x)
        return q, scale.repeat_interleave(4, dim=-1).contiguous()

    def capture(fn):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        return graph

    for capacity in args.capacity:
        x = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16)
        offsets = torch.tensor(
            _offsets(experts, 6, "last"), device="cuda", dtype=torch.int32
        )
        # A contiguous subview gives real external guard rows without changing
        # the supported output contract or aliasing any GEMM operand.
        sentinel = 13.25
        backing = torch.full(
            (capacity + 2, n), sentinel, device="cuda", dtype=torch.bfloat16
        )
        output = backing[1:-1]

        def forward():
            return layer.prepared(*layer.quantize_input(x), offsets, outputs=output)

        graph = capture(forward)
        plans = [
            (0, "empty"),
            (1, "first"),
            (6, "last"),
            (8, "ends"),
            (33, "striped"),
            (capacity - 1, "last"),
            (capacity, "ends"),
        ]
        if capacity > 64:
            plans += [(63, "striped"), (64, "last"), (65, "ends")]
        for phase, (valid, layout) in enumerate(plans):
            torch.manual_seed(25100 + phase + capacity)
            x.copy_(torch.randn_like(x) * (0.03125 if phase % 2 else 16.0))
            offsets.copy_(
                torch.tensor(
                    _offsets(experts, valid, layout), device="cuda", dtype=torch.int32
                )
            )
            q, sf = layer.quantize_input(x)
            q_ref, sf_ref = quant_reference(x)
            exact(q, q_ref)
            exact(sf, sf_ref)
            before = _hash_tensor(x), _hash_tensor(offsets)
            backing.fill_(sentinel)
            eager = forward().clone()
            if arm == "candidate":
                exact(eager[valid:], torch.full_like(eager[valid:], sentinel))
            exact(backing[:1], torch.full_like(backing[:1], sentinel))
            exact(backing[-1:], torch.full_like(backing[-1:], sentinel))
            backing.fill_(sentinel)
            graph.replay()
            torch.cuda.synchronize()
            exact(output[:valid], eager[:valid])
            if arm == "candidate":
                exact(output[valid:], torch.full_like(output[valid:], sentinel))
            exact(backing[:1], torch.full_like(backing[:1], sentinel))
            exact(backing[-1:], torch.full_like(backing[-1:], sentinel))
            assert before == (_hash_tensor(x), _hash_tensor(offsets))
            assert bool(torch.isfinite(output[:valid]).all())
            key = f"capacity{capacity}_valid{valid}_{layout}"
            records[key] = {
                "valid_output": output[:valid].cpu(),
                "offsets": offsets.cpu(),
                "q_sha256": _hash_tensor(q),
                "scale_sha256": _hash_tensor(sf),
                "input_sha256": before[0],
                "valid": valid,
                "capacity": capacity,
            }
            if valid == 0 and arm == "candidate":
                output.zero_()
                graph.replay()
                exact(output, torch.zeros_like(output))
        if args.small:
            # Independent exact one-hot oracle: each output has one product,
            # so no native reduced-accumulation tolerance exception is needed.
            valid = min(6, capacity)
            x.zero_()
            x[:valid, 0] = 1
            offsets.copy_(
                torch.tensor(
                    _offsets(experts, valid, "ends"), device="cuda", dtype=torch.int32
                )
            )
            backing.fill_(sentinel)
            graph.replay()
            lut = torch.tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                device="cuda",
            )
            offset_list = offsets.cpu().tolist()
            for expert in (0, experts - 1):
                begin, end = offset_list[expert : expert + 2]
                code = weight[expert, :, 0].to(torch.int16) & 15
                expected = (
                    lut[code.long()]
                    * torch.exp2(raw_scales[expert, :, 0].float() - 127)
                ).bfloat16()
                # Zero products may canonicalize signed zero in accumulation;
                # exact numerical equality is appropriate for this math oracle.
                torch.testing.assert_close(
                    output[begin:end], expected.expand(end - begin, -1), rtol=0, atol=0
                )
            if arm == "candidate":
                exact(output[valid:], torch.full_like(output[valid:], sentinel))
        if args.benchmark:
            for valid in (8, min(48, capacity), capacity):
                offsets.copy_(
                    torch.tensor(
                        _offsets(experts, valid, "ends"),
                        device="cuda",
                        dtype=torch.int32,
                    )
                )
                # Include the same zero initialization on both arms. This is
                # quant+one grouped GEMM, not a complete MoE/DeepEP benchmark.
                timed = capture(lambda: (output.zero_(), forward()))
                samples = []
                for _ in range(5):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    for _ in range(100):
                        timed.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000 / 100)
                timing_records.append(
                    dict(capacity=capacity, valid=valid, us=statistics.median(samples))
                )
    assert source_hashes == (_hash_tensor(weight), _hash_tensor(raw_scales))
    package = Path(humming.__file__).resolve().parent
    provenance = dict(
        arm=arm,
        package=str(package),
        version=humming.__version__,
        helper=str(helper),
        activation=activation,
        cache=os.environ.get("HUMMING_CACHE_DIR"),
        scheduler_sha256=hashlib.sha256(
            (package / "include/humming/scheduler.cuh").read_bytes()
        ).hexdigest(),
        config=json.loads(layer.config.to_str()),
        compute_config=layer.compute_config,
    )
    torch.save(
        dict(
            records=records,
            source_hashes=source_hashes,
            timings=timing_records,
            provenance=provenance,
        ),
        args.output,
    )
    print("TERMINAL_OFFSETS_WORKER " + json.dumps(provenance), flush=True)


def _compare_shape(tmp_path, name, experts, n, k, *, small=False, prefill=False):
    assert os.environ.get(
        "DSV41_HUMMING_TERMINAL_HELPER"
    ), "Set source-isolation helper; missing candidate is not PASS"
    import torch

    if os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        tmp_path = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / f"terminal_{name}"
        tmp_path.mkdir(parents=True, exist_ok=False)
    results = {}
    for arm in ("baseline", "candidate"):
        output = tmp_path / f"{name}_{arm}.pt"
        environment = os.environ.copy()
        environment["DSV41_HUMMING_TERMINAL_ARM"] = arm
        # Preserve Bazel's import roots for the child and any launcher-builder
        # grandchildren; the source helper prepends the selected Humming arm.
        # Bazel adds overlapping roots to sys.path and PYTHONPATH. Concatenating
        # both verbatim can exceed Linux's per-environment-string exec limit.
        roots = sys.path + environment.get("PYTHONPATH", "").split(os.pathsep)
        roots = dict.fromkeys(
            str(Path(root).resolve()) for root in roots if root and Path(root).is_dir()
        )
        environment["PYTHONPATH"] = os.pathsep.join(roots)
        assert len(environment["PYTHONPATH"].encode()) < 120_000
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--terminal-offset-worker",
            "--experts",
            str(experts),
            "--n",
            str(n),
            "--k",
            str(k),
            "--capacity",
            *(["4096"] if prefill else ["64", "256"]),
            "--output",
            str(output),
        ]
        if small:
            command.append("--small")
        if prefill or os.getenv("DSV41_SM90_TERMINAL_BENCH") == "1":
            command.append("--benchmark")
        completed = subprocess.run(
            command, env=environment, capture_output=True, text=True, timeout=3600
        )
        log = tmp_path / f"{name}_{arm}.log"
        log.write_text(completed.stdout + "\n" + completed.stderr)
        print(completed.stdout, flush=True)
        assert (
            completed.returncode == 0
        ), f"{arm} failed; log={log}\n{completed.stderr[-12000:]}"
        results[arm] = torch.load(output, map_location="cpu", weights_only=False)
    baseline, candidate = results["baseline"], results["candidate"]
    assert baseline["source_hashes"] == candidate["source_hashes"]
    assert baseline["provenance"]["package"] != candidate["provenance"]["package"]
    assert baseline["provenance"]["cache"] != candidate["provenance"]["cache"]
    assert (
        baseline["provenance"]["scheduler_sha256"]
        != candidate["provenance"]["scheduler_sha256"]
    )
    assert baseline["records"].keys() == candidate["records"].keys()
    for key in baseline["records"]:
        a, b = baseline["records"][key], candidate["records"][key]
        for field in ("q_sha256", "scale_sha256", "input_sha256", "valid", "capacity"):
            assert a[field] == b[field], (key, field)
        torch.testing.assert_close(a["offsets"], b["offsets"], rtol=0, atol=0)
        torch.testing.assert_close(
            a["valid_output"].view(torch.int16),
            b["valid_output"].view(torch.int16),
            rtol=0,
            atol=0,
            msg=lambda message: f"{key}: {message}",
        )
    for old, new in zip(baseline["timings"], candidate["timings"]):
        assert (old["capacity"], old["valid"]) == (new["capacity"], new["valid"])
        print(
            "TERMINAL_OFFSETS_BENCH "
            + json.dumps(
                dict(
                    shape=name,
                    **old,
                    candidate_us=new["us"],
                    speedup=old["us"] / new["us"],
                    scope="zero-init+G128 quant+one grouped W4A8 GEMM; not MoE or TPOT",
                )
            ),
            flush=True,
        )


def test_small_terminal_offsets_bitexact_graph_and_onehot(tmp_path):
    _compare_shape(tmp_path, "small", 48, 128, 512, small=True)


@pytest.mark.parametrize("name,n,k", [("w13", 4608, 5120), ("w2", 5120, 2304)])
def test_real_decode_terminal_offsets_bitexact_and_graph(tmp_path, name, n, k):
    _compare_shape(tmp_path, name, 48, n, k)


@pytest.mark.skipif(
    os.getenv("DSV41_SM90_TERMINAL_PREFILL_BENCH") != "1",
    reason="optional large-M experiment, not decode acceptance",
)
def test_optional_prefill_large_m_terminal_offsets(tmp_path):
    _compare_shape(tmp_path, "prefill_w13", 48, 4608, 5120, prefill=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal-offset-worker", action="store_true", required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--capacity", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    _worker(parser.parse_args())
