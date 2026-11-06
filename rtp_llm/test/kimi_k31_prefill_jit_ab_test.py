"""Cold-request and warm-path A/B for the K3.1 Prefill JIT fix.

Run as a Bazel test: one GPU lock covers fresh A/B/B/A subprocesses. The
baseline archive is exported from the fix's parent, without editing kernels.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

BASELINE = "ddc6b5c5e7313bdbae82080cc024e07392a1388d"
CANDIDATE = "8e60c1f300e7eda7a5c478406b385ddd342191fc"
ROWS = (3596, 58, 924, 34, 1252, 2118, 1030, 176, 5836, 8192)
MODULES = {
    "slots": "rtp_llm.models_py.kernels.cuda.dsa_kv_chunk",
    "query": "rtp_llm.models_py.kernels.cuda.dsa_buffer_ops",
    "norm": "rtp_llm.models_py.modules.kimi_k3.dsa_jit_warmup",
}
HASHES = {
    "A": (
        "61f7059da2bb92ef24a02e540eeb53c1e55766625dd88400f1b4983a35b3a8e2",
        "fc66f1f673e0918adeab7eff28f11951ffd5023f33a5c7b9098af1c5eefde767",
        "6c927a093de2b7f09db7944454fed6b25295d40a393bea6ed615a6fb8c137496",
    ),
    "B": (
        "b671a231fb662ef207ec9e135b5f86b004b1110fc157455ed5c2c1ee8d76a5e5",
        "d6d67e1dd126b895e6800faf7494ba98c12bcb89a7ebbf8f53b210baca1e8851",
        "9befdf89430174516ef449fa5c2369efe59c775374812121dd5885c5125c948f",
    ),
}


def load_sources(variant, root):
    loaded, identities = {}, {}
    archive = Path(__file__).with_name("kimi_k31_prefill_jit_ab_baseline.tar")
    with tarfile.open(archive) as baseline:
        for (key, name), expected in zip(MODULES.items(), HASHES[variant]):
            if variant == "A":
                relative = name.replace(".", "/") + ".py"
                source = baseline.extractfile(relative).read()
                path = root / (key + ".py")
                path.write_bytes(source)
                spec = importlib.util.spec_from_file_location("baseline_" + key, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(name)
                source = Path(module.__file__).read_bytes()
            digest = hashlib.sha256(source).hexdigest()
            if digest != expected:
                raise AssertionError(f"{variant} {name}: unexpected source {digest}")
            loaded[key] = module
            identities[name] = digest
    return loaded, identities


def telemetry(uuid):
    command = [
        "nvidia-smi",
        "-i",
        uuid,
        "--query-gpu=uuid,name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    return subprocess.check_output(command, text=True).strip()


def worker(variant, destination):
    import torch
    import triton

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    if props.major != 10 or torch.cuda.device_count() != 1:
        raise RuntimeError("The A/B requires exactly one SM10 GPU under gpu_lock")
    uuid = str(props.uuid)
    if not uuid.startswith(("GPU-", "MIG-")):
        uuid = "GPU-" + uuid
    root = Path(destination).parent
    modules, identities = load_sources(variant, root)
    from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
        grouped_q_kv_rmsnorm,
    )

    def cold(function):
        torch.cuda.synchronize()
        misses = []
        with triton.knobs.compilation.scope():

            def observe(*, cache_hit, src, **_):
                if not cache_hit:
                    misses.append(src.fn.__name__)

            triton.knobs.compilation.listener = observe
            started = time.perf_counter_ns()
            output = function()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - started) / 1e6
        return output, {"wall_ms": elapsed, "jit_misses": misses}

    def steady(function):
        for _ in range(5):
            output = function()
        torch.cuda.synchronize()
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(30)
        ]
        for start, end in events:
            start.record()
            output = function()
            end.record()
        torch.cuda.synchronize()
        eager = [start.elapsed_time(end) * 1000 for start, end in events]
        # Graph batching amortizes Python launch gaps, separately from eager.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(32):
                output = function()
        graph.replay()
        torch.cuda.synchronize()
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize()
        replay = [start.elapsed_time(end) * 1000 / 32 for start, end in events]
        return {
            "eager_cuda_event_us": eager,
            "graph_cuda_event_us_per_call": replay,
            "eager_p50_us": statistics.median(eager),
            "graph_p50_us": statistics.median(replay),
        }

    attention = SimpleNamespace(
        _qkv_norm=grouped_q_kv_rmsnorm,
        q_rank=1536,
        kv_rank=512,
        suffix_dim=64,
        heads=96,
        v_dim=128,
        input=SimpleNamespace(
            weight=torch.empty((14464, 7168), device="meta", dtype=torch.float8_e4m3fn)
        ),
        q_norm=SimpleNamespace(
            weight=torch.ones(1536, device="cuda", dtype=torch.bfloat16),
            variance_epsilon=1e-6,
        ),
        kv_norm=SimpleNamespace(
            weight=torch.ones(512, device="cuda", dtype=torch.bfloat16)
        ),
    )
    weight = torch.arange(96 * 128 * 512, device="cuda").remainder_(4)
    weight = weight.reshape(96, 128, 512).to(torch.bfloat16).div_(8)
    cases = {}
    for rows in ROWS:
        source = torch.ones((rows, 14464), device="cuda", dtype=torch.bfloat16)
        q, kv, suffix = source[:, :2112].split((1536, 512, 64), dim=-1)
        query = torch.full((rows, 96, 192), 0.5, device="cuda", dtype=torch.bfloat16)
        query[..., 128:] = torch.arange(64, device="cuda", dtype=query.dtype) / 64
        indices = (torch.arange(2048, device="cuda", dtype=torch.int32) - 3).repeat(
            rows, 1
        )
        requests = torch.zeros(rows, device="cuda", dtype=torch.int32)
        slices = torch.tensor([[0, rows * 8, 0]], device="cuda", dtype=torch.int32)
        params = SimpleNamespace(
            is_prefill=True,
            query_requests=requests,
            page_size=128,
            shard_size=8,
            shard_rank=0,
        )
        block = SimpleNamespace(start=0, width=rows * 8, slices=slices)
        slots = torch.empty_like(indices)
        # Warm cuBLAS and the allocator for every M without calling Triton.
        scratch = torch.empty((rows, 128, 576), device="cuda", dtype=torch.bfloat16)
        torch.bmm(
            query[..., :128].transpose(0, 1),
            weight,
            out=scratch[:, :96, :512].transpose(0, 1),
        )
        del scratch
        cases[rows] = (q, kv, suffix, query, indices, params, block, slots)
    torch.cuda.synchronize()

    def operations(rows):
        q, kv, suffix, query, indices, params, block, slots = cases[rows]

        def map_slots():
            modules["slots"].map_dsa_chunk_slots(indices, params, block, slots)
            return slots

        return {
            "slots": map_slots,
            "absorbed_query": lambda: modules["query"].absorbed_query(
                query, weight, 512, 128
            ),
            "norm": lambda: grouped_q_kv_rmsnorm(
                q,
                kv,
                attention.q_norm.weight,
                attention.kv_norm.weight,
                1e-6,
                kv_suffix=suffix,
            ),
        }

    def correct(name, rows, output):
        q, kv, suffix, query, indices, _, _, _ = cases[rows]
        if name == "slots":
            expected = torch.where((indices >= 0) & (indices < rows * 8), indices, -1)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        elif name == "absorbed_query":
            expected = torch.zeros_like(output)
            expected[:, :96, :512] = torch.arange(512, device="cuda").remainder_(4) * 8
            expected[:, :96, 512:] = query[..., 128:]
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        else:
            for value in output:
                torch.testing.assert_close(
                    value, torch.ones_like(value), rtol=0, atol=0
                )
            torch.testing.assert_close(
                output[1].as_strided((rows, 576), (576, 1))[:, 512:],
                suffix,
                rtol=0,
                atol=0,
            )

    result = {
        "variant": variant,
        "source_sha256": identities,
        "environment": {
            "host": platform.node(),
            "gpu": props.name,
            "uuid": uuid,
            "capability": [props.major, props.minor],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "nccl": torch.cuda.nccl.version(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "telemetry_before": telemetry(uuid),
        "initial": {},
        "cold_requests": [],
        "steady": [],
    }
    stopped, occupancy = threading.Event(), []

    def monitor():
        while not stopped.is_set():
            try:
                lines = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        uuid,
                        "--query-compute-apps=gpu_uuid,pid",
                        "--format=csv,noheader",
                    ],
                    text=True,
                    timeout=5,
                ).splitlines()
                occupancy.append(
                    {
                        "time_ns": time.time_ns(),
                        "pids": [
                            pid.strip()
                            for line in lines
                            for owner, _, pid in [line.partition(",")]
                            if owner.strip() == uuid and pid.strip().isdigit()
                        ],
                    }
                )
            except Exception as error:
                occupancy.append({"error": str(error)})
            stopped.wait(1)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    _, result["norm_model_warmup"] = cold(
        lambda: modules["norm"]._warmup_grouped_norm(attention, set())
    )
    for name, function in operations(8192).items():
        output, measurement = cold(function)
        correct(name, 8192, output)
        result["initial"][name] = measurement
        del output
    for rows in ROWS:
        for name, function in operations(rows).items():
            output, measurement = cold(function)
            correct(name, rows, output)
            result["cold_requests"].append({"op": name, "rows": rows, **measurement})
            del output
    for _ in range(100):
        output = operations(8192)["absorbed_query"]()
    torch.cuda.synchronize()
    del output
    result["telemetry_steady_start"] = telemetry(uuid)
    for rows in ROWS:
        for name, function in operations(rows).items():
            result["steady"].append({"op": name, "rows": rows, **steady(function)})
    stopped.set()
    watcher.join(6)
    if watcher.is_alive() or any(
        "error" in x or len(x["pids"]) != 1 for x in occupancy
    ):
        raise RuntimeError(f"GPU occupancy was not exclusive: {occupancy}")
    result["gpu_occupancy"] = occupancy
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["telemetry_after"] = telemetry(uuid)
    result["correctness"] = {"exact_reference": True, "max_abs_error": 0}
    Path(destination).write_text(json.dumps(result, indent=2))


class K31PrefillJitABTest(unittest.TestCase):
    def test_ab(self):
        with tempfile.TemporaryDirectory(prefix="k31-jit-ab-") as scratch:
            output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", scratch))
            trials = []
            for trial, order in enumerate((("A", "B"), ("B", "A"))):
                for variant in order:
                    root = Path(scratch) / f"{trial}-{variant}"
                    root.mkdir()
                    destination = output / f"trial-{trial}-{variant}.json"
                    environment = dict(
                        os.environ,
                        TRITON_CACHE_DIR=str(root / "triton"),
                        DG_JIT_CACHE_DIR=str(root / "native"),
                        CUDA_CACHE_PATH=str(root / "cuda"),
                        PYTHONPATH=os.pathsep.join(sys.path),
                    )
                    completed = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--worker",
                            variant,
                            str(destination),
                        ],
                        env=environment,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=900,
                    )
                    (output / f"trial-{trial}-{variant}.log").write_text(
                        completed.stdout
                    )
                    self.assertEqual(completed.returncode, 0, completed.stdout[-12000:])
                    result = json.loads(destination.read_text())
                    result["trial"] = trial
                    trials.append(result)
                    print(f"AB completed trial={trial} variant={variant}", flush=True)
            self.assertEqual(len({x["environment"]["uuid"] for x in trials}), 1)
            report = {
                "baseline": BASELINE,
                "candidate": CANDIDATE,
                "rows": ROWS,
                "contract": {
                    "physical_gpus": 1,
                    "modeled_tp": 8,
                    "modeled_rank": 0,
                    "input": "synthetic values at trace M, FP8-projection BF16 output",
                    "chunk_bounds": "synthetic BS1 [0,8*M), TopK=2048",
                    "cold": "host wall including wrapper, allocation, JIT and CUDA synchronize",
                    "steady": "eager CUDA events and separate 32-call graph diagnostic; 30 samples",
                    "excludes": "full model, projection GEMM before norm, collectives and serving",
                },
                "trials": trials,
            }
            (output / "k31-prefill-jit-ab.json").write_text(
                json.dumps(report, indent=2)
            )
            for trial in (0, 1):
                for variant in ("A", "B"):
                    result = next(
                        x
                        for x in trials
                        if x["trial"] == trial and x["variant"] == variant
                    )
                    for op in ("slots", "absorbed_query", "norm"):
                        values = [x for x in result["cold_requests"] if x["op"] == op]
                        misses = sum(len(x["jit_misses"]) for x in values)
                        print(
                            f"AB trial={trial} variant={variant} op={op} "
                            f"new_shape_wall_ms={sum(x['wall_ms'] for x in values):.6f} jit_misses={misses}",
                            flush=True,
                        )
                        if variant == "B":
                            self.assertEqual(misses, 0, (op, values))
                        elif op != "norm":
                            self.assertEqual(misses, 9, (op, values))
                        else:
                            self.assertEqual(misses, 1, (op, values))
                    self.assertEqual(
                        len(result["initial"]["norm"]["jit_misses"]),
                        1 if variant == "A" else 0,
                    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
