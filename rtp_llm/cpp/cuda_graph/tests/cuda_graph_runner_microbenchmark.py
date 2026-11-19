"""Real runner preparation/replay timings with synthetic attention backends.

This measures metadata and graph overhead, not full-model token throughput.
Raw samples and device identity are emitted as Bazel undeclared outputs.
"""

import ctypes
import gc
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import unittest
from pathlib import Path

import torch

from rtp_llm.ops.compute_ops import LinearReplayInputs


def load_fixtures():
    # Separate processes can load a saved runner before the test fixtures import
    # the current library. Production configuration is unaffected.
    baseline_library = os.environ.get("CUDA_GRAPH_BENCHMARK_BASELINE_LIBRARY")
    if baseline_library:
        module_name = "rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner"
        spec = importlib.util.spec_from_file_location(module_name, baseline_library)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"invalid runner library: {baseline_library}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return (
        importlib.import_module(
            "rtp_llm.cpp.cuda_graph.tests.cuda_graph_tagged_cache_test"
        ),
        importlib.import_module(
            "rtp_llm.cpp.cuda_graph.tests.cuda_graph_attention_prepare_test"
        ),
    )


fixtures, metadata = load_fixtures()


def make_inputs(role, batch, step):
    values = {"full": step + 1, "aux": step + 2}
    if role == "decode":
        inputs = fixtures._build_decode_inputs(
            fixtures.GROUP_TAGS, values, batch_size=batch
        )
    else:
        inputs = fixtures._build_target_verify_inputs(
            fixtures.GROUP_TAGS,
            values,
            batch_size=batch,
            query_len=4,
            prefix_len=step % 4,
        )
    if role == "verify_large_hidden":
        inputs.input_hiddens = torch.empty(
            (batch * 4, 16384), dtype=torch.bfloat16, device="cuda"
        )
    if role.startswith("draft"):
        attention = inputs.attention_inputs["full"]
        attention.is_target_verify = False
        attention.is_mtp_draft_update = True
        if role == "draft":
            inputs.attention_inputs = attention
        else:
            tagged = inputs.attention_inputs
            for item in tagged.values():
                item.is_target_verify = False
                item.is_mtp_draft_update = True
                item.valid_token_mask = torch.ones(
                    batch * 4, dtype=torch.bool, device="cuda"
                )
            inputs.attention_inputs = tagged
        inputs.combo_position_ids = (
            attention.prefix_lengths.cuda()[:, None]
            + torch.arange(4, dtype=torch.int32, device="cuda")
        ).reshape(-1)
    # Match the device-metadata pipeline. The test wrapper's .cuda() is then a
    # no-op; benchmark timings do not include fixture-only H2D allocations.
    attentions = inputs.attention_inputs
    if not isinstance(attentions, dict):
        attentions = {"full": attentions}
    for tag, attention in attentions.items():
        attention.input_lengths = attention.input_lengths.cuda()
        attention.prefix_lengths = attention.prefix_lengths.cuda()
        attention.cu_seqlens = attention.cu_seqlens_device
        attention.kv_cache_kernel_block_id = attention.kv_cache_kernel_block_id_device
        if role == "verify_wide":
            attention.kv_cache_kernel_block_id_device = torch.full(
                (batch, 1024),
                values[tag],
                dtype=torch.int32,
                device="cuda",
            )
            attention.kv_cache_kernel_block_id = (
                attention.kv_cache_kernel_block_id_device
            )
        if role == "decode":
            attention.sequence_lengths = attention.sequence_lengths.cuda()
        if role == "replay":
            replay = LinearReplayInputs()
            for name in (
                "slot_ids",
                "prev_accept_lengths",
                "history_valid_lengths",
                "history_log_banks",
                "init_kinds",
                "anchor_processed_lengths",
            ):
                setattr(
                    replay, name, torch.zeros(batch, dtype=torch.int32, device="cuda")
                )
            replay.active_block_ids = torch.ones(
                (2, batch), dtype=torch.int32, device="cuda"
            )
            replay.state_read_block_ids = torch.ones(
                (2, batch), dtype=torch.int32, device="cuda"
            )
            attention.linear_replay = replay
    inputs.attention_inputs = attentions if role != "draft" else attentions["full"]
    inputs.input_hiddens.fill_(step)
    return inputs


def measure(call, iterations):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start = time.perf_counter_ns()
    start_event.record()
    for _ in range(iterations):
        call()
    end_event.record()
    end_event.synchronize()
    wall_us = (time.perf_counter_ns() - start) / (1000 * iterations)
    # The stream interval includes host submission gaps. It is not pure kernel
    # duration; the paired C++ benchmark isolates that separately.
    stream_us = start_event.elapsed_time(end_event) * 1000 / iterations
    return wall_us, stream_us


class RunnerMicrobenchmark(unittest.TestCase):
    def test_prepare_and_replay(self):
        self.assertTrue(torch.cuda.is_available(), "remote GPU execution is required")
        properties = torch.cuda.get_device_properties(0)
        library = Path(
            sys.modules[
                "rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner"
            ].__file__
        )
        library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
        identity = {
            "gpu": properties.name,
            "uuid": str(getattr(properties, "uuid", "unavailable")),
            "torch": torch.__version__,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_num_threads": torch.get_num_threads(),
            "source_sha256": os.environ.get("RUNNER_SOURCE_SHA256", "unknown"),
            "kernel_source_sha256": os.environ.get(
                "RUNNER_KERNEL_SOURCE_SHA256", "unknown"
            ),
            "library_sha256": library_hash,
            "variant": os.environ.get("RUNNER_BENCHMARK_VARIANT", "working_tree"),
            "dtype": "bfloat16",
        }
        output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        with (output_dir / "runner_samples.jsonl").open("w") as output:
            for role in (
                "decode",
                "verify",
                "verify_wide",
                "verify_large_hidden",
                "replay",
                "draft",
                "draft_sp",
            ):
                for capability in ("host", "eager", "capture"):
                    model_type = (
                        metadata.MetadataMtpModel
                        if role.startswith("draft")
                        else metadata.MetadataModel
                    )
                    model = model_type((capability, capability))
                    runner = fixtures.CudaGraphRunner()
                    is_verify = role.startswith("verify") or role == "replay"
                    width = (
                        16384 if role == "verify_large_hidden" else fixtures.HIDDEN_SIZE
                    )
                    if role == "draft_sp":
                        runner.init_mtp_draft_prefill_sp(model, 128, 4, 8, 1)
                    elif role == "draft":
                        runner.init_mtp_draft_prefill(
                            model, 128, fixtures.HIDDEN_SIZE, 4, 1
                        )
                    else:
                        runner.init_decode(
                            model,
                            width,
                            128,
                            (
                                1024
                                if role == "verify_wide"
                                else fixtures.KERNEL_BLOCK_TABLE_WIDTH
                            ),
                            [8, 32, 128],
                            fixtures.GROUP_TAGS,
                            is_verify,
                            4 if is_verify else 1,
                            0,
                            128,
                            2 if role == "replay" else 0,
                        )
                    batches = (
                        (1, 31) if role == "verify_large_hidden" else (128, 1, 31, 127)
                    )
                    for batch in batches:
                        with self.subTest(
                            role=role, capability=capability, batch=batch
                        ):
                            inputs = make_inputs(role, batch, 3)
                            self.assertTrue(runner.canRun(inputs))
                            query_width = 1 if role == "decode" else 4
                            graph_size = runner.getCurrentRealGraphSize()
                            capture_tokens = (
                                graph_size
                                if role.startswith("draft")
                                else graph_size * query_width
                            )
                            tail_bytes = (
                                (capture_tokens - inputs.input_ids.numel())
                                * width
                                * inputs.input_hiddens.element_size()
                            )
                            first = runner.forward(inputs).hidden_states.clone()
                            torch.cuda.synchronize()
                            # Validate changing metadata after both increasing
                            # and decreasing batch sizes, outside timed regions.
                            changed = make_inputs(role, batch, 5)
                            self.assertTrue(runner.canRun(changed))
                            second = runner.forward(changed).hidden_states.clone()
                            torch.cuda.synchronize()
                            self.assertTrue(torch.all(second != first).item())
                            self.assertTrue(runner.canRun(inputs))
                            self.assertTrue(runner.prepare(inputs, True))
                            operations = {
                                "prepare": lambda: runner.prepareAttentionInputs(
                                    inputs
                                ),
                                "async_prepare": lambda: runner.prepare(inputs, True),
                                "forward": lambda: runner.forward(inputs),
                            }
                            for operation, call in operations.items():
                                for _ in range(30):
                                    call()
                                for sample in range(9):
                                    wall_us, stream_us = measure(call, 100)
                                    record = dict(
                                        identity,
                                        role=role,
                                        capability=capability,
                                        batch=batch,
                                        query_width=query_width,
                                        groups=1 if role == "draft" else 2,
                                        sequence_parallel_size=(
                                            8 if role == "draft_sp" else 1
                                        ),
                                        linear_replay_groups=(
                                            2 if role == "replay" else 0
                                        ),
                                        capture_block_table_width=(
                                            1024
                                            if role == "verify_wide"
                                            else fixtures.KERNEL_BLOCK_TABLE_WIDTH
                                        ),
                                        source_block_table_width=(
                                            1024 if role == "verify_wide" else 1
                                        ),
                                        hidden_width=width,
                                        graph_size=graph_size,
                                        capture_token_capacity=capture_tokens,
                                        hidden_tail_bytes=tail_bytes,
                                        operation=operation,
                                        sample=sample,
                                        iterations=100,
                                        wall_us=wall_us,
                                        stream_us=stream_us,
                                    )
                                    output.write(json.dumps(record) + "\n")
                            torch.testing.assert_close(
                                runner.forward(inputs).hidden_states,
                                first,
                                rtol=0,
                                atol=0,
                            )
                    del runner, model
                    gc.collect()
                    torch.cuda.empty_cache()


def compare_saved_runner():
    """ABBA subprocesses share the parent Bazel action's GPU-lock allocation."""
    cpu = ctypes.CDLL(None).sched_getcpu()
    if cpu not in os.sched_getaffinity(0):
        raise RuntimeError("cannot select an allowed CPU for the paired benchmark")
    os.sched_setaffinity(0, {cpu})
    root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    saved = Path(os.environ["CUDA_GRAPH_BENCHMARK_SAVED_LIBRARY"]).resolve()
    if not saved.is_file():
        raise RuntimeError(f"missing baseline library: {saved}")
    records = []
    for index, variant in enumerate(("baseline", "candidate", "candidate", "baseline")):
        directory = root / f"{index}_{variant}"
        directory.mkdir()
        env = dict(os.environ)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = "1"
        env.pop("CUDA_GRAPH_BENCHMARK_SAVED_LIBRARY", None)
        env["TEST_UNDECLARED_OUTPUTS_DIR"] = str(directory)
        env["RUNNER_BENCHMARK_VARIANT"] = variant
        if variant == "baseline":
            env["CUDA_GRAPH_BENCHMARK_BASELINE_LIBRARY"] = str(saved)
            env["RUNNER_SOURCE_SHA256"] = os.environ["RUNNER_BASELINE_SOURCE_SHA256"]
            env["RUNNER_KERNEL_SOURCE_SHA256"] = os.environ[
                "RUNNER_BASELINE_KERNEL_SOURCE_SHA256"
            ]
        with (directory / "process.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve())],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        records.extend(
            json.loads(line)
            for line in (directory / "runner_samples.jsonl").read_text().splitlines()
        )
    identities = {(r["gpu"], r["uuid"]) for r in records}
    if len(identities) != 1 or next(iter(identities))[1] == "unavailable":
        raise RuntimeError(f"paired runs did not identify the same GPU: {identities}")
    libraries = {
        variant: {r["library_sha256"] for r in records if r["variant"] == variant}
        for variant in ("baseline", "candidate")
    }
    if (
        any(len(values) != 1 for values in libraries.values())
        or libraries["baseline"] == libraries["candidate"]
    ):
        raise RuntimeError(
            f"paired runs did not load two distinct, stable libraries: {libraries}"
        )
    keys = sorted(
        {(r["role"], r["capability"], r["batch"], r["operation"]) for r in records}
    )
    comparisons = []
    for key in keys:
        medians = {}
        for variant in ("baseline", "candidate"):
            samples = [
                r["wall_us"]
                for r in records
                if (r["role"], r["capability"], r["batch"], r["operation"]) == key
                and r["variant"] == variant
            ]
            if len(samples) != 18:
                raise RuntimeError(
                    f"incomplete samples: {variant} {key}: {len(samples)}"
                )
            medians[variant] = statistics.median(samples)
        # A gate on batched, end-to-end metadata overhead. Preserve raw samples
        # so a failure can be attributed rather than retried blindly.
        limit = medians["baseline"] * 1.05 + 2.0
        comparisons.append(
            dict(
                role=key[0],
                capability=key[1],
                batch=key[2],
                operation=key[3],
                **medians,
                ratio=medians["candidate"] / medians["baseline"],
                limit_us=limit,
                regression=medians["candidate"] > limit,
            )
        )
    summary = dict(
        gpu=list(identities)[0],
        libraries=libraries,
        threshold="candidate median <= baseline median * 1.05 + 2 us",
        comparisons=comparisons,
    )
    summary["libraries"] = {key: list(value)[0] for key, value in libraries.items()}
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    regressions = [r for r in comparisons if r["regression"]]
    print(json.dumps(dict(cases=len(comparisons), regressions=regressions), indent=2))
    if regressions:
        raise RuntimeError("runner preparation/replay performance gate failed")


if __name__ == "__main__":
    if os.environ.get("CUDA_GRAPH_BENCHMARK_SAVED_LIBRARY"):
        compare_saved_runner()
    else:
        unittest.main(failfast=True)
