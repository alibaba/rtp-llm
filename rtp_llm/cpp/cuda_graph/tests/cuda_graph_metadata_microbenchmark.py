"""Paired timings of the production runner and real paged metadata hooks."""

import gc
import hashlib
import json
import os
import platform
import statistics
import time
import unittest
from pathlib import Path

import torch
from rtp_llm.cpp.cuda_graph.tests import libtest_cuda_graph_runner
from rtp_llm.cpp.cuda_graph.tests.cuda_graph_attention_prepare_test import (
    GROUP_TAGS,
    HIDDEN_SIZE,
    CudaGraphRunner,
    MetadataModel,
    make_inputs,
)

VARIANTS = (
    "legacy",
    "skip_host",
    "capture_only",
    "captured_with_host",
    "captured",
)


class MetadataHook:
    def __init__(self, probe, variant):
        self.impl, self.tokens = probe.impl, probe.tokens
        self.requires_host_metadata = variant == "legacy"
        self.can_capture_cuda_graph_prepare = variant in (
            "capture_only",
            "captured_with_host",
            "captured",
        )

    def prepare_cuda_graph(self, inputs):
        # Match the component adapter without its per-call counters/probes.
        inputs.is_cuda_graph = True
        inputs.physical_token_count = self.tokens
        self.impl.prepare_cuda_graph(inputs)


class BenchmarkModel(MetadataModel):
    def __init__(self, backend, variant, max_seq_len):
        super().__init__(backend, max_seq_len=max_seq_len)
        self.variant = variant

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        return {
            tag: MetadataHook(probe, self.variant)
            for tag, probe in super()
            .prepare_fmha_impl(inputs, is_cuda_graph=is_cuda_graph)
            .items()
        }


def build_runner(backend, role, batch, width, variant):
    model = BenchmarkModel(backend, variant, 128 * width)
    runner = CudaGraphRunner()
    if role == "draft_update":
        runner.init_mtp_draft_prefill(model, batch, 2, 1, 1, width, 128 * width)
    else:
        runner.init_decode(
            model,
            HIDDEN_SIZE,
            128 * width,
            width,
            [batch],
            GROUP_TAGS,
            role == "verify",
            2 if role == "verify" else 1,
        )
    return runner


def build_inputs(role, batch, width, device_metadata, alternate):
    inputs = make_inputs(role, batch, width)
    tagged = inputs.attention_inputs
    for attn in tagged.values():
        # Keep every query inside the declared physical page table. The
        # correctness fixture's original prefix ends one token beyond it.
        delta = 16 if width <= 4 else 512
        prefix = 128 * width - 2 - delta * alternate
        if role != "decode":
            attn.prefix_lengths = torch.full_like(attn.prefix_lengths, prefix)
            attn.sequence_lengths_plus_1_device.fill_(prefix + 1)
            attn.cu_kv_seqlens_device = torch.arange(
                batch + 1, device="cuda", dtype=torch.int32
            ) * (prefix + 2)
            attn.context_total_kv_length = batch * (prefix + 2)
        else:
            length = 128 * width - 1 - delta * alternate
            attn.sequence_lengths = torch.full_like(attn.sequence_lengths, length)
            attn.sequence_lengths_plus_1_device.fill_(length + 1)
        attn.kv_cache_kernel_block_id = (
            attn.kv_cache_kernel_block_id.clone() + 32 * alternate
        ).pin_memory()
        attn.kv_cache_kernel_block_id_device = attn.kv_cache_kernel_block_id.cuda()
        attn.kv_cache_block_id = attn.kv_cache_kernel_block_id
        attn.kv_cache_block_id_device = attn.kv_cache_kernel_block_id_device
        for field in (
            "input_lengths",
            "prefix_lengths",
            "sequence_lengths",
            "cu_seqlens",
        ):
            value = getattr(attn, field)
            setattr(
                attn, field, value.cuda() if device_metadata else value.pin_memory()
            )
        if device_metadata:
            attn.kv_cache_kernel_block_id = attn.kv_cache_kernel_block_id_device
            attn.kv_cache_block_id = attn.kv_cache_block_id_device
    inputs.attention_inputs = tagged
    return inputs


def step(runner, inputs, stream):
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        if not runner.prepare(inputs, skip_forward_event_sync=True):
            raise AssertionError("metadata benchmark missed its graph bucket")
    return runner.forward(inputs).hidden_states


def timed_step(runner, inputs, stream, start, end):
    torch.cuda.synchronize()
    start.record()
    # Include cross-stream ordering in all variants of the async experiment.
    stream.wait_event(start)
    begin = time.perf_counter_ns()
    with torch.cuda.stream(stream):
        if not runner.prepare(inputs, skip_forward_event_sync=True):
            raise AssertionError("metadata benchmark missed its graph bucket")
    prepared = time.perf_counter_ns()
    output = runner.forward(inputs).hidden_states
    submitted = time.perf_counter_ns()
    end.record()
    end.synchronize()
    finished = time.perf_counter_ns()
    if not output.numel():
        raise AssertionError("empty metadata checksum output")
    return {
        "prepare_wall_us": (prepared - begin) / 1000,
        "forward_submit_wall_us": (submitted - prepared) / 1000,
        "step_wall_us": (finished - begin) / 1000,
        "cuda_event_us": start.elapsed_time(end) * 1000,
    }


def summary(samples):
    return {
        metric: {
            "p50": statistics.median(values),
            "p90": sorted(values)[(9 * len(values) + 9) // 10 - 1],
            "min": min(values),
            "max": max(values),
        }
        for metric in samples[0]
        for values in [[sample[metric] for sample in samples]]
    }


class MetadataMicrobenchmark(unittest.TestCase):
    def test_paired_metadata_paths(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability()[0], 10)
        torch.set_num_threads(1)
        output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"))
        output_dir.mkdir(parents=True, exist_ok=True)
        props = torch.cuda.get_device_properties(0)
        async_prepare = os.environ.get("RTP_LLM_STREAM_ASYNC") == "1"
        stream = torch.cuda.Stream() if async_prepare else torch.cuda.current_stream()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        result = {
            "contract": {
                "boundary": "prepare -> graph replay -> GPU completion; metadata checksum body",
                "scope": "one GPU, logical Page-RR TP2 rank0, two cache tags, hidden width 4, query width 1/2; no model weights, attention compute or collectives",
                "variants": {
                    "legacy": "force host mirrors; prepare GPU metadata eagerly",
                    "skip_host": "disable host mirrors; prepare GPU metadata eagerly",
                    "capture_only": "retain host copies/clears and legacy D2H CPU fence; prepare GPU metadata in the consuming graph (test-only independent control)",
                    "captured_with_host": "retain host copies/clears, omit the unnecessary CPU fence; prepare GPU metadata in the consuming graph",
                    "captured": "disable host mirrors; prepare GPU metadata in the consuming graph",
                },
                "baseline_method": "same production binary; capability gates plus test-only post-capture host-mirror override and legacy CPU fence; not historical binaries",
                "ordering": [list(VARIANTS), list(reversed(VARIANTS))],
                "excluded": "input construction, graph capture, compilation, warmup and profiling",
                "cuda_event_boundary": "whole device interval including host launch gaps and stream waits; not pure kernel sum",
                "metadata_location": "lengths and table mirrors; padding_offset remains CUDA for both locations",
                "primary_metric": "step_wall_us",
                "samples_per_variant_per_round": 30,
                "warmup_steps_per_variant": 6,
            },
            "environment": {
                "hostname": platform.node(),
                "gpu": props.name,
                "gpu_uuid": str(props.uuid),
                "capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "async_prepare": async_prepare,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "runner_dso": libtest_cuda_graph_runner.__file__,
                "runner_dso_sha256": hashlib.sha256(
                    Path(libtest_cuda_graph_runner.__file__).read_bytes()
                ).hexdigest(),
            },
            "cases": [],
        }
        for backend in ("mla", "mla_compact", "dsa", "dsa_fp8"):
            for role in ("decode", "verify", "draft_update"):
                for batch, width in ((1, 4), (32, 4), (128, 4), (32, 512)):
                    runners = {
                        name: build_runner(backend, role, batch, width, name)
                        for name in VARIANTS
                    }
                    for device_metadata in (False, True):
                        runners["capture_only"].retain_host_metadata_for_benchmark(
                            device_metadata or role == "draft_update"
                        )
                        runners[
                            "captured_with_host"
                        ].retain_host_metadata_for_benchmark(False)
                        inputs = [
                            build_inputs(
                                role,
                                max(1, batch - alternate),
                                width,
                                device_metadata,
                                alternate,
                            )
                            for alternate in (0, 1)
                        ]
                        # Validate both content updates before measuring.
                        reference = []
                        for inp in inputs:
                            expected = step(runners["legacy"], inp, stream).clone()
                            for name in VARIANTS[1:]:
                                actual = step(runners[name], inp, stream)
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )
                            reference.append(expected)
                        self.assertFalse(torch.equal(*reference))
                        torch.cuda.synchronize()
                        scopes = {}
                        for name, runner in runners.items():
                            with torch.profiler.profile(
                                activities=[torch.profiler.ProfilerActivity.CPU]
                            ) as prof:
                                step(runner, inputs[0], stream)
                            keys = {event.key for event in prof.key_averages()}
                            prefix = "cuda_graph.prepareAttentionInputs("
                            self.assertEqual(
                                prefix + "host_mirror_copy)" in keys,
                                name
                                in (
                                    "legacy",
                                    "capture_only",
                                    "captured_with_host",
                                ),
                            )
                            self.assertEqual(
                                prefix + "wait_host_mirror_d2h)" in keys,
                                name in ("legacy", "capture_only")
                                and (device_metadata or role == "draft_update"),
                            )
                            self.assertEqual(
                                prefix + "prepare_cuda_graph)" in keys,
                                name in ("legacy", "skip_host"),
                            )
                            scopes[name] = sorted(
                                k for k in keys if k.startswith(prefix)
                            )
                            for i in range(6):
                                step(runner, inputs[i % 2], stream)
                        torch.cuda.synchronize()
                        rounds = []
                        for order in (VARIANTS, tuple(reversed(VARIANTS))):
                            raw = {name: [] for name in VARIANTS}
                            for i in range(30):
                                for name in order:
                                    raw[name].append(
                                        timed_step(
                                            runners[name],
                                            inputs[i % 2],
                                            stream,
                                            start,
                                            end,
                                        )
                                    )
                            rounds.append(
                                {
                                    "order": list(order),
                                    "raw": raw,
                                    "summary_us": {
                                        name: summary(values)
                                        for name, values in raw.items()
                                    },
                                }
                            )
                        case = {
                            "backend": backend,
                            "role": role,
                            "batch": batch,
                            "actual_batch_pattern": [batch, max(1, batch - 1)],
                            "table_width": width,
                            "page_size": 128,
                            "metadata_location": (
                                "cuda" if device_metadata else "pinned_host"
                            ),
                            "correctness": "exact equality for both changing inputs",
                            "scopes": scopes,
                            "rounds": rounds,
                        }
                        result["cases"].append(case)
                        (output_dir / "metadata_microbenchmark.json").write_text(
                            json.dumps(result, indent=2)
                        )
                        medians = {
                            name: statistics.median(
                                sample["step_wall_us"]
                                for rnd in rounds
                                for sample in rnd["raw"][name]
                            )
                            for name in VARIANTS
                        }
                        print(
                            json.dumps(
                                {
                                    "backend": backend,
                                    "role": role,
                                    "batch": batch,
                                    "table_width": width,
                                    "metadata_location": case["metadata_location"],
                                    "step_wall_p50_us": medians,
                                }
                            ),
                            flush=True,
                        )
                    del runners
                    gc.collect()
        result["status"] = "PASSED"
        (output_dir / "metadata_microbenchmark.json").write_text(
            json.dumps(result, indent=2)
        )


if __name__ == "__main__":
    unittest.main()
