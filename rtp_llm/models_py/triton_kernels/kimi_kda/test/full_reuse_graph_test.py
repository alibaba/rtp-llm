"""Exact math and stable graph reuse for heterogeneous high-pressure prefills."""

import itertools
import json
import os
import shutil
import statistics
import time
import unittest
from collections import Counter, OrderedDict
from pathlib import Path
from unittest import mock

import rtp_llm.models_py.triton_kernels.kimi_kda.chunk_copies as chunk_copies
import rtp_llm.models_py.triton_kernels.kimi_kda.chunk_delta_h as recurrence
import rtp_llm.models_py.triton_kernels.kimi_kda.full_reuse_graph as graph_module
import rtp_llm.models_py.triton_kernels.kimi_kda.state_update as state_update
import torch
from rtp_llm.models_py.modules.dsv4.forward_metadata import _FORWARD_METADATA
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk import chunk_kda
from rtp_llm.models_py.triton_kernels.kimi_kda.full_reuse_graph import (
    _BASE_LENGTHS,
    _GRAPHS,
    _PREPARED_SMALL_PAIRS,
    chunk_kda_full_reuse_graph,
    graph_stats,
)


class FullReuseGraphTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "GPU validation must not skip")
        # Each test owns its graph cache; unrelated test families must not
        # consume the headroom reserved by this test's preparation contract.
        torch.cuda.synchronize()
        _GRAPHS.clear()
        _BASE_LENGTHS.clear()
        _PREPARED_SMALL_PAIRS.clear()
        self.env = mock.patch.dict(os.environ, {"GLM53_KDA_FULL_REUSE_GRAPH": "1"})
        dispatch = mock.patch.object(graph_module, "_COMPILED_DISPATCH", True)
        compact = mock.patch.object(graph_module, "_COMPACT_SINGLE_LONG", True)
        compact.start()
        self.addCleanup(compact.stop)
        fused_state = mock.patch.object(state_update, "_ENABLED", True)
        fused_state.start()
        checkpoints = mock.patch.object(state_update, "_WRITE_CHECKPOINTS", True)
        checkpoints.start()
        self.addCleanup(checkpoints.stop)
        self.addCleanup(fused_state.stop)
        dispatch.start()
        self.addCleanup(dispatch.stop)
        self.env.start()
        self.addCleanup(self.env.stop)
        print(
            "GPU tools:",
            {name: shutil.which(name) for name in ("nsys", "ncu", "compute-sanitizer")},
        )
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.addCleanup(no_grad.__exit__, None, None, None)
        torch.manual_seed(531015)
        torch.backends.cuda.matmul.allow_tf32 = False

    def inputs(self, lengths, heads=2):
        shape = (1, sum(lengths), heads, 128)
        return dict(
            q=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
            k=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
            v=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
            g=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1 - 4,
            beta=torch.rand(shape[:3], device="cuda"),
            initial_state=torch.randn(len(lengths), heads, 128, 128, device="cuda")
            * 0.1,
            cu_seqlens=torch.tensor(
                [0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32
            ),
            A_log=torch.randn(heads, device="cuda") * 0.1,
            dt_bias=torch.randn(heads * 128, device="cuda") * 0.1,
        )

    def compare(self, values, beta_logits=False, **graph_options):
        # The reference must retain the unfused recurrence even when the graph
        # under test captures the new state-update kernel.
        reference_values = values
        if beta_logits:
            reference_values = {
                **values,
                "beta": values["beta"].float().sigmoid(),
            }
        with mock.patch.object(state_update, "_ENABLED", False), mock.patch.object(
            recurrence, "_REUSE_BIAS_OUTPUT", False
        ):
            expected = chunk_kda(
                **reference_values,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=-5.0,
                return_intermediate_states=True,
                fuse_state_recurrence=False,
                chunk_size=64,
            )
        actual = chunk_kda_full_reuse_graph(
            **values, _beta_logits=beta_logits, **graph_options
        )
        self.assertIsNotNone(actual, "eligible cache tails must not fall back")
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            bits = torch.int32 if a.element_size() == 4 else torch.int16
            self.assertTrue(torch.equal(a.view(bits), b.view(bits)))
        return actual

    def test_bf16_beta_logits_are_captured_exactly(self):
        for lengths in ((418, 419), (31, 63, 65, 129, 257)):
            with self.subTest(lengths=lengths):
                values = self.inputs(lengths, heads=4)
                # Packed projection returns a narrow view whose row stride is
                # the full packed output width. The graph packer must consume
                # that view directly instead of launching a contiguous copy.
                packed = torch.randn(
                    (1, values["beta"].shape[1], 23),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                values["beta"] = packed[..., 7:11]
                self.assertFalse(values["beta"].is_contiguous())
                actual = self.compare(values, beta_logits=True)
                frozen = tuple(t.clone() for t in actual)
                captures = graph_stats()["captures"]
                values["beta"].add_(
                    torch.tensor(0.5, device="cuda", dtype=torch.bfloat16)
                )
                changed = self.compare(values, beta_logits=True)
                self.assertEqual(graph_stats()["captures"], captures)
                self.assertTrue(
                    any(not torch.equal(a, b) for a, b in zip(changed, frozen))
                )

    def test_beta_logits_capture_benchmark(self):
        values = self.inputs((419,), heads=16)
        packed = torch.randn(
            (1, values["beta"].shape[1], 79),
            device="cuda",
            dtype=torch.bfloat16,
        )
        logits = packed[..., 31:47]
        self.assertFalse(logits.is_contiguous())

        def run(beta_logits):
            beta = logits if beta_logits else logits.float().sigmoid()
            return chunk_kda_full_reuse_graph(
                values["q"],
                values["k"],
                values["v"],
                values["g"],
                beta,
                values["initial_state"],
                values["cu_seqlens"],
                values["A_log"],
                values["dt_bias"],
                _beta_logits=beta_logits,
            )

        for enabled in (False, True):
            for _ in range(3):
                self.assertIsNotNone(run(enabled))
        torch.cuda.synchronize()
        captures = graph_stats()["captures"]
        samples = {False: [], True: []}
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for repeat in range(40):
            for enabled in ((False, True) if repeat % 2 else (True, False)):
                start.record()
                begin = time.perf_counter_ns()
                self.assertIsNotNone(run(enabled))
                cpu_us = (time.perf_counter_ns() - begin) / 1000
                end.record()
                end.synchronize()
                samples[enabled].append(
                    {
                        "cpu_us": cpu_us,
                        "event_us": start.elapsed_time(end) * 1000,
                    }
                )
        self.assertEqual(graph_stats()["captures"], captures)
        medians = {
            str(enabled): {
                metric: statistics.median(row[metric] for row in rows)
                for metric in ("cpu_us", "event_us")
            }
            for enabled, rows in samples.items()
        }
        print("KDA_BETA_LOGITS", medians, flush=True)
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "beta_logits.json"
        output.write_text(json.dumps({"samples": samples, "medians": medians}, indent=2))

    def test_exact_ragged_outputs_and_changed_weights(self):
        for lengths in (
            (1,),
            (31, 63, 64, 65, 129, 257, 512),
            (418,) * 7 + (419,) * 32,
            (75,) * 64,
            (513, 1024, 2048),
        ):
            with self.subTest(lengths=lengths):
                values = self.inputs(lengths)
                actual = self.compare(values)
                frozen = tuple(t.clone() for t in actual)
                for name in (
                    "q",
                    "k",
                    "v",
                    "g",
                    "beta",
                    "initial_state",
                    "A_log",
                    "dt_bias",
                ):
                    values[name].mul_(0.75)
                self.compare(values)
                for a, b in zip(actual, frozen):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_request_order_and_group_capacity_do_not_recapture(self):
        layouts = [
            (418,) * 7 + (419,) * 32,
            tuple(reversed((418,) * 7 + (419,) * 32)),
            (418,) * 8 + (419,) * 31,
            (418,) * 4 + (419,) * 35,
            (418,) * 2 + (419,) * 37,
            (418,) * 21 + (419,) * 18,
        ]
        self.compare(self.inputs(layouts[0]))
        captures = graph_stats()["captures"]
        for lengths in layouts[1:]:
            self.compare(self.inputs(lengths))
            self.assertEqual(
                graph_stats()["captures"],
                captures,
                "request order/frequency within capacity must reuse graph",
            )

    def test_heterogeneous_batch_compositions_reuse_length_groups(self):
        lengths = (31, 63, 65, 75, 129, 257, 419, 512)
        self.compare(self.inputs(lengths * 2, heads=16))
        self.compare(self.inputs(lengths, heads=16))
        captured = graph_stats()["captures"]
        for order in (lengths, lengths[::-1], lengths[2:] + lengths[:2]):
            self.compare(self.inputs(order, heads=16))
            self.assertEqual(graph_stats()["captures"], captured)
        # Changing weights and request order must not leak old outputs/states.
        self.assertGreater(graph_stats()["grouped_calls"], 0)

    def test_normalization_configuration_survives_capacity_padding(self):
        for lengths in ((419,) * 9, (129,) * 17, (256,) * 16, (256,) * 17):
            with self.subTest(lengths=lengths):
                self.compare(self.inputs(lengths, heads=16))

    def test_high_pressure_39_sequences_heads32(self):
        values = self.inputs((418,) * 7 + (419,) * 32, heads=32)
        self.compare(values)
        captured = graph_stats()["captures"]
        for _ in range(3):
            values["A_log"].add_(0.01)
            self.compare(values)
            self.assertEqual(graph_stats()["captures"], captured)
        print("High pressure graph stats:", graph_stats())

    def test_streams_and_cached_metadata_lifetimes(self):
        values = self.inputs((75, 129, 75))
        outputs = self.compare(values)
        frozen = tuple(t.clone() for t in outputs)
        # Churn FLA's external four-entry tensor cache; graph metadata must be
        # held by its own workspace, never only by that unrelated cache.
        for length in range(81, 88):
            self.compare(self.inputs((length,)))
        self.compare(values)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.compare(self.inputs((129, 75, 75)))
        torch.cuda.current_stream().wait_stream(stream)
        for a, b in zip(outputs, frozen):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_base_group_graph_prefix_overflow_and_owned_outputs(self):
        lengths = tuple(range(80, 420, 17)) * 3
        with mock.patch.dict(os.environ, {"GLM53_KDA_BASE_GROUP_GRAPH": "1"}):
            values = self.inputs(lengths, heads=16)
            outputs = self.compare(values)
            frozen = tuple(t.clone() for t in outputs)
            captured = graph_stats()["captures"]
            self.compare(self.inputs(lengths[::-1], heads=16))
            self.compare(self.inputs(tuple(range(80, 420, 17)) * 2, heads=16))
            self.assertEqual(graph_stats()["captures"], captured)
            for a, b in zip(outputs, frozen):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertGreater(graph_stats().get("base_replays", 0), 0)

    def test_base_family_keeps_new_lengths_in_independent_overflow(self):
        with mock.patch.dict(os.environ, {"GLM53_KDA_BASE_GROUP_GRAPH": "1"}):
            original = (87, 137, 187, 237) * 6
            self.compare(self.inputs(original, heads=4))
            added = (87, 137, 187, 237, 287, 337) * 4
            self.compare(self.inputs(added, heads=4))
            captured = graph_stats()["captures"]
            self.compare(self.inputs(added[::-1], heads=4))
            self.assertEqual(graph_stats()["captures"], captured)

    def test_one_two_request_overflow_share_exact_template(self):
        self.compare(self.inputs((97, 97)), _minimum_capacity=2)
        captured = graph_stats()["captures"]
        for lengths in ((97,), (97, 97), (97,)):
            self.compare(self.inputs(lengths), _minimum_capacity=2)
            self.assertEqual(graph_stats()["captures"], captured)

    def test_small_pair_capacities_are_prepared_before_steady_pressure(self):
        self.compare(self.inputs((80,) * 7 + (81,) * 56, heads=16))
        captured = graph_stats()["captures"]
        for n in (2, 3, 4, 7, 9, 15, 17, 23, 25, 33, 41, 49, 51, 52, 57, 63):
            self.compare(self.inputs((80,) + (81,) * (n - 1), heads=16))
            self.assertEqual(graph_stats()["captures"], captured)

    def test_single_length_groups_keep_both_normalization_templates_warm(self):
        # Preparation deliberately stops before the reserved cache headroom.
        # Exercise this cold-family contract with an empty owned test cache.
        torch.cuda.synchronize()
        _GRAPHS.clear()
        lengths = (96, 113, 130, 147, 164, 181, 198, 215)
        self.compare(self.inputs(lengths, heads=32))
        captured = graph_stats()["captures"]
        for values in (lengths[::-1], lengths * 2, lengths[2:] + lengths[:2]):
            self.compare(self.inputs(values, heads=32))
            self.assertEqual(graph_stats()["captures"], captured)

    def test_compiled_dispatch_unaligned_metadata_and_inputs(self):
        values = self.inputs((75, 129, 75))
        # Exercise every externally supplied pointer, including FP32 state and
        # bias. A contiguous tensor can still have an unaligned storage offset.
        for name, value in list(values.items()):
            backing = torch.empty(
                value.numel() + 1, device=value.device, dtype=value.dtype
            )
            view = backing[1:].view(value.shape)
            view.copy_(value)
            values[name] = view
        self.compare(values)
        values["cu_seqlens"] = values["cu_seqlens"].to(torch.int64)
        self.compare(values)

    def test_compiled_dispatch_device_and_pending_stream(self):
        self.assertGreaterEqual(
            torch.cuda.device_count(), min(2, int(os.environ.get("GPU_COUNT", "1")))
        )
        for device in range(min(2, torch.cuda.device_count())):
            with torch.cuda.device(device):
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    values = self.inputs((81, 81))
                    token = _FORWARD_METADATA.set({})
                    try:
                        expected = self.compare(values)
                        # Warm the metadata and both dispatchers on this stream
                        # before checking that replay introduces no host wait.
                        chunk_kda_full_reuse_graph(**values)
                        torch.cuda.synchronize()
                        torch.cuda._sleep(100_000_000)
                        pending = torch.cuda.Event()
                        pending.record()
                        actual = chunk_kda_full_reuse_graph(**values)
                        self.assertFalse(pending.query(), "replay synchronized the GPU")
                    finally:
                        _FORWARD_METADATA.reset(token)
                stream.synchronize()
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_compiled_dispatch_complete_wrapper_benchmark(self):
        rows = []
        for lengths in ((81,), (81,) * 63, (419,) * 3, (9889,)):
            values = self.inputs(lengths, heads=16)
            token = _FORWARD_METADATA.set({})
            try:
                outputs = []
                for enabled in (False, True):
                    graph_module._COMPILED_DISPATCH = enabled
                    for _ in range(3):
                        result = chunk_kda_full_reuse_graph(**values)
                    outputs.append(result)
                for a, b in zip(*outputs):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                baseline = graph_stats()
                samples = {False: [], True: []}
                for repeat in range(60):
                    for enabled in (
                        (False, True) if repeat % 2 == 0 else (True, False)
                    ):
                        graph_module._COMPILED_DISPATCH = enabled
                        torch.cuda.synchronize()
                        start, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        start.record()
                        begin = time.perf_counter_ns()
                        result = chunk_kda_full_reuse_graph(**values)
                        cpu_us = (time.perf_counter_ns() - begin) / 1000
                        end.record()
                        end.synchronize()
                        samples[enabled].append(
                            dict(cpu_us=cpu_us, event_us=start.elapsed_time(end) * 1000)
                        )
                stats = graph_stats()
                for key in ("captures", "evictions", "estimated_bytes"):
                    self.assertEqual(stats.get(key, 0), baseline.get(key, 0))
                row = dict(lengths=lengths, heads=16, samples=samples, graphs=stats)
                rows.append(row)
                print(
                    "KDA_DISPATCH",
                    len(lengths),
                    sum(lengths),
                    {
                        enabled: {
                            metric: statistics.median(r[metric] for r in sample)
                            for metric in ("cpu_us", "event_us")
                        }
                        for enabled, sample in samples.items()
                    },
                    flush=True,
                )
            finally:
                graph_module._COMPILED_DISPATCH = True
                _FORWARD_METADATA.reset(token)
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "kda_dispatch.json"
        output.write_text(json.dumps(rows, indent=2))

    def _state_update_graph_comparison(
        self,
        profile_only,
        checkpoint_output=False,
        bias_output=False,
        copy_output=False,
    ):
        directory = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        owner = (
            chunk_copies if copy_output else recurrence if bias_output else state_update
        )
        attribute = (
            "_REUSE_BIAS_OUTPUT"
            if bias_output
            else ("_WRITE_CHECKPOINTS" if checkpoint_output else "_ENABLED")
        )
        output_name = (
            "chunk_copies"
            if copy_output
            else (
                "bias_output"
                if bias_output
                else ("checkpoint_output" if checkpoint_output else "state_recurrence")
            )
        )
        old_bias = recurrence._REUSE_BIAS_OUTPUT
        old_copies = chunk_copies._ENABLED
        old_prefetch = chunk_copies._PREFETCH_STATE
        chunk_copies._PREFETCH_STATE = False
        rows = []
        for lengths in ((81,), (81,) * 63, (419,) * 3, (9889,)):
            values = self.inputs(lengths, heads=16)
            token = _FORWARD_METADATA.set({})
            caches, outputs, kernels = {}, {}, {}
            state_update._ENABLED = True
            state_update._WRITE_CHECKPOINTS = bias_output or copy_output
            chunk_copies._ENABLED = False
            chunk_copies._PREFETCH_STATE = False
            if copy_output:
                recurrence._REUSE_BIAS_OUTPUT = True
            try:
                for enabled in (False, True):
                    torch.cuda.synchronize()
                    # The flag changes captured code. Never compare replays
                    # from the same cache and accidentally test the old graph twice.
                    _GRAPHS.clear()
                    _BASE_LENGTHS.clear()
                    _PREPARED_SMALL_PAIRS.clear()
                    setattr(owner, attribute, enabled)
                    outputs[enabled] = self.compare(values)
                    caches[enabled] = OrderedDict(_GRAPHS)
                    for _ in range(3 if profile_only else 30):
                        chunk_kda_full_reuse_graph(**values)
                for old, new in zip(outputs[False], outputs[True]):
                    bits = torch.int32 if old.element_size() == 4 else torch.int16
                    self.assertTrue(torch.equal(old.view(bits), new.view(bits)))
                captures = graph_stats().get("captures", 0)
                samples = {False: [], True: []}
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                if not profile_only:
                    for repeat in range(60):
                        for enabled in ((False, True) if repeat % 2 else (True, False)):
                            torch.cuda.synchronize()
                            _GRAPHS.clear()
                            _GRAPHS.update(caches[enabled])
                            setattr(owner, attribute, enabled)
                            start.record()
                            begin = time.perf_counter_ns()
                            chunk_kda_full_reuse_graph(**values)
                            cpu_us = (time.perf_counter_ns() - begin) / 1000
                            end.record()
                            end.synchronize()
                            samples[enabled].append(
                                dict(
                                    cpu_us=cpu_us,
                                    event_us=start.elapsed_time(end) * 1000,
                                )
                            )
                    self.assertEqual(graph_stats().get("captures", 0), captures)
                if profile_only:
                    for enabled in (False, True):
                        _GRAPHS.clear()
                        _GRAPHS.update(caches[enabled])
                        setattr(owner, attribute, enabled)
                        torch.cuda.synchronize()
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as profile:
                            chunk_kda_full_reuse_graph(**values)
                            torch.cuda.synchronize()
                        path = (
                            directory
                            / f"{output_name}_{len(lengths)}_{sum(lengths)}_{enabled}.json"
                        )
                        profile.export_chrome_trace(str(path))
                        trace = json.loads(path.read_text())["traceEvents"]
                        events = [e for e in trace if e.get("cat") == "kernel"]
                        copies = [e for e in trace if e.get("cat") == "gpu_memcpy"]
                        kernels[enabled] = dict(
                            names_counts=dict(Counter(e["name"] for e in events)),
                            launch_count=len(events),
                            kernel_sum_us=sum(e["dur"] for e in events),
                            copy_count=len(copies),
                            copy_sum_us=sum(e["dur"] for e in copies),
                            copy_bytes=sum(e["args"]["bytes"] for e in copies),
                        )
                    old = kernels[False]["names_counts"]
                    new = kernels[True]["names_counts"]
                    fused_count = new.get("_state_update", 0)
                    self.assertGreater(fused_count, 0)
                    if bias_output or copy_output:
                        self.assertEqual(old.get("_state_update", 0), fused_count)

                        def gemms(counts):
                            return {
                                k: v
                                for k, v in counts.items()
                                if any(
                                    word in k.lower()
                                    for word in ("gemm", "cutlass", "cublas")
                                )
                            }

                        self.assertTrue(gemms(old))
                        self.assertEqual(gemms(old), gemms(new))
                        self.assertEqual(
                            kernels[False]["launch_count"]
                            + kernels[False]["copy_count"]
                            - kernels[True]["launch_count"]
                            - kernels[True]["copy_count"],
                            (3 if copy_output else 1) * fused_count,
                        )
                        if copy_output:
                            self.assertEqual(new.get("_prepare", 0), fused_count)
                            self.assertEqual(new.get("_finish", 0), fused_count)
                    elif checkpoint_output:
                        self.assertEqual(old.get("_state_update", 0), fused_count)
                        # CUDA may execute graph copies as memcpy kernels or
                        # copy-engine nodes. Count both, with unchanged state
                        # update launches; neither form may increase.
                        removed_kernels = (
                            kernels[False]["launch_count"]
                            - kernels[True]["launch_count"]
                        )
                        removed_copies = (
                            kernels[False]["copy_count"] - kernels[True]["copy_count"]
                        )
                        self.assertGreaterEqual(removed_kernels, 0)
                        self.assertGreaterEqual(removed_copies, 0)
                        self.assertEqual(removed_kernels + removed_copies, fused_count)
                        self.assertEqual(
                            kernels[False]["copy_bytes"] - kernels[True]["copy_bytes"],
                            removed_copies * 16 * 128 * 128 * 4,
                        )
                    else:
                        self.assertEqual(old.get("_state_update", 0), 0)
                        self.assertEqual(
                            sum(n for k, n in old.items() if "exp2_kernel" in k),
                            fused_count,
                        )
                        self.assertEqual(
                            sum(n for k, n in new.items() if "exp2_kernel" in k), 0
                        )
                        self.assertEqual(
                            kernels[False]["launch_count"]
                            - kernels[True]["launch_count"],
                            2 * fused_count,
                        )
                rows.append(
                    dict(lengths=lengths, heads=16, samples=samples, kernels=kernels)
                )
                print(
                    output_name,
                    len(lengths),
                    sum(lengths),
                    {
                        enabled: {
                            metric: statistics.median(r[metric] for r in sample)
                            for metric in ("cpu_us", "event_us")
                        }
                        for enabled, sample in samples.items()
                        if sample
                    },
                    flush=True,
                )
            finally:
                torch.cuda.synchronize()
                _GRAPHS.clear()
                caches.clear()
                state_update._ENABLED = True
                state_update._WRITE_CHECKPOINTS = True
                recurrence._REUSE_BIAS_OUTPUT = old_bias
                chunk_copies._ENABLED = old_copies
                chunk_copies._PREFETCH_STATE = old_prefetch
                _FORWARD_METADATA.reset(token)
        (
            directory
            / (
                f"{output_name}_kernels.json"
                if profile_only
                else f"{output_name}_benchmark.json"
            )
        ).write_text(json.dumps(rows, indent=2))

    def test_chunk_copy_graph_path_and_benchmark(self):
        with torch.enable_grad():
            self._state_update_graph_comparison(profile_only=False, copy_output=True)

    def test_z_chunk_copy_graph_kernel_counts(self):
        with torch.enable_grad():
            self._state_update_graph_comparison(profile_only=True, copy_output=True)

    def test_chunk_copy_fallbacks(self):
        with mock.patch.object(chunk_copies, "_ENABLED", True):
            state = torch.randn(2, 128, 128, device="cuda")
            u = torch.randn(65, 2, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn_like(u)
            self.assertIsNotNone(chunk_copies.prepare_chunk_copies(state, u, k))
            for args in (
                (state.half(), u, k),
                (state.transpose(1, 2), u, k),
                (state, u.float(), k),
                (state, u, k[:, :1]),
                (state, u[:, :, ::2], k[:, :, ::2]),
                (state, u[:0], k[:0]),
                (state.clone().requires_grad_(), u, k),
            ):
                self.assertIsNone(chunk_copies.prepare_chunk_copies(*args))
            value = torch.randn(2, 65, 128, device="cuda")
            target = torch.empty_like(u)
            self.assertIsNotNone(chunk_copies.finish_chunk_copies(value, target))
            for source, output in (
                (value, target.float()),
                (value.transpose(1, 2), target),
                (value, target[:, :, ::2]),
                (value.clone().requires_grad_(), target),
            ):
                self.assertIsNone(chunk_copies.finish_chunk_copies(source, output))

    def test_bias_output_graph_path_and_benchmark(self):
        # Framework prefill builds this graph with grad mode enabled and
        # non-differentiable inputs, unlike standalone inference_mode callers.
        with torch.enable_grad():
            self._state_update_graph_comparison(profile_only=False, bias_output=True)

    def test_z_bias_output_graph_kernel_counts(self):
        with torch.enable_grad():
            self._state_update_graph_comparison(profile_only=True, bias_output=True)

    def test_bias_output_grad_mode_and_differentiable_fallback(self):
        with torch.enable_grad(), mock.patch.object(
            state_update, "_ENABLED", False
        ), mock.patch.object(recurrence, "_REUSE_BIAS_OUTPUT", True):
            tensors = dict(
                k=torch.randn(1, 65, 2, 128, device="cuda", dtype=torch.bfloat16) * 0.1,
                w=torch.randn(1, 65, 2, 128, device="cuda", dtype=torch.bfloat16) * 0.1,
                u=torch.randn(1, 65, 2, 128, device="cuda", dtype=torch.bfloat16) * 0.1,
                gk=torch.randn(1, 65, 2, 128, device="cuda") * 0.1,
                initial_state=torch.randn(1, 2, 128, 128, device="cuda") * 0.1,
            )
            for differentiable in (None, *tensors):
                values = {name: value.clone() for name, value in tensors.items()}
                if differentiable is not None:
                    values[differentiable].requires_grad_()
                original = torch.baddbmm
                with mock.patch.object(torch, "baddbmm", wraps=original) as gemm:
                    actual = recurrence.chunk_gated_delta_rule_fwd_h_cublas(
                        **values, output_final_state=True
                    )
                self.assertEqual(gemm.call_count, 2)
                for call in gemm.call_args_list:
                    if differentiable is None:
                        self.assertIs(call.kwargs["out"], call.args[0])
                    else:
                        self.assertNotIn("out", call.kwargs)
                with mock.patch.object(recurrence, "_REUSE_BIAS_OUTPUT", False):
                    expected = recurrence.chunk_gated_delta_rule_fwd_h_cublas(
                        **values, output_final_state=True
                    )
                for a, b in zip(actual, expected):
                    bits = torch.int32 if a.element_size() == 4 else torch.int16
                    self.assertTrue(torch.equal(a.view(bits), b.view(bits)))
                if differentiable is not None:
                    losses = [
                        sum(t.float().sum() for t in result)
                        for result in (actual, expected)
                    ]

                    # This forward-only backend's out_dtype BLAS overload may
                    # reject backward. Preserve the original result OR exact
                    # error, rather than claiming it supports training.
                    def gradient_or_error(loss):
                        try:
                            return torch.autograd.grad(loss, values[differentiable])[0]
                        except RuntimeError as error:
                            return str(error)

                    expected_grad = gradient_or_error(losses[1])
                    actual_grad = gradient_or_error(losses[0])
                    if isinstance(expected_grad, str):
                        self.assertIn("derivative for aten::", expected_grad)
                        self.assertEqual(actual_grad, expected_grad)
                    else:
                        self.assertIsInstance(actual_grad, torch.Tensor)
                        self.assertTrue(torch.equal(actual_grad, expected_grad))
                for name, value in values.items():
                    self.assertTrue(torch.equal(value, tensors[name]))

    def test_fused_state_update_graph_path_and_benchmark(self):
        self._state_update_graph_comparison(profile_only=False)

    def test_z_fused_state_update_graph_kernel_counts(self):
        # Initialize CUPTI only after every unprofiled timing test has finished.
        self._state_update_graph_comparison(profile_only=True)

    def test_checkpoint_output_graph_path_and_benchmark(self):
        self._state_update_graph_comparison(profile_only=False, checkpoint_output=True)

    def test_z_checkpoint_output_graph_kernel_counts(self):
        self._state_update_graph_comparison(profile_only=True, checkpoint_output=True)

    @torch.inference_mode()
    def test_checkpoint_output_optional_states_and_layouts(self):
        from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_h_cublas,
        )

        for ragged in (False, True):
            shape = (1, 129, 2, 128) if ragged else (2, 129, 2, 128)
            values = dict(
                k=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
                w=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
                u=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
                gk=-4 * torch.rand(shape, device="cuda"),
                cu_seqlens=(
                    torch.tensor([0, 63, 129], device="cuda", dtype=torch.int32)
                    if ragged
                    else None
                ),
            )
            for transpose, dtype, final, new_value, initial in (
                (False, torch.float32, True, True, True),
                (False, torch.float32, False, False, False),
                (True, torch.float32, True, True, True),
                (False, torch.bfloat16, True, False, True),
            ):
                options = dict(
                    transpose_state_layout=transpose,
                    intermediate_state_dtype=dtype,
                    output_final_state=final,
                    save_new_value=new_value,
                    initial_state=(
                        torch.randn(2, 2, 128, 128, device="cuda") if initial else None
                    ),
                )
                with mock.patch.object(
                    state_update, "_WRITE_CHECKPOINTS", False
                ), mock.patch.object(recurrence, "_REUSE_BIAS_OUTPUT", False):
                    expected = chunk_gated_delta_rule_fwd_h_cublas(**values, **options)
                actual = chunk_gated_delta_rule_fwd_h_cublas(**values, **options)
                for old, new in zip(expected, actual):
                    if old is None:
                        self.assertIsNone(new)
                    else:
                        bits = torch.int32 if old.element_size() == 4 else torch.int16
                        self.assertTrue(torch.equal(old.view(bits), new.view(bits)))

    def test_single_long_capacity_exact_and_benchmark(self):
        records = []
        for length in (8192, 8193, 9889, 16383, 16384):
            torch.cuda.synchronize()
            _GRAPHS.clear()
            values = self.inputs((length,), heads=16)
            token = _FORWARD_METADATA.set({})
            try:
                outputs = []
                for capacity in (2, 1):
                    for _ in range(2):
                        result = chunk_kda_full_reuse_graph(
                            **values, _minimum_capacity=capacity
                        )
                    self.assertIsNotNone(result)
                    outputs.append(result)
                for a, b in zip(*outputs):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                frozen = tuple(t.clone() for t in outputs[1])
                values["A_log"].add_(0.01)
                values["initial_state"].mul_(0.75)
                changed = [
                    chunk_kda_full_reuse_graph(**values, _minimum_capacity=c)
                    for c in (2, 1)
                ]
                for a, b in zip(*changed):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                for a, b in zip(outputs[1], frozen):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                baseline = graph_stats()
                memory = {len(key[3]): entry[4] for key, entry in _GRAPHS.items()}
                self.assertEqual(set(memory), {1, 2})
                samples = {1: [], 2: []}
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                for repeat in range(40):
                    for capacity in ((1, 2) if repeat % 2 else (2, 1)):
                        torch.cuda.synchronize()
                        start.record()
                        begin = time.perf_counter_ns()
                        result = chunk_kda_full_reuse_graph(
                            **values, _minimum_capacity=capacity
                        )
                        cpu_us = (time.perf_counter_ns() - begin) / 1000
                        end.record()
                        end.synchronize()
                        samples[capacity].append(
                            dict(cpu_us=cpu_us, event_us=start.elapsed_time(end) * 1000)
                        )
                after = graph_stats()
                for key in ("captures", "evictions", "estimated_bytes"):
                    self.assertEqual(after.get(key, 0), baseline.get(key, 0))
                records.append(
                    dict(length=length, heads=16, samples=samples, graph_bytes=memory)
                )
                print(
                    "KDA_SINGLE_LONG",
                    length,
                    memory,
                    {
                        capacity: {
                            metric: statistics.median(v[metric] for v in sample)
                            for metric in ("cpu_us", "event_us")
                        }
                        for capacity, sample in samples.items()
                    },
                    flush=True,
                )
                # Warm current-stream replay must retain its asynchronous boundary.
                torch.cuda._sleep(100_000_000)
                pending = torch.cuda.Event()
                pending.record()
                chunk_kda_full_reuse_graph(**values, _minimum_capacity=1)
                self.assertFalse(pending.query())
                torch.cuda.synchronize()
            finally:
                _FORWARD_METADATA.reset(token)
        output = (
            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "kda_single_long.json"
        )
        output.write_text(json.dumps(records, indent=2))

    def test_single_long_default_capacity_gate(self):
        for lengths in ((8192,), (8193,), (16384,), (8192, 8192)):
            torch.cuda.synchronize()
            _GRAPHS.clear()
            values = self.inputs(lengths)
            with mock.patch.object(graph_module, "_COMPACT_SINGLE_LONG", False):
                expected = chunk_kda_full_reuse_graph(**values)
            actual = chunk_kda_full_reuse_graph(**values)
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            capacities = {len(key[3]) for key in _GRAPHS}
            compact = len(lengths) == 1 and lengths[0] > 8192
            self.assertEqual(capacities, {1, 2} if compact else {2})
        self.assertIsNone(chunk_kda_full_reuse_graph(**self.inputs((16385,))))

    def test_copy_grids_exceed_old_y_limit(self):
        # Both pack token tiles and unpack checkpoint tiles exceed 65535.
        values = self.inputs((16384,), heads=32)
        for compiled in (False, True):
            with mock.patch.object(graph_module, "_COMPILED_DISPATCH", compiled):
                self.compare(values, _minimum_capacity=1)

    def test_explicit_opt_out(self):
        with mock.patch.dict(os.environ, {"GLM53_KDA_FULL_REUSE_GRAPH": "0"}):
            self.assertIsNone(chunk_kda_full_reuse_graph(**self.inputs((75,))))

    def test_heterogeneous_frequency_changes_reuse_two_request_tiles(self):
        torch.cuda.synchronize()
        _GRAPHS.clear()
        lengths = (96, 113, 130, 147)
        self.compare(self.inputs(lengths, heads=8))
        captured = graph_stats()["captures"]
        for count in (3, 4, 7, 8):
            self.compare(self.inputs(lengths * count, heads=8))
            self.assertEqual(graph_stats()["captures"], captured)

    def test_short_single_length_drain_capacities_are_prepared(self):
        torch.cuda.synchronize()
        _GRAPHS.clear()
        self.compare(self.inputs((81,) * 63, heads=16))
        captured = graph_stats()["captures"]
        for count in (1, 2, 3, 5, 7, 9, 15, 17, 25, 31, 33, 51, 52, 63):
            self.compare(self.inputs((81,) * count, heads=16))
            self.assertEqual(graph_stats()["captures"], captured)

    def test_medium_single_length_drain_capacities_are_prepared(self):
        for length in (418, 419):
            self.compare(self.inputs((length,) * 39, heads=16))
        captured = graph_stats()["captures"]
        evictions = graph_stats().get("evictions", 0)
        for length in (418, 419):
            for count in (1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 39):
                self.compare(self.inputs((length,) * count, heads=16))
                self.assertEqual(graph_stats()["captures"], captured)
                self.assertEqual(graph_stats().get("evictions", 0), evictions)

    def test_medium_close_pairs_share_independent_length_templates(self):
        torch.cuda.synchronize()
        _GRAPHS.clear()
        self.compare(self.inputs((193, 210), heads=8))
        captured = graph_stats()["captures"]
        for lengths in ((193,) * 3 + (210,) * 10, (193, 210) * 7, (210, 193)):
            self.compare(self.inputs(lengths, heads=8))
            self.assertEqual(graph_stats()["captures"], captured)


if __name__ == "__main__":
    unittest.main()
