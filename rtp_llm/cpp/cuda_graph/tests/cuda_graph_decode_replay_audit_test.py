"""Baseline audit of CudaGraphRunner decode prepare, capture, and replay."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from collections import Counter

import torch

from rtp_llm.cpp.cuda_graph.tests.cuda_graph_tagged_cache_test import (
    GROUP_TAGS,
    HIDDEN_SIZE,
    KERNEL_BLOCK_TABLE_WIDTH,
    TOKENS_PER_BLOCK,
    TaggedBlockTableModel,
    _build_decode_inputs,
    _build_target_verify_inputs,
)
from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner


class _DeviceOnlyFmhaImpl:
    """Device-only FMHA impl stub matching DSv4DecodeFmhaImplFP8 graph contract."""

    def support_cuda_graph(self) -> bool:
        return True

    def requires_host_attention_inputs(self) -> bool:
        return False

    def uses_in_graph_prepare(self) -> bool:
        return True

    def prepare_cuda_graph(self, attn_inputs) -> None:
        raise RuntimeError(
            "prepare_cuda_graph must not run when uses_in_graph_prepare=True"
        )


class DeviceOnlyDecodeModel:
    """Graph-safe decode model that reads only device metadata and block tables."""

    def prepare_fmha_impl(
        self,
        inputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: str | None = None,
    ):
        return _DeviceOnlyFmhaImpl()

    def forward(self, inputs, fmha_impl=None):
        from rtp_llm.ops.compute_ops import PyModelOutputs

        attn = inputs.attention_inputs
        full_bt = attn["full"].kv_cache_kernel_block_id_device
        aux_bt = attn["aux"].kv_cache_kernel_block_id_device
        seq_p1 = attn["full"].sequence_lengths_plus_1_device.to(torch.int32)
        per_row = (full_bt[:, 0] + 16 * aux_bt[:, 0] + 64 * seq_p1).to(
            inputs.input_hiddens.dtype
        )
        return PyModelOutputs(inputs.input_hiddens + per_row[:, None])


def _expected_device_only_output(
    inputs, full_val: int, aux_val: int, seq_p1_val: int, bs: int
) -> torch.Tensor:
    sig = full_val + 16 * aux_val + 64 * seq_p1_val
    return inputs.input_hiddens + torch.full_like(inputs.input_hiddens, sig)[:bs]


def _run_probe(drop_broad: str, stream_async: str) -> None:
    os.environ["RTP_LLM_DROP_BROAD_SYNC"] = drop_broad
    os.environ["RTP_LLM_STREAM_ASYNC"] = stream_async

    buckets = [1, 8, 32, 64, 96]
    runner = CudaGraphRunner()
    t0 = time.perf_counter()
    runner.init_decode(
        TaggedBlockTableModel(),
        HIDDEN_SIZE,
        TOKENS_PER_BLOCK,
        KERNEL_BLOCK_TABLE_WIDTH,
        buckets,
        GROUP_TAGS,
    )
    torch.cuda.synchronize()
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[CG_AUDIT] init_decode(drop={drop_broad}, async={stream_async}) init_ms={init_ms:.2f}"
    )

    for bs in buckets:
        inputs = _build_decode_inputs(
            GROUP_TAGS,
            {"full": bs + 1, "aux": bs + 2},
            batch_size=bs,
        )
        # Warmup
        for _ in range(5):
            assert runner.prepare(inputs)
            _ = runner.forward(inputs)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            assert runner.prepare(inputs)
            _ = runner.forward(inputs)
        torch.cuda.synchronize()

        events: Counter[str] = Counter()
        for evt in prof.events():
            kl = evt.name.lower()
            if any(
                tok in kl
                for tok in (
                    "memset",
                    "memcpy",
                    "eventsynchronize",
                    "cuda_graph.",
                    "indexselect",
                    "graphlaunch",
                    "zero_",
                    "fill_",
                )
            ):
                events[evt.name] += 1

        latencies_us: list[float] = []
        for _ in range(100):
            ts = time.perf_counter()
            runner.prepare(inputs)
            _ = runner.forward(inputs)
            latencies_us.append((time.perf_counter() - ts) * 1e6)
        torch.cuda.synchronize()
        ts_batch = time.perf_counter()
        for _ in range(100):
            runner.prepare(inputs)
            _ = runner.forward(inputs)
        torch.cuda.synchronize()
        batched_sync_us = ((time.perf_counter() - ts_batch) * 1e6) / 100.0
        latencies_us.sort()
        print(
            f"[CG_AUDIT] drop={drop_broad} async={stream_async} bs={bs} "
            f"prep+fwd_host_med={latencies_us[50]:.2f}us mean={sum(latencies_us)/100:.2f}us "
            f"batched_sync={batched_sync_us:.2f}us events={dict(events)}"
        )

    # Also test target_verify q_len=4
    tv_runner = CudaGraphRunner()
    tv_runner.init_decode(
        TaggedBlockTableModel(),
        HIDDEN_SIZE,
        TOKENS_PER_BLOCK,
        KERNEL_BLOCK_TABLE_WIDTH,
        [1, 8, 32],
        GROUP_TAGS,
        True,
        4,
    )
    tv_inputs = _build_target_verify_inputs(
        GROUP_TAGS,
        {"full": 3, "aux": 5},
        batch_size=8,
        query_len=4,
        prefix_len=2,
    )
    for _ in range(5):
        assert tv_runner.prepare(tv_inputs)
        _ = tv_runner.forward(tv_inputs)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        assert tv_runner.prepare(tv_inputs)
        _ = tv_runner.forward(tv_inputs)
    torch.cuda.synchronize()
    tv_events = Counter(
        evt.name
        for evt in prof.events()
        if any(
            tok in evt.name.lower()
            for tok in (
                "memset",
                "memcpy",
                "eventsynchronize",
                "cuda_graph.",
                "indexselect",
                "graphlaunch",
            )
        )
    )
    print(
        f"[CG_AUDIT] target_verify drop={drop_broad} async={stream_async} events={dict(tv_events)}"
    )

    # Device-only decode runner matching DSv4 in-graph metadata path (needs_host_mirrors=False)
    dev_runner = CudaGraphRunner()
    dev_runner.init_decode(
        DeviceOnlyDecodeModel(),
        HIDDEN_SIZE,
        TOKENS_PER_BLOCK,
        KERNEL_BLOCK_TABLE_WIDTH,
        buckets,
        GROUP_TAGS,
    )
    torch.cuda.synchronize()
    for bs in buckets:
        inputs = _build_decode_inputs(
            GROUP_TAGS,
            {"full": bs + 1, "aux": bs + 2},
            batch_size=bs,
            block_count=KERNEL_BLOCK_TABLE_WIDTH,
        )
        inputs.attention_inputs["full"].sequence_lengths_plus_1_device.fill_(bs % 7 + 2)
        for _ in range(5):
            assert dev_runner.prepare(inputs)
            _ = dev_runner.forward(inputs)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            assert dev_runner.prepare(inputs)
            _ = dev_runner.forward(inputs)
        torch.cuda.synchronize()

        dev_events: Counter[str] = Counter()
        for evt in prof.events():
            kl = evt.name.lower()
            if any(
                tok in kl
                for tok in (
                    "memset",
                    "memcpy",
                    "eventsynchronize",
                    "cuda_graph.",
                    "graphlaunch",
                    "zero_",
                    "fill_",
                )
            ):
                dev_events[evt.name] += 1

        latencies_us = []
        for _ in range(100):
            ts = time.perf_counter()
            dev_runner.prepare(inputs)
            _ = dev_runner.forward(inputs)
            latencies_us.append((time.perf_counter() - ts) * 1e6)
        torch.cuda.synchronize()
        ts_batch = time.perf_counter()
        for _ in range(100):
            dev_runner.prepare(inputs)
            _ = dev_runner.forward(inputs)
        torch.cuda.synchronize()
        batched_sync_us = ((time.perf_counter() - ts_batch) * 1e6) / 100.0
        latencies_us.sort()
        print(
            f"[CG_AUDIT_DEVONLY] drop={drop_broad} async={stream_async} bs={bs} "
            f"prep+fwd_host_med={latencies_us[50]:.2f}us mean={sum(latencies_us)/100:.2f}us "
            f"batched_sync={batched_sync_us:.2f}us events={dict(dev_events)}"
        )
        if drop_broad == "1":
            assert dev_events.get("cudaEventSynchronize", 0) == 0, dev_events
            assert (
                dev_events.get(
                    "cuda_graph.prepareAttentionInputs(wait_forward_event)", 0
                )
                == 0
            ), dev_events
            assert dev_events.get("aten::zero_", 0) == 0, dev_events
            assert dev_events.get("aten::fill_", 0) == 0, dev_events

    # Multi-step changing bs + padded batch + varying block tables bit-exact check
    for step_idx, bs in enumerate((96, 1, 64, 5, 8, 32, 3, 1, 96)):
        full_val = 2 + (step_idx % 5)
        aux_val = 3 + (step_idx % 4)
        seq_p1_val = 4 + (step_idx % 6)
        inputs = _build_decode_inputs(
            GROUP_TAGS,
            {"full": full_val, "aux": aux_val},
            batch_size=bs,
            block_count=KERNEL_BLOCK_TABLE_WIDTH,
        )
        inputs.attention_inputs["full"].sequence_lengths_plus_1_device.fill_(seq_p1_val)
        inputs.input_hiddens.copy_(
            torch.arange(bs * HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda").view(
                bs, HIDDEN_SIZE
            )
            * 0.5
        )
        assert dev_runner.prepare(inputs)
        out = dev_runner.forward(inputs).hidden_states
        expected = _expected_device_only_output(
            inputs, full_val, aux_val, seq_p1_val, bs
        )
        assert torch.equal(out, expected), f"step={step_idx} bs={bs} mismatch"
    torch.cuda.synchronize()


class CudaGraphDecodeReplayAuditTest(unittest.TestCase):
    def test_audit_modes(self) -> None:
        if os.environ.get("_CG_AUDIT_CHILD") == "1":
            _run_probe(os.environ["_CG_DROP"], os.environ["_CG_ASYNC"])
            return
        for drop_broad, stream_async in (("0", "0"), ("1", "0"), ("1", "1")):
            env = dict(os.environ)
            env["_CG_AUDIT_CHILD"] = "1"
            env["_CG_DROP"] = drop_broad
            env["_CG_ASYNC"] = stream_async
            res = subprocess.run(
                [sys.executable, __file__],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            print(res.stdout)
            if res.stderr:
                print(res.stderr)
            self.assertEqual(res.returncode, 0)


if __name__ == "__main__":
    unittest.main()
