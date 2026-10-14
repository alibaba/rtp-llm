import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
)

WARMUP_ENV = "RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"
HIDDEN_SIZE = 4
PREPARED_VALUE = 7


class StreamWarmupModel:
    """A backend with stream/shape-local initialization, like MegaMHC."""

    def __init__(self, fail_warmup: bool = False) -> None:
        self.fail_warmup = fail_warmup
        self.forward_calls = 0
        self.phases = []
        self.warmed = set()
        self.warmup_snapshots = []
        self.failure_done = None
        self.failure_snapshot = None

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph=False):
        prepared = torch.zeros(1, device="cuda", dtype=torch.bfloat16)
        # Model an asynchronous metadata producer on the caller's stream.
        # Initialize the sentinel first, then delay its final value so a warmup
        # moved to another stream without a dependency reads the wrong value.
        torch.cuda.current_stream().synchronize()
        torch.cuda._sleep(100_000_000)
        prepared.fill_(PREPARED_VALUE)
        return SimpleNamespace(prepared=prepared)

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        self.forward_calls += 1
        if self.fail_warmup and self.forward_calls == 2:
            # The first forward is the datatype probe, before per-key warmup.
            # Submit a delayed read of caller-owned inputs before failing. The
            # runner must drain this work before freeing its input buffers.
            self.failure_snapshot = torch.empty_like(inputs.input_hiddens)
            self.failure_done = torch.cuda.Event()
            torch.cuda._sleep(200_000_000)
            self.failure_snapshot.copy_(inputs.input_hiddens)
            self.failure_done.record()
            raise RuntimeError("injected stream warmup failure")

        stream = torch.cuda.current_stream().cuda_stream
        capturing = torch.cuda.is_current_stream_capturing()
        shape = tuple(inputs.input_hiddens.shape)
        key = (stream, shape)
        flag = os.environ.get(WARMUP_ENV)
        self.phases.append((stream, shape, capturing, flag))
        if not capturing:
            if flag == "1":
                self.warmed.add(key)
            self.warmup_snapshots.append(fmha_impl.prepared.clone())
        # Deliberately produce a distinguishable fallback if capture missed
        # stream-local warmup. Do not raise inside capture and poison CUDA state.
        increment = 1 if key in self.warmed else -100
        return PyModelOutputs(inputs.input_hiddens + fmha_impl.prepared + increment)


def make_inputs(batch_size: int, query_len: int, target_verify: bool):
    token_count = batch_size * query_len
    inputs = PyModelInputs()
    inputs.input_ids = torch.arange(token_count, dtype=torch.int32, device="cuda")
    inputs.input_hiddens = torch.arange(
        token_count * HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    ).view(token_count, HIDDEN_SIZE)
    attn = PyAttentionInputs()
    attn.is_prefill = target_verify
    attn.is_target_verify = target_verify
    attn.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
    attn.total_tokens = token_count
    attn.input_lengths = torch.full(
        (batch_size,), query_len, dtype=torch.int32
    ).pin_memory()
    attn.prefix_lengths = torch.full(
        (batch_size if target_verify else 0,), 1, dtype=torch.int32
    ).pin_memory()
    attn.sequence_lengths = torch.ones(
        0 if target_verify else batch_size, dtype=torch.int32
    ).pin_memory()
    attn.sequence_lengths_plus_1_device = torch.full(
        (batch_size,), 2, dtype=torch.int32, device="cuda"
    )
    attn.cu_seqlens = torch.arange(
        0, token_count + 1, query_len, dtype=torch.int32
    ).pin_memory()
    attn.cu_seqlens_device = attn.cu_seqlens.cuda()
    attn.cu_kv_seqlens_device = torch.arange(
        0,
        batch_size * (query_len + 1) + 1,
        query_len + 1,
        dtype=torch.int32,
        device="cuda",
    )
    attn.decode_cu_seqlens = torch.arange(
        batch_size + 1, dtype=torch.int32
    ).pin_memory()
    attn.decode_cu_seqlens_device = attn.decode_cu_seqlens.cuda()
    attn.context_total_kv_length = batch_size * (query_len + 1)
    attn.padding_offset = torch.zeros(token_count, dtype=torch.int32, device="cuda")
    attn.kv_cache_kernel_block_id = torch.zeros(
        (batch_size, 1), dtype=torch.int32
    ).pin_memory()
    attn.kv_cache_kernel_block_id_device = attn.kv_cache_kernel_block_id.cuda()
    attn.kv_cache_block_id = attn.kv_cache_kernel_block_id
    attn.kv_cache_block_id_device = attn.kv_cache_kernel_block_id_device
    inputs.attention_inputs = attn
    return inputs


class TestCudaGraphStreamWarmup(unittest.TestCase):
    def check_capture_and_replay(self, target_verify: bool) -> None:
        query_len = 3 if target_verify else 1
        # A nondefault caller stream also checks restoration without relying on
        # legacy default-stream implicit synchronization.
        caller = torch.cuda.Stream()
        caller.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(caller), patch.dict(
            os.environ, {WARMUP_ENV: "previous-value"}
        ):
            model = StreamWarmupModel()
            runner = CudaGraphRunner()
            runner.init_decode(
                model, HIDDEN_SIZE, 64, 1, [1, 2], [], target_verify, query_len
            )
            self.assertEqual(torch.cuda.current_stream(), caller)
            self.assertEqual(os.environ[WARMUP_ENV], "previous-value")
            captures = [phase for phase in model.phases if phase[2]]
            self.assertEqual(len(captures), 2)
            for stream, shape, _, flag in captures:
                self.assertNotEqual(stream, caller.cuda_stream)
                self.assertEqual(flag, "previous-value")
                matching_warmups = [
                    phase
                    for phase in model.phases
                    if phase[0] == stream and phase[1] == shape and not phase[2]
                ]
                self.assertEqual(len(matching_warmups), 2)
                self.assertTrue(all(phase[3] == "1" for phase in matching_warmups))
            for snapshot in model.warmup_snapshots:
                torch.testing.assert_close(
                    snapshot, torch.full_like(snapshot, PREPARED_VALUE)
                )

            calls_after_capture = model.forward_calls
            for batch_size in (2, 1, 2):
                inputs = make_inputs(batch_size, query_len, target_verify)
                self.assertTrue(runner.canRun(inputs))
                output = runner.forward(inputs)
                torch.testing.assert_close(
                    output.hidden_states,
                    inputs.input_hiddens + PREPARED_VALUE + 1,
                )
                self.assertEqual(model.forward_calls, calls_after_capture)
                self.assertEqual(torch.cuda.current_stream(), caller)
            caller.synchronize()

    def test_decode_and_draft_propose_warm_the_capture_stream(self) -> None:
        # Target decode and MTP draft propose share captureDecode().
        self.check_capture_and_replay(target_verify=False)

    def test_target_verify_warms_the_capture_stream(self) -> None:
        self.check_capture_and_replay(target_verify=True)

    def test_warmup_failure_restores_stream_and_environment(self) -> None:
        caller = torch.cuda.Stream()
        caller.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(caller), patch.dict(
            os.environ, {WARMUP_ENV: "previous-value"}
        ):
            runner = CudaGraphRunner()
            model = StreamWarmupModel(True)
            try:
                with self.assertRaisesRegex(
                    RuntimeError, "injected stream warmup failure"
                ):
                    runner.init_decode(model, HIDDEN_SIZE, 64, 1, [1])
                # No test-side synchronization may precede this assertion.
                self.assertIsNotNone(model.failure_done)
                self.assertTrue(model.failure_done.query())
                self.assertEqual(torch.cuda.current_stream(), caller)
                self.assertEqual(os.environ[WARMUP_ENV], "previous-value")
                self.assertFalse(runner.captureSessionMayBeDirty())
                torch.testing.assert_close(
                    model.failure_snapshot, torch.zeros_like(model.failure_snapshot)
                )
            finally:
                # Keep a failed regression run from leaking asynchronous work
                # into the next test even when testing the old native runner.
                torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
