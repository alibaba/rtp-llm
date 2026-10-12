"""Bitwise state/output parity and owned output storage for cached prefills."""

import ctypes
import threading
import unittest
from unittest import mock

import torch
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h_cublas,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.reuse_state_graph import (
    _GRAPHS,
    chunk_gated_delta_rule_fwd_h_reuse_graph,
)


class ReuseStateGraphTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "GPU validation must not skip")
        torch.manual_seed(530710)

    def inputs(self, length, heads=16, sequences=1):
        shape = (1, length, heads, 128)
        values = dict(
            k=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
            w=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
            u=torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
            gk=-torch.rand(shape, device="cuda", dtype=torch.float32) * 4,
            initial_state=torch.randn(sequences, heads, 128, 128, device="cuda") * 0.1,
            output_final_state=True,
            intermediate_state_dtype=torch.float32,
            chunk_size=64,
            use_exp2=True,
        )
        return values

    def compare(self, args):
        expected = chunk_gated_delta_rule_fwd_h_cublas(**args)
        actual = chunk_gated_delta_rule_fwd_h_reuse_graph(**args)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        return actual

    def test_partial_chunks_nonzero_states_and_replays(self):
        for heads in (1, 16):
            for length in (1, 31, 63, 64, 65, 75, 127, 128, 129, 257, 512):
                with self.subTest(heads=heads, length=length):
                    args = self.inputs(length, heads)
                    args["cu_seqlens"] = torch.tensor(
                        [0, length], device="cuda", dtype=torch.int32
                    )
                    old = self.compare(args)
                    stream = torch.cuda.current_stream(args["k"].device)
                    key = (
                        args["k"].device.index,
                        stream.cuda_stream,
                        args["k"].shape,
                        args["gk"].dtype,
                        (length,),
                    )
                    self.assertIn(key, _GRAPHS, "supported inputs must replay a graph")
                    frozen = tuple(t.clone() for t in old)
                    for name in ("k", "w", "u", "gk", "initial_state"):
                        args[name].mul_(0.75)
                    self.compare(args)
                    for a, b in zip(old, frozen):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertLessEqual(len(_GRAPHS), 8)

    def test_ragged_batch_replay_preserves_math_and_outputs(self):
        for lengths in (
            (1, 75, 73),
            (31, 63, 64, 65, 129, 257, 512),
            (75,) * 16,
            (411,) * 32,
        ):
            with self.subTest(lengths=lengths):
                args = self.inputs(sum(lengths), sequences=len(lengths))
                args["cu_seqlens"] = torch.tensor(
                    [0, *torch.tensor(lengths).cumsum(0).tolist()],
                    device="cuda",
                    dtype=torch.int32,
                )
                old = self.compare(args)
                stream = torch.cuda.current_stream(args["k"].device)
                key = (
                    args["k"].device.index,
                    stream.cuda_stream,
                    args["k"].shape,
                    args["gk"].dtype,
                    lengths,
                )
                self.assertIn(key, _GRAPHS)
                frozen = tuple(t.clone() for t in old)
                for name in ("k", "w", "u", "gk", "initial_state"):
                    args[name].mul_(0.75)
                self.compare(args)
                for a, b in zip(old, frozen):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertLessEqual(len(_GRAPHS), 8)

    def test_capture_allows_unrelated_cache_copy_thread(self):
        from rtp_llm.models_py.triton_kernels.kimi_kda import (
            reuse_state_graph as module,
        )

        args = self.inputs(179, sequences=3)
        args["cu_seqlens"] = torch.tensor(
            [0, 59, 118, 179], device="cuda", dtype=torch.int32
        )
        expected = chunk_gated_delta_rule_fwd_h_cublas(**args)
        copy_stream = torch.cuda.Stream()
        host = torch.ones(4096, pin_memory=True)
        device = torch.empty(4096, device="cuda")
        begin, done = threading.Event(), threading.Event()
        errors = []

        def cache_copy():
            try:
                torch.cuda.set_device(args["k"].device)
                self.assertTrue(begin.wait(10))
                # Cache-copy scratch buffers can require a real cudaMalloc,
                # which global capture forbids on unrelated host threads.
                runtime = ctypes.CDLL("libcudart.so")
                runtime.cudaMalloc.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_size_t,
                ]
                runtime.cudaMalloc.restype = ctypes.c_int
                runtime.cudaFree.argtypes = [ctypes.c_void_p]
                runtime.cudaFree.restype = ctypes.c_int
                scratch = ctypes.c_void_p()
                self.assertEqual(runtime.cudaMalloc(ctypes.byref(scratch), 4096), 0)
                try:
                    with torch.cuda.stream(copy_stream):
                        device.copy_(host, non_blocking=True)
                        copy_stream.synchronize()
                finally:
                    self.assertEqual(runtime.cudaFree(scratch), 0)
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        worker = threading.Thread(target=cache_copy)
        worker.start()
        reference = module.chunk_gated_delta_rule_fwd_h_cublas

        def during_capture(**values):
            if torch.cuda.is_current_stream_capturing():
                begin.set()
                self.assertTrue(done.wait(10), "cache copy must finish during capture")
                self.assertFalse(errors, errors)
            return reference(**values)

        try:
            with mock.patch.object(
                module, "chunk_gated_delta_rule_fwd_h_cublas", during_capture
            ):
                actual = module.chunk_gated_delta_rule_fwd_h_reuse_graph(**args)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        finally:
            begin.set()
            worker.join(10)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors, errors)
        torch.testing.assert_close(device, host.cuda(), rtol=0, atol=0)

    def test_uniform_dynamic_batches_share_capacity_graph(self):
        retained = []
        key = None
        for sequences in (17, 31, 32, 17):
            args = self.inputs(75 * sequences, sequences=sequences)
            args["cu_seqlens"] = (
                torch.arange(sequences + 1, device="cuda", dtype=torch.int32) * 75
            )
            actual = self.compare(args)
            retained.append((actual, tuple(t.clone() for t in actual)))
            stream = torch.cuda.current_stream(args["k"].device)
            new_key = (
                args["k"].device.index,
                stream.cuda_stream,
                torch.Size((1, 75 * 32, 16, 128)),
                args["gk"].dtype,
                (75,) * 32,
            )
            self.assertIn(new_key, _GRAPHS)
            entry = _GRAPHS[new_key]
            if key is not None:
                self.assertIs(entry, key)
            key = entry
        for actual, snapshot in retained:
            for a, b in zip(actual, snapshot):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_same_total_tokens_different_ragged_layouts(self):
        args = self.inputs(149, sequences=3)
        stream = torch.cuda.current_stream(args["k"].device)
        keys = []
        # Reuse the metadata tensor in-place, as the executor does between steps.
        args["cu_seqlens"] = torch.empty(4, device="cuda", dtype=torch.int32)
        for boundaries in ((0, 1, 76, 149), (0, 64, 128, 149), (0, 1, 76, 149)):
            args["cu_seqlens"].copy_(torch.tensor(boundaries, device="cuda"))
            self.compare(args)
            lengths = tuple(b - a for a, b in zip(boundaries, boundaries[1:]))
            keys.append(
                (
                    args["k"].device.index,
                    stream.cuda_stream,
                    args["k"].shape,
                    args["gk"].dtype,
                    lengths,
                )
            )
        self.assertNotEqual(keys[0], keys[1])
        self.assertIn(keys[0], _GRAPHS)
        self.assertIn(keys[1], _GRAPHS)

    def test_long_sequence_and_explicit_opt_out_use_reference(self):
        for lengths in ((513, 75), (75, 75)):
            args = self.inputs(sum(lengths), sequences=len(lengths))
            args["cu_seqlens"] = torch.tensor(
                [0, lengths[0], sum(lengths)],
                device="cuda",
                dtype=torch.int32,
            )
            with mock.patch.dict("os.environ", {"GLM53_KDA_BATCH_REUSE_GRAPH": "0"}):
                before = set(_GRAPHS)
                self.compare(args)
                self.assertEqual(set(_GRAPHS), before)
        args = self.inputs(588, sequences=2)
        args["cu_seqlens"] = torch.tensor(
            [0, 513, 588], device="cuda", dtype=torch.int32
        )
        before = set(_GRAPHS)
        self.compare(args)
        self.assertEqual(set(_GRAPHS), before)

    def test_stream_workspace_isolation(self):
        args = self.inputs(75)
        first = self.compare(args)
        snapshots = tuple(t.clone() for t in first)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.compare(self.inputs(75))
        torch.cuda.current_stream().wait_stream(stream)
        for a, b in zip(first, snapshots):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
