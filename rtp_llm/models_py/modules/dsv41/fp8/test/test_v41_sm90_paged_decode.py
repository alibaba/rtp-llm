"""V4-format 584B cache -> Hopper paged decode, without prefill fallback.

The optional benchmark measures both complete Python call chains inside CUDA
graphs. It is a component benchmark, not end-to-end serving TPOT.
"""

import contextlib
import importlib
import json
import math
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch


def _pool(entries, width, count):
    # The physical stride deliberately cannot be used as the slot divisor.
    pages = math.ceil(count / entries) + 1
    stride = math.ceil(entries * width / 576) * 576 + 576
    storage = torch.zeros((pages, stride), device="cuda", dtype=torch.uint8)
    return storage.as_strided((pages, entries, width), (stride, width, 1))


def _inputs(batch, query_len, heads, dual, *, global_count=64, native_v41=False):
    from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
        quantize_and_insert_k_cache_fp4,
    )
    from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
        quantize_and_insert_swa_k_cache,
    )
    from rtp_llm.models_py.modules.dsv41.fp8._swa_kv_insert_triton import (
        quantize_and_insert_k_cache,
    )
    from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_v4_cache import (
        quantize_and_insert_k_cache as global_fp4_roundtrip_v4_writer,
    )

    swa_writer = quantize_and_insert_swa_k_cache if native_v41 else quantize_and_insert_k_cache
    global_writer = quantize_and_insert_k_cache_fp4 if native_v41 else global_fp4_roundtrip_v4_writer
    swa = _pool(128, 528 if native_v41 else 584, 128)
    swa_slots = torch.arange(128, 256, device="cuda")
    swa_writer(
        torch.randn(128, 512, device="cuda", dtype=torch.bfloat16) * 0.2,
        swa,
        swa_slots,
    )
    result = dict(
        q=torch.randn(batch, query_len, heads, 512, device="cuda", dtype=torch.bfloat16)
        * 0.1,
        swa_pool=swa,
        swa_slots=swa_slots.int().view(1, 1, -1).repeat(batch, query_len, 1),
        sink=torch.linspace(-1, 1, heads, device="cuda"),
        softmax_scale=512**-0.5,
        return_lse=True,
    )
    if dual:
        glob = _pool(64, 288 if native_v41 else 584, global_count)
        global_slots = torch.arange(64, 64 + global_count, device="cuda")
        global_writer(
            torch.randn(global_count, 512, device="cuda", dtype=torch.bfloat16) * 0.2
            + 0.125,
            glob,
            global_slots,
        )
        result.update(
            global_pool=glob,
            global_slots=global_slots.int().view(1, 1, -1).repeat(batch, query_len, 1),
        )
    return result


def _read_v4_rows(pool, slots):
    """Independent page-planar oracle; NOT a production dequantization call."""
    pages, entries, width = pool.shape
    assert width == 584 and pool.dtype == torch.uint8
    stride = pool.stride(0)
    raw = pool.as_strided((pages * stride,), (1,))
    page = slots.long() // entries
    offset = slots.long() % entries
    data_base = page * stride + offset * 576
    payload = raw[data_base[:, None] + torch.arange(448, device=pool.device)]
    payload = payload.contiguous().view(torch.float8_e4m3fn).float()
    rope_bytes = raw[data_base[:, None] + torch.arange(448, 576, device=pool.device)]
    rope = rope_bytes.contiguous().view(torch.bfloat16).float()
    scale_base = page * stride + entries * 576 + offset * 8
    scales = raw[scale_base[:, None] + torch.arange(7, device=pool.device)]
    nope = payload.reshape(-1, 7, 64) * torch.exp2(scales.float() - 127)[..., None]
    return torch.cat((nope.flatten(1), rope), dim=-1)


def _reference(inputs):
    from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_flash_mla_cache_format import (
        _read_rows as read_native_v41_rows,
    )

    q = inputs["q"]
    batch, query_len, heads, _ = q.shape
    out = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.full((batch, heads, query_len), float("inf"), device=q.device)
    for b in range(batch):
        for s in range(query_len):
            values = []
            for prefix in ("swa", "global"):
                pool = inputs.get(prefix + "_pool")
                if pool is None:
                    continue
                slots = inputs[prefix + "_slots"][b, s]
                lengths = inputs.get(prefix + "_lengths")
                if lengths is not None:
                    length = int(
                        lengths.reshape(batch, -1)[b, 0 if lengths.numel() == batch else s]
                    )
                    slots = slots[: max(0, min(length, slots.numel()))]
                slots = slots[slots >= 0]
                if slots.numel():
                    reader = _read_v4_rows if pool.shape[-1] == 584 else read_native_v41_rows
                    values.append(reader(pool, slots))
            if not values:
                continue
            kv = torch.cat(values)
            logits = (q[b, s].float() @ kv.T) * inputs["softmax_scale"]
            logz = torch.logsumexp(logits, -1)
            # The sink affects the denominator, but NOT the returned LSE.
            denominator = torch.logaddexp(logz, inputs["sink"].float())
            out[b, s] = torch.exp(logits - denominator[:, None]) @ kv
            lse[b, :, s] = logz
    return out, lse


@contextlib.contextmanager
def _no_sparse_prefill():
    # Guard both lazy imports and already imported aliases. Providers themselves
    # are never mocked with fake numerical output: forbidden calls fail loudly.
    importlib.import_module("flash_mla")
    importlib.import_module("rtp_llm.models_py.modules.dsv41.flash_mla")
    importlib.import_module("rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_attention")
    with contextlib.ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith(("flash_mla", "rtp_llm.models_py.modules.dsv41")):
                if module is not None and hasattr(module, "flash_mla_sparse_fwd"):
                    stack.enter_context(
                        mock.patch.object(
                            module,
                            "flash_mla_sparse_fwd",
                            side_effect=AssertionError("decode called sparse-prefill attention"),
                        )
                    )
        yield


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SM90PagedDecodeTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires an SM90 GPU")
        torch.manual_seed(419210)

    def decode(
        self, q, swa_pool, swa_slots, sink, softmax_scale, *, global_pool=None,
        global_slots=None, swa_lengths=None, global_lengths=None, return_lse=False,
        metadata=None,
    ):
        from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
            get_or_build_sched_meta,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.decode.fp8_sparse_attn_decode_op import (
            SparseAttnV4DecodeFp8Op,
        )

        if metadata is None:
            # A fresh scheduler for each eager step, just like real per-step
            # metadata. Reusing initialized metadata after changing lengths is
            # unsupported unless its planner is captured inside a CUDA graph.
            metadata = SimpleNamespace(sched_meta_cache={})
        batch, query_len, heads, _ = q.shape
        sched_meta = get_or_build_sched_meta(
            metadata, batch_size=batch, q_len=query_len, num_heads=heads,
            topk=swa_slots.shape[-1],
            extra_attn_type="global_kv_1" if global_pool is not None else None,
        )
        return SparseAttnV4DecodeFp8Op(heads, 512, softmax_scale).forward(
            q, swa_pool, sink, swa_slots, sched_meta,
            topk_length=swa_lengths, extra_k_cache=global_pool,
            extra_topk_idxs=global_slots, extra_topk_length=global_lengths,
            return_lse=return_lse,
        )

    def _check(self, inputs, actual):
        expected, expected_lse = _reference(inputs)
        out, lse = actual
        self.assertEqual(out.shape, expected.shape)
        self.assertEqual(out.dtype, inputs["q"].dtype)
        self.assertEqual(lse.shape, expected_lse.shape)
        self.assertEqual(lse.dtype, torch.float32)
        # Exactly the tolerances of test_v41_sm90_attention, not relaxed for
        # this candidate. Reference reads original packed producer bytes.
        torch.testing.assert_close(out.float(), expected, rtol=0.02, atol=0.003)
        torch.testing.assert_close(lse, expected_lse, rtol=0.003, atol=0.003)

    def test_shape_pool_and_length_matrix(self):
        with _no_sparse_prefill():
            for batch in (1, 2):
                for query_len in (1, 2):
                    for heads in (8, 32, 64, 128):
                        for dual in (False, True):
                            with self.subTest(B=batch, S=query_len, H=heads, dual=dual):
                                args = _inputs(batch, query_len, heads, dual)
                                args["swa_slots"][..., 1::11] = -1
                                # Native paged decode accepts [B] int32 lengths,
                                # shared across S query positions. Query-specific
                                # exclusions are represented by -1 slot ids.
                                args["swa_lengths"] = torch.arange(
                                    113, 113 + batch, device="cuda", dtype=torch.int32
                                )
                                if dual:
                                    args["global_slots"][..., 3::9] = -1
                                    args["global_lengths"] = torch.arange(
                                        batch, device="cuda", dtype=torch.int32
                                    ) + 47
                                actual = self.decode(**args)
                                self._check(args, actual)
                                if batch == query_len == 1 and heads == 64:
                                    tensor_only = self.decode(**dict(args, return_lse=False))
                                    torch.testing.assert_close(tensor_only, actual[0])

    def test_empty_rows_and_sink_semantics(self):
        args = _inputs(2, 2, 32, True)
        args["swa_lengths"] = torch.tensor([0, 17], device="cuda", dtype=torch.int32)
        args["global_lengths"] = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
        # The last row is empty by invalid slots rather than by length.
        args["swa_slots"][1, 1] = -1
        args["sink"][0] = float("inf")
        args["sink"][1] = -float("inf")
        with _no_sparse_prefill():
            self._check(args, self.decode(**args))

    def test_full_width_without_lengths(self):
        for dual in (False, True):
            with self.subTest(dual=dual), _no_sparse_prefill():
                args = _inputs(1, 1, 64, dual)
                self._check(args, self.decode(**args))

    def test_graph_replay_updates_all_device_inputs(self):
        from rtp_llm.models_py.modules.dsv41.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
        )
        from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_v4_cache import (
            quantize_and_insert_k_cache as global_writer,
        )

        for heads in (8, 64):
            with self.subTest(heads=heads), _no_sparse_prefill():
                args = _inputs(2, 2, heads, True, global_count=2048)
                args["swa_lengths"] = torch.tensor([0, 17], device="cuda", dtype=torch.int32)
                args["global_lengths"] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
                metadata = SimpleNamespace(sched_meta_cache={})
                run = lambda: self.decode(**args, metadata=metadata)
                self._check(args, run())
                warmup_schedule = next(iter(metadata.sched_meta_cache.values()))
                graph, captured = _capture(run)
                captured_schedule = next(iter(metadata.sched_meta_cache.values()))
                self.assertIsNot(captured_schedule, warmup_schedule)
                graph.replay()
                self._check(args, captured)

                args["q"].copy_(args["q"].flip(-2) * 1.3)
                args["sink"].copy_(args["sink"].flip(0) + 0.7)
                args["swa_slots"].copy_(args["swa_slots"].flip(-1))
                args["global_slots"].copy_(args["global_slots"].flip(-1))
                args["swa_slots"][..., 2::7] = -1
                args["global_slots"][..., 1::5] = -1
                args["swa_lengths"].copy_(torch.tensor([119, 0], device="cuda", dtype=torch.int32))
                args["global_lengths"].copy_(torch.tensor([2048, 0], device="cuda", dtype=torch.int32))
                graph.replay()
                self._check(args, captured)

                # Update cache bytes in place with actual producers, retaining
                # addresses captured by the decode graph.
                for prefix, writer, entries, count in (
                    ("swa", quantize_and_insert_k_cache, 128, 128),
                    ("global", global_writer, 64, 2048),
                ):
                    writer(
                        torch.randn(count, 512, device="cuda", dtype=torch.bfloat16) * 0.3 - 0.1,
                        args[prefix + "_pool"],
                        torch.arange(entries, entries + count, device="cuda"),
                    )
                graph.replay()
                self._check(args, captured)
                # A second capture must drop its eager warmup schedule too.
                # Do not replay the previous initialized scheduler eagerly
                # after changing lengths: start this step with empty cache.
                metadata.sched_meta_cache.clear()
                second_graph, second_captured = _capture(run)
                second_schedule = next(iter(metadata.sched_meta_cache.values()))
                self.assertIsNot(second_schedule, captured_schedule)
                args["global_lengths"].copy_(torch.tensor([0, 1024], device="cuda", dtype=torch.int32))
                second_graph.replay()
                self._check(args, second_captured)
                graph.replay()
                self._check(args, captured)

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_PAGED_DECODE_BENCH") == "1",
        "explicit component benchmark opt-in required",
    )
    def test_benchmark_correct_before_timing(self):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_attention import (
            selected_rows_attention,
        )

        def event_ms(graph, repetitions=100):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repetitions):
                graph.replay()
            stop.record()
            stop.synchronize()
            return start.elapsed_time(stop) / repetitions

        for count in (512, 1024, 2048):
            torch.manual_seed(419210 + count)
            args = _inputs(1, 1, 64, True, global_count=count)
            torch.manual_seed(419210 + count)
            old_args = _inputs(1, 1, 64, True, global_count=count, native_v41=True)
            metadata = SimpleNamespace(sched_meta_cache={})
            with _no_sparse_prefill():
                self._check(args, self.decode(**args))
                new_graph, new_out = _capture(lambda: self.decode(**args, metadata=metadata))
                new_graph.replay()
                self._check(args, new_out)
            self._check(old_args, selected_rows_attention(**old_args))
            old_graph, old_out = _capture(lambda: selected_rows_attention(**old_args))
            old_graph.replay()
            self._check(old_args, old_out)
            timing = {"selected_rows_ms": [], "paged_decode_ms": []}
            for round_id in range(5):
                order = (("selected_rows_ms", old_graph), ("paged_decode_ms", new_graph))
                if round_id % 2:
                    order = tuple(reversed(order))
                for name, graph in order:
                    timing[name].append(event_ms(graph))
            print(
                "SM90_PAGED_DECODE_BENCH "
                + json.dumps(dict(
                    batch=1, query_len=1, heads=64, swa=128, global_count=count,
                    scope="complete attention call CUDA graph replay; not serving TPOT",
                    precision="BF16 Q; old FP8 G32/FP4 G16 vs V4 G64+BF16 RoPE (GLOBAL retains FP4 roundtrip)",
                    comparison="same prequant inputs, different cache precision; independent FP32 reference per arm",
                    profile="not collected in clean timing test", **timing,
                )),
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
