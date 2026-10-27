"""Bit-exact CP KV pack regression against the unfused production CUDA op.

Run the Bazel target on one idle CUDA GPU. No model/checkpoint is required.
Padding table entries are deliberately invalid: padding must never read them.
"""

import os
import random
import statistics
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_sparse_cp_impl as cp,
)


def starts(lengths):
    return [sum(lengths[:i]) for i in range(len(lengths))]


def case(actual, padded, page=64, strided=False, all_fp8_bits=False):
    assert len(actual) == len(padded) and all(a <= p for a, p in zip(actual, padded))
    device = "cuda"
    columns = max([((a + page - 1) // page) for a in actual] + [0])
    blocks = max(1, sum((a + page - 1) // page for a in actual))
    backing = torch.empty(
        (blocks * (2 if strided else 1), page, 704 if strided else 656),
        dtype=torch.uint8,
        device=device,
    )
    cache = backing[::2, :, :656] if strided else backing
    codes = torch.randint(
        0, 127, cache[..., :512].shape, dtype=torch.uint8, device=device
    )
    codes |= torch.randint(0, 2, codes.shape, dtype=torch.uint8, device=device) * 128
    if all_fp8_bits:
        codes.copy_(torch.arange(512, device=device).to(torch.uint8).expand_as(codes))
    cache[..., :512].copy_(codes)
    scales = torch.rand((blocks, page, 4), device=device) * 2 + 0.001
    scales[0, 0] = torch.tensor([0, 2**-20, 1.333, 2**12], device=device)
    cache[..., 512:528].view(torch.float32).copy_(scales)
    rope = torch.randn((blocks, page, 64), device=device).to(torch.bfloat16)
    cache[..., 528:].view(torch.bfloat16).copy_(rope)
    table_storage = torch.full(
        (len(actual), columns + 3), -2147483647, dtype=torch.int32, device=device
    )
    table = table_storage[:, :columns]
    ids = torch.randperm(blocks, device=device).int()
    pos = 0
    for b, a in enumerate(actual):
        n = (a + page - 1) // page
        table[b, :n] = ids[pos : pos + n]
        pos += n
    a = torch.tensor(actual, dtype=torch.int32, device=device)
    p = torch.tensor(padded, dtype=torch.int32, device=device)
    s = torch.tensor(starts(padded), dtype=torch.int32, device=device)
    return cache, table, a, p, s


def legacy(cache, table, a, p, s, actual, padded):
    packed = torch.empty((sum(actual), 576), dtype=torch.bfloat16, device=cache.device)
    if sum(actual):
        cp.rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
            cache,
            packed,
            table,
            a,
            torch.tensor(starts(actual), dtype=torch.int32, device=cache.device),
            len(actual),
            sum(actual),
        )
    out = torch.zeros((sum(padded), 576), dtype=torch.bfloat16, device=cache.device)
    for a0, p0, n in zip(starts(actual), starts(padded), actual):
        out[p0 : p0 + n].copy_(packed[a0 : a0 + n])
    return out


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PaddedPackTest(unittest.TestCase):
    def check_case(self, actual, padded, page=64, strided=False, all_fp8_bits=False):
        tensors = case(actual, padded, page, strided, all_fp8_bits)
        cache, table, a, p, s = tensors
        # Guard rows and strided row tails must remain untouched.
        storage = torch.full(
            (sum(padded) + 2, 608), 37, dtype=torch.bfloat16, device="cuda"
        )
        out = storage[1:-1, 16:592]
        expected = legacy(*tensors, actual, padded)
        for poison in (float("nan"), -73.0):
            out.fill_(poison)
            cp.rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_padded(
                cache, out, table, a, p, s
            )
            # Compare raw BF16 bits, including signed zeros and NaN payloads.
            self.assertTrue(
                torch.equal(out.view(torch.int16), expected.view(torch.int16))
            )
            self.assertTrue(bool((storage[[0, -1]] == 37).all()))
            self.assertTrue(bool((storage[:, :16] == 37).all()))
            self.assertTrue(bool((storage[:, 592:] == 37).all()))
        return tensors, out, expected

    def test_padding_empty_requests_and_page_boundaries(self):
        cases = [
            ([], []),
            ([0], [0]),
            ([0], [128]),
            ([0, 0, 0], [64, 0, 128]),
            ([1], [64]),
            ([63, 64, 65], [64, 64, 128]),
            ([0, 1, 0, 65, 0], [0, 64, 64, 128, 0]),
            ([3, 0, 2], [4, 4, 4]),
            ([512, 513, 1], [512, 1024, 512]),
            ([128, 0, 64], [128, 0, 64]),
            ([0, 0], [0, 0]),
        ]
        for page in (64, 512):
            for actual, padded in cases:
                with self.subTest(page=page, actual=actual, padded=padded):
                    self.check_case(actual, padded, page, strided=True)

    def test_random_batches(self):
        rng = random.Random(10727)
        for _ in range(40):
            actual = [
                rng.choice([0, 1, 63, 64, 65, 511, 512, 513])
                for _ in range(rng.randint(1, 16))
            ]
            padded = [((a + 511) // 512 + rng.randint(0, 1)) * 512 for a in actual]
            self.check_case(actual, padded)

    def test_every_fp8_bit_pattern_exact(self):
        self.check_case([65, 0, 1], [128, 64, 64], all_fp8_bits=True)

    def test_nondefault_stream_and_attention_consumer(self):
        tensors = case([3, 0, 65], [64, 64, 128])
        cache, table, a, p, s = tensors
        out = torch.empty((256, 576), device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            cp.rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_padded(
                cache, out, table, a, p, s
            )
            ready = torch.cuda.Event()
            ready.record()
        torch.cuda.current_stream().wait_event(ready)
        expected = legacy(*tensors, [3, 0, 65], [64, 64, 128])
        q = torch.randn((4, 576), device="cuda", dtype=torch.float32) * 0.01

        def attend(kv):
            scores = torch.softmax(q @ kv.float().T, dim=-1)
            return scores @ kv[:, :512].float()

        self.assertTrue(torch.equal(out.view(torch.int16), expected.view(torch.int16)))
        self.assertTrue(torch.equal(attend(out), attend(expected)))

    def test_production_pack_calls_fused_op_and_preserves_graph_fallback(self):
        tensors = case([3, 0, 65], [64, 64, 128])
        cache, table, a, p, s = tensors
        op = object.__new__(cp.SparseMlaFp8CPOp)
        op._gather = SimpleNamespace(batch_size=3)
        op.block_table = table
        op.sharded_actual_local_kv_lens = a
        op.sharded_local_kv_lens = p
        op.sharded_workspace_starts = s
        op.sharded_actual_workspace_starts = torch.tensor(
            [0, 3, 3], device="cuda", dtype=torch.int32
        )
        op.sharded_total_local_kv_len = 256
        op.sharded_actual_total_local_kv_len = 68
        op.sharded_kv_restore_indices = torch.arange(256, device="cuda")
        op.indexer_copy_dst_idx = None
        out = torch.empty((256, 576), device="cuda", dtype=torch.bfloat16)
        cache_obj = SimpleNamespace(kv_cache_base=cache)
        op.use_cuda_graph = True
        reference = op._pack_sharded_kv_cache(cache_obj, out)
        op.use_cuda_graph = False
        with patch.object(
            cp,
            "_scatter_actual_to_padded",
            side_effect=AssertionError("unfused scatter"),
        ):
            got = op._pack_sharded_kv_cache(cache_obj, out, local_fused=out)
        self.assertEqual(got.data_ptr(), out.data_ptr())
        self.assertTrue(torch.equal(got.view(torch.int16), reference.view(torch.int16)))

    def test_same_layer_raw_history_omits_destination_and_matches_restored(self):
        # Real padded CUDA packing and stream events; substitute only transport
        # and cache write, whose values are not changed by workspace elision.
        for actual, padded, new_rows in (
            ([3, 0, 65], [64, 64, 128], 2),
            ([3, 0, 65], [64, 64, 128], 0),
            ([0], [0], 0),
        ):
            with self.subTest(actual=actual, new_rows=new_rows):
                cache, table, a, p, s = case(actual, padded)
                op = object.__new__(cp.SparseMlaFp8CPOp)
                op._gather = SimpleNamespace(batch_size=len(actual))
                op.block_table = table
                op.use_cuda_graph = False
                op.sharded_actual_local_kv_lens = a
                op.sharded_local_kv_lens = p
                op.sharded_workspace_starts = s
                op.sharded_actual_workspace_starts = torch.tensor(
                    starts(actual), device="cuda", dtype=torch.int32
                )
                op.sharded_total_local_kv_len = sum(padded)
                op.sharded_actual_total_local_kv_len = sum(actual)
                op.indexer_copy_dst_idx = None
                logical_rows = [
                    start + i
                    for start, n in zip(starts(padded), actual)
                    for i in range(n)
                ]
                mapping = torch.tensor(
                    [
                        rank * sum(padded) + row
                        for rank in range(4)
                        for row in logical_rows
                    ],
                    device="cuda",
                    dtype=torch.int64,
                )
                op.sharded_kv_restore_indices = mapping
                op._rank_major_write_slots = Mock(
                    return_value=torch.arange(new_rows * 4, device="cuda")
                )
                op.write_kv = Mock()
                op._allocate_fused_kv = Mock(
                    side_effect=lambda: torch.empty(
                        (sum(actual) * 4, 576), device="cuda", dtype=torch.bfloat16
                    )
                )
                wrapper = object.__new__(cp.SparseMlaCpImpl)
                wrapper.fmha_impl = op
                wrapper._cp_size = 4
                wrapper._kv_overlap_stream = torch.cuda.Stream()
                ckv = torch.ones((new_rows, 512), device="cuda", dtype=torch.bfloat16)
                kpe = torch.ones((new_rows, 64), device="cuda", dtype=torch.bfloat16)
                outputs = []
                for enabled in (0, 1):
                    op._allocate_fused_kv.reset_mock()
                    with patch.dict(
                        os.environ, {"RTP_CP_REMAP_HISTORY_TOPK": str(enabled)}
                    ), patch.object(
                        cp,
                        "all_gather_kv_overlap",
                        side_effect=lambda src, _group, *, output: torch.cat(
                            [src] * 4, out=output
                        ),
                    ):
                        with wrapper._prepare_same_layer_kv_async(
                            ckv, kpe, SimpleNamespace(kv_cache_base=cache)
                        ) as (prepared, ready):
                            torch.cuda.current_stream().wait_event(ready)
                            value = (
                                prepared.values.index_select(
                                    0, prepared.restore_indices
                                )
                                if enabled
                                else prepared
                            )
                            outputs.append(value.clone())
                        if enabled:
                            op._allocate_fused_kv.assert_not_called()
                        else:
                            op._allocate_fused_kv.assert_called_once()
                self.assertTrue(
                    torch.equal(
                        outputs[0].view(torch.int16), outputs[1].view(torch.int16)
                    )
                )

    def test_graph_replay_primitive_uses_updated_lengths(self):
        tensors = case([65, 3], [128, 64])
        cache, table, a, p, s = tensors
        out = torch.empty((192, 576), device="cuda", dtype=torch.bfloat16)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cp.rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_padded(
                cache, out, table, a, p, s
            )
        for actual in ([65, 3], [1, 0], [0, 0], [64, 2]):
            a.copy_(torch.tensor(actual, device="cuda", dtype=torch.int32))
            out.fill_(float("nan"))
            g.replay()
            expected = legacy(*tensors, actual, [128, 64])
            self.assertTrue(
                torch.equal(out.view(torch.int16), expected.view(torch.int16))
            )

    def test_invalid_metadata_rejected_without_launch(self):
        cache, table, a, p, s = case([1], [64])
        out = torch.empty((64, 576), device="cuda", dtype=torch.bfloat16)
        for args in [
            (cache, out.float(), table, a, p, s),
            (cache, out, table.long(), a, p, s),
            (cache, out, table, a.cpu(), p, s),
            (cache, out, table, a, p[:0], s),
            (cache, out[:, ::2], table, a, p, s),
        ]:
            with self.assertRaises(RuntimeError):
                cp.rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_padded(*args)

    @unittest.skipUnless(
        os.environ.get("CP_KV_PACK_BENCHMARK") == "1", "opt-in benchmark"
    )
    def test_pack_benchmark(self):
        # Same per-rank padded row count as B4x64K CP4, plus tail/empty stress.
        for actual, padded in [
            ([16384] * 4, [16896] * 4),
            ([3, 0, 65535], [512, 512, 65536]),
        ]:
            cache, table, a, p, s = case(actual, padded)
            tmp = torch.empty((sum(actual), 576), device="cuda", dtype=torch.bfloat16)
            out = torch.empty((sum(padded), 576), device="cuda", dtype=torch.bfloat16)
            ast = torch.tensor(starts(actual), device="cuda", dtype=torch.int32)
            ids = torch.cat(
                [
                    torch.arange(p0, p0 + n, device="cuda")
                    for p0, n in zip(starts(padded), actual)
                ]
            )

            def old():
                out.zero_()
                cp.rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
                    cache, tmp, table, a, ast, len(actual), sum(actual)
                )
                out.index_copy_(0, ids, tmp)

            def new():
                cp.rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_padded(
                    cache, out, table, a, p, s
                )

            timings = {}
            for name, fn in [("legacy", old), ("fused", new)]:
                for _ in range(20):
                    fn()
                torch.cuda.synchronize()
                samples = []
                for _ in range(9):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    begin.record()
                    for _ in range(100):
                        fn()
                    end.record()
                    end.synchronize()
                    samples.append(begin.elapsed_time(end) * 10)
                timings[name] = statistics.median(samples)
            print(
                "CP_KV_PACK_BENCHMARK",
                actual,
                padded,
                timings,
                "microseconds",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
