"""Focused query-conversion compatibility gates for the SM100 GPU test target."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    round_to_e4m3_compute_grid_,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.decode import q8kv4_decode as decode
from rtp_llm.models_py.triton_kernels.sparse_msa.decode.nvfp4_q8_query_cast import (
    fused_query_cast,
)


def exact(a, b):
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class QueryCastTest(unittest.TestCase):
    def test_wrapper_conversion_contract(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for fused in (False, True):
                with self.subTest(dtype=dtype, fused=fused):
                    self.check_wrapper_conversion_contract(dtype, fused)

    def check_wrapper_conversion_contract(self, dtype, fused):
        # Stop after actual wrapper conversion, before scoring/TopK/attention.
        q = (
            torch.tensor(
                [0.0, -0.0, 1.0, 448.0, 464.0, 500.0, float("inf"), -float("inf")],
                device="cuda",
                dtype=dtype,
            )
            .repeat(1024)
            .reshape(1, 64, 128)
        )
        iq = q[:, :4].contiguous().clone()
        original = (q.clone(), iq.clone())
        layout = SimpleNamespace(
            page_size=128,
            head_dim=128,
            num_heads=4,
            logical_views=lambda _: SimpleNamespace(idx_k_fp4=None, idx_k_scale=None),
        )
        table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
        lens = torch.ones(1, device="cuda", dtype=torch.int32)
        options = dict(
            indexer_dim=128,
            block_size=128,
            topk=16,
            init_blocks=0,
            local_blocks=0,
            score_type="max",
        )
        if fused and dtype != torch.bfloat16:
            with self.assertRaisesRegex(ValueError, "requires BF16"):
                decode.q8kv4_paged_sparse_decode(
                    q, iq, layout, table, lens, fuse_bf16_query_rounding=True, **options
                )
            return

        class Converted(Exception):
            pass

        def stop_after_conversion(*args, **kwargs):
            raise Converted()

        with patch.object(decode, "q8kv4_index_score", stop_after_conversion):
            with self.assertRaises(Converted):
                # False case intentionally omits the argument to exercise its default.
                extra = {"fuse_bf16_query_rounding": True} if fused else {}
                decode.q8kv4_paged_sparse_decode(
                    q, iq, layout, table, lens, **options, **extra
                )
        workspace = decode._Q8KV4DecodeWorkspace.acquire(q, iq, 1, 16, 16)
        for inp, out in zip(original, (workspace.q8, workspace.idx_q8)):
            expected = inp.clone()
            if fused:
                round_to_e4m3_compute_grid_(expected)
            exact(out.clone(), expected.to(torch.float8_e4m3fn))
        exact(q, original[0])
        exact(iq, original[1])

    def test_exhaustive_bf16_and_changed_graph_replay(self):
        # All 65,536 BF16 bit patterns in both streams, including raw NaN payloads.
        q = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        q = q.reshape(8, 64, 128)
        iq = q.clone()
        q8, iq8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn) for x in (q, iq))

        def check():
            for inp, out in ((q, q8), (iq, iq8)):
                expected = inp.clone()
                round_to_e4m3_compute_grid_(expected)
                exact(out, expected.to(torch.float8_e4m3fn))

        fused_query_cast(q, iq, q8, iq8)
        check()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fused_query_cast(q, iq, q8, iq8)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fused_query_cast(q, iq, q8, iq8)
        pointers = (q8.data_ptr(), iq8.data_ptr())
        for value in (0.0, 500.0, -500.0):
            q.fill_(value)
            iq.fill_(-value)
            graph.replay()
            torch.cuda.synchronize()
            check()
            assert pointers == (q8.data_ptr(), iq8.data_ptr())

    def test_index_only_cast_preserves_main_buffer(self):
        q = torch.randn(20, 64, 128, device="cuda", dtype=torch.bfloat16)
        iq = torch.randn(20, 4, 128, device="cuda", dtype=torch.bfloat16)
        q8 = torch.full_like(q, 2.0, dtype=torch.float8_e4m3fn)
        iq8 = torch.empty_like(iq, dtype=torch.float8_e4m3fn)
        original_q8 = q8.clone()

        def run():
            fused_query_cast(q, iq, q8, iq8, cast_main_query=False)

        def check():
            expected = iq.clone()
            round_to_e4m3_compute_grid_(expected)
            exact(iq8, expected.to(torch.float8_e4m3fn))
            exact(q8, original_q8)

        run()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for value in (0.0, 500.0, -500.0):
            iq.fill_(value)
            graph.replay()
            check()

    def test_helper_rejects_incompatible_buffers(self):
        q = torch.ones(2, 64, 128, device="cuda", dtype=torch.bfloat16)
        iq = torch.ones(2, 4, 128, device="cuda", dtype=torch.bfloat16)
        q8, iq8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn) for x in (q, iq))
        for bad in (q.float(), q.transpose(1, 2), q.cpu()):
            with self.assertRaises(ValueError):
                fused_query_cast(bad, iq, q8, iq8)
        with self.assertRaises(ValueError):
            fused_query_cast(q, iq[:1], q8, iq8[:1])
        with self.assertRaises(ValueError):
            fused_query_cast(q, iq, q8.transpose(1, 2), iq8)


if __name__ == "__main__":
    unittest.main()
