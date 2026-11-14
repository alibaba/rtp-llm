"""FP8 QSA cache layout, independent attention reference and graph replay."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import main_kv_scale_view
from rtp_llm.models_py.kernels.cuda.qsa.test.fp8_kv_fixture import Fp8MainKvTestFixture
from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import SparseGqaFmhaImpl


class Fp8MainKvTest(Fp8MainKvTestFixture, unittest.TestCase):

    def test_native_prefill_and_decode_writers_with_graph_replay(self):
        from rtp_llm.ops.fused_rope_kvcache_op import _get_fused_rope_kvcache

        op = _get_fused_rope_kvcache()
        q_heads, kv_heads, dim, page_size = 6, 2, 256, 128
        width = (q_heads + 2 * kv_heads) * dim
        cache = torch.zeros(5, 2, kv_heads, page_size, dim, device="cuda").to(
            torch.float8_e4m3fn
        )
        scales = torch.full((5, 2, kv_heads, page_size), float("nan"), device="cuda")
        table = torch.tensor([[2], [4]], device="cuda", dtype=torch.int32)
        offsets = op.convert_offset_to_block_array(table)
        qkv = torch.randn(5, width, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
        padding = torch.zeros(5, device="cuda", dtype=torch.int32)
        op.prefill_fused_rope_kvcache(
            qkv,
            cu,
            2,
            3,
            q_heads,
            kv_heads,
            dim,
            tokens_per_block=page_size,
            store_qkv=True,
            store_cache=True,
            kv_cache=cache,
            kv_cache_scale=scales,
            kv_cache_offset=offsets,
            padding_offset=padding,
            rope_style=0,
            rope_dim=0,
        )
        for batch, start, length, page in ((0, 0, 3, 2), (1, 3, 2, 4)):
            for which in (0, 1):
                expected = (
                    qkv[
                        start : start + length,
                        (q_heads + which * kv_heads)
                        * dim : (q_heads + (which + 1) * kv_heads)
                        * dim,
                    ]
                    .reshape(length, kv_heads, dim)
                    .transpose(0, 1)
                )
                torch.testing.assert_close(
                    cache[page, which, :, :length].float(),
                    expected.to(cache.dtype).float(),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    scales[page, which, :, :length],
                    torch.ones_like(scales[page, which, :, :length]),
                    atol=0,
                    rtol=0,
                )
        decode_qkv = torch.randn(2, width, device="cuda", dtype=torch.bfloat16)
        lengths = torch.tensor([3, 2], device="cuda", dtype=torch.int32)

        def decode():
            return op.decode_fused_rope_kvcache(
                decode_qkv,
                None,
                lengths,
                2,
                q_heads,
                kv_heads,
                dim,
                cache,
                offsets,
                tokens_per_block=page_size,
                kv_cache_scale=scales,
                rope_style=0,
                rope_dim=0,
            )

        decode()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = decode()
        for _ in range(2):
            decode_qkv.mul_(0.75)
            graph.replay()
            torch.testing.assert_close(
                output.float().reshape(2, -1),
                decode_qkv[:, : q_heads * dim].float(),
                atol=0,
                rtol=0,
            )
            for row, page in ((0, 2), (1, 4)):
                for which in (0, 1):
                    expected = decode_qkv[
                        row,
                        (q_heads + which * kv_heads)
                        * dim : (q_heads + (which + 1) * kv_heads)
                        * dim,
                    ].reshape(kv_heads, dim)
                    torch.testing.assert_close(
                        cache[page, which, :, int(lengths[row])].float(),
                        expected.to(cache.dtype).float(),
                        atol=0,
                        rtol=0,
                    )
                    torch.testing.assert_close(
                        scales[page, which, :, int(lengths[row])],
                        torch.ones(kv_heads, device="cuda"),
                        atol=0,
                        rtol=0,
                    )

    def test_invalid_scale_rejected_before_any_cache_write(self):
        _, quant, scale, _, _, _ = self._fixture()
        impl = SparseGqaFmhaImpl.__new__(SparseGqaFmhaImpl)
        impl.kv_head_num = 2
        impl.attn_configs = SimpleNamespace(kernel_tokens_per_block=128)
        for invalid in (
            None,
            scale.half(),
            scale.cpu(),
            scale[:, :, :, :-1],
            scale.transpose(1, 2),
        ):
            with self.subTest(invalid=None if invalid is None else invalid.shape):
                with self.assertRaisesRegex(ValueError, "scale"):
                    impl._paged_kv_scale_args(
                        SimpleNamespace(kv_cache_base=quant, kv_scale_base=invalid)
                    )
        self.assertIsNone(
            main_kv_scale_view(quant.bfloat16(), None, kv_heads=2, page_size=128)
        )


if __name__ == "__main__":
    unittest.main()
