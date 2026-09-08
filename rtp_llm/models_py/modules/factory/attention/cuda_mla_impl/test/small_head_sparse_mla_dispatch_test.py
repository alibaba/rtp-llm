"""Exercise the production H8 dispatch with pooled top-k expansion and tails."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaOp,
    _pad_flashmla_topk,
)
from rtp_llm.models_py.modules.indexer_grouping import (
    append_incomplete_tail_indices,
    expand_indexer_group_indices,
)
from small_head_sparse_mla_test import reference


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DispatchTest(unittest.TestCase):
    def test_native_heads_chunk_tail_and_launch_count(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            flashmla_sparse_impl,
        )

        torch.manual_seed(9361)
        kv = torch.randn(8193, 1, 512, device="cuda", dtype=torch.bfloat16)
        for tokens, legacy_chunk in ((17, 7), (4097, 4096)):
            lengths = torch.arange(tokens, device="cuda", dtype=torch.int32) % 8193 + 1
            pooled = (
                torch.randint(0, 2048, (tokens, 512), device="cuda", dtype=torch.int32)
                .sort(dim=-1)
                .values
            )
            ids = expand_indexer_group_indices(pooled, 4, raw_sequence_lengths=lengths)
            ids = append_incomplete_tail_indices(ids, lengths, 4)
            ids = _pad_flashmla_topk(ids.unsqueeze(1), 2176)
            q = torch.randn(tokens, 64, 512, device="cuda", dtype=torch.bfloat16)
            outputs = []
            for native_chunk in (legacy_chunk, 8192):
                with patch.dict(
                    os.environ,
                    {
                        "GLM53_SPARSE_MLA_BF16_Q_CHUNK": str(legacy_chunk),
                        "GLM53_SPARSE_MLA_NATIVE_Q_CHUNK": str(native_chunk),
                    },
                ):
                    op = SparseMlaOp(
                        64,
                        512,
                        0,
                        256,
                        128,
                        1.0,
                        2051,
                        indexer_top_k=512,
                        indexer_group_size=4,
                    )
                    with patch.object(
                        flashmla_sparse_impl,
                        "flash_mla_sparse_fwd",
                        wraps=flashmla_sparse_impl.flash_mla_sparse_fwd,
                    ) as kernel:
                        outputs.append(op._forward_sparse(q, kv, ids))
                    self.assertEqual(
                        kernel.call_count, (tokens + native_chunk - 1) // native_chunk
                    )
            torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)

    def test_pooled_tail_chunks_and_graph(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.flashinfer_bf16_small_head import (
            flashinfer_sparse_supported,
        )

        if not flashinfer_sparse_supported(
            torch.device("cuda", torch.cuda.current_device())
        ):
            self.skipTest("FlashInfer H8 requires SM100/SM103 and FlashInfer >= 0.6.14")
        env_patch = patch.dict(os.environ)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        torch.manual_seed(92)
        lengths = torch.tensor(
            [
                1,
                3,
                4,
                5,
                127,
                128,
                129,
                511,
                512,
                1023,
                1024,
                2048,
                4095,
                4096,
                8191,
                8192,
                8193,
            ],
            device="cuda",
            dtype=torch.int32,
        )
        pooled = (
            torch.randint(0, 2048, (17, 512), device="cuda", dtype=torch.int32)
            .sort(dim=-1)
            .values
        )
        expanded = expand_indexer_group_indices(pooled, 4, raw_sequence_lengths=lengths)
        expanded = append_incomplete_tail_indices(expanded, lengths, 4)
        ids = _pad_flashmla_topk(expanded.unsqueeze(1), 2176)
        q = torch.randn(17, 16, 512, device="cuda", dtype=torch.bfloat16)[:, ::2]
        kv = torch.randn(8193, 1, 512, device="cuda", dtype=torch.bfloat16)
        expected = reference(q, kv, ids, 0.0625)
        for backend in ("tilelang", "flashinfer", "auto"):
            with self.subTest(backend=backend):
                os.environ["GLM53_SPARSE_MLA_BF16_BACKEND"] = backend
                os.environ["GLM53_SPARSE_MLA_BF16_Q_CHUNK"] = "7"
                op = SparseMlaOp(
                    8,
                    512,
                    0,
                    256,
                    128,
                    1.0,
                    2051,
                    indexer_top_k=512,
                    indexer_group_size=4,
                )
                torch.testing.assert_close(
                    op._forward_sparse(q, kv, ids), expected, atol=0.016, rtol=0.015
                )
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = op._forward_sparse(q, kv, ids)
                g.replay()
                torch.testing.assert_close(out, expected, atol=0.016, rtol=0.015)
                # The real PD Prefill role restores page-RR KV through the CP
                # operator even when query tokens are not sequence-parallel.
                from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
                    SparseMlaFp8CPOp,
                )

                cp_op = SparseMlaFp8CPOp(
                    8,
                    512,
                    0,
                    256,
                    128,
                    1.0,
                    2051,
                    parallelism_config=SimpleNamespace(
                        tp_size=8,
                        tp_rank=3,
                        get_attn_tp_size=lambda: 8,
                        get_attn_tp_rank=lambda: 3,
                        prefill_cp_config=SimpleNamespace(
                            is_enabled=lambda: False,
                            kv_cache_sharded=True,
                        ),
                    ),
                    indexer_top_k=512,
                    indexer_group_size=4,
                )
                torch.testing.assert_close(
                    cp_op._forward_sparse_prefill(q, kv, ids),
                    expected,
                    atol=0.016,
                    rtol=0.015,
                )
                cp_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cp_graph):
                    cp_out = cp_op._forward_sparse_prefill(q, kv, ids)
                cp_graph.replay()
                torch.testing.assert_close(cp_out, expected, atol=0.016, rtol=0.015)
                # Also exercise the direct-return path when all queries fit.
                os.environ["GLM53_SPARSE_MLA_BF16_Q_CHUNK"] = "4096"
                full_op = SparseMlaOp(
                    8,
                    512,
                    0,
                    256,
                    128,
                    1.0,
                    2051,
                    indexer_top_k=512,
                    indexer_group_size=4,
                )
                torch.testing.assert_close(
                    full_op._forward_sparse(q, kv, ids),
                    expected,
                    atol=0.016,
                    rtol=0.015,
                )


if __name__ == "__main__":
    unittest.main()
