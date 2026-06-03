"""CPU-only checks for FlashMLA sparse decode graph-shaped inputs."""

from __future__ import annotations

import importlib
import sys
import types
import unittest

import torch


class FlashMlaDecodeGraphShapeTest(unittest.TestCase):
    def test_hca_extra_topk_width_tracks_max_seq_len(self) -> None:
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
            allocate_decode_metadata_fp8,
        )

        cases = (
            (128 * 1024, 1024),
            (512 * 1024, 4096),
            (1024 * 1024, 8192),
        )
        for max_seq_len, expected_hca_width in cases:
            with self.subTest(max_seq_len=max_seq_len):
                meta = allocate_decode_metadata_fp8(
                    max_batch_size=16,
                    q_len=4,
                    window_size=128,
                    head_dim=512,
                    max_seq_len=max_seq_len,
                    compress_ratios=[0, 4, 128],
                    index_topk=1024,
                    device=torch.device("cpu"),
                )
                hca_width = meta.topk_total_by_ratio[128].shape[-1] - meta.window_size
                self.assertEqual(hca_width, expected_hca_width)
                self.assertEqual(meta.batch_size, 16)
                self.assertEqual(meta.q_len_per_req, 4)

    def test_sparse_decode_wrapper_passes_graph_shape_without_effective_lengths(
        self,
    ) -> None:
        calls = []
        fake_flash_mla = types.ModuleType("flash_mla")

        def fake_get_mla_metadata(*_args, **_kwargs):
            return object(), None

        def fake_flash_mla_with_kvcache(**kwargs):
            calls.append(
                {
                    "q_shape": tuple(kwargs["q"].shape),
                    "indices_shape": tuple(kwargs["indices"].shape),
                    "extra_indices_shape": tuple(
                        kwargs["extra_indices_in_kvcache"].shape
                    ),
                    "block_table": kwargs["block_table"],
                    "cache_seqlens": kwargs["cache_seqlens"],
                    "topk_length": kwargs["topk_length"],
                    "extra_topk_length": kwargs["extra_topk_length"],
                }
            )
            q = kwargs["q"]
            head_dim_v = int(kwargs["head_dim_v"])
            out = torch.zeros(
                q.shape[0],
                q.shape[1],
                q.shape[2],
                head_dim_v,
                dtype=q.dtype,
                device=q.device,
            )
            lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1], device=q.device)
            return out, lse

        fake_flash_mla.get_mla_metadata = fake_get_mla_metadata
        fake_flash_mla.flash_mla_with_kvcache = fake_flash_mla_with_kvcache

        old_flash_mla = sys.modules.get("flash_mla")
        sys.modules["flash_mla"] = fake_flash_mla
        try:
            module = importlib.import_module(
                "rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op"
            )
            module._FLASH_MLA_AVAILABLE = True

            op = module.SparseAttnV4DecodeFp8Op(
                n_heads=2,
                head_dim=512,
                softmax_scale=512**-0.5,
            )

            graph_batch = 16
            q_len = 4
            hca_width_for_1m = 8192
            q = torch.zeros(graph_batch, q_len, 2, 512, dtype=torch.bfloat16)
            swa_pool = torch.zeros(2, 128, 584, dtype=torch.uint8)
            hca_pool = torch.zeros(2, 64, 584, dtype=torch.uint8)
            swa_indices = torch.full(
                (graph_batch, q_len, 128), -1, dtype=torch.int32
            )
            hca_indices = torch.full(
                (graph_batch, q_len, hca_width_for_1m), -1, dtype=torch.int32
            )

            # Simulate short real seq len by marking only a small prefix valid.
            # The wrapper still passes the full graph/max-width tensors below.
            swa_indices[:, :, :32] = 0
            hca_indices[:, :, :64] = 0

            op.forward(
                q=q,
                kv_cache=swa_pool,
                attn_sink=torch.zeros(2, dtype=torch.float32),
                topk_idxs=swa_indices,
                sched_meta=object(),
                extra_k_cache=hca_pool,
                extra_topk_idxs=hca_indices,
            )
        finally:
            if old_flash_mla is None:
                sys.modules.pop("flash_mla", None)
            else:
                sys.modules["flash_mla"] = old_flash_mla

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["q_shape"], (16, 4, 2, 512))
        self.assertEqual(call["indices_shape"], (16, 4, 128))
        self.assertEqual(call["extra_indices_shape"], (16, 4, 8192))
        self.assertIsNone(call["block_table"])
        self.assertIsNone(call["cache_seqlens"])
        self.assertIsNone(call["topk_length"])
        self.assertIsNone(call["extra_topk_length"])


if __name__ == "__main__":
    unittest.main()
