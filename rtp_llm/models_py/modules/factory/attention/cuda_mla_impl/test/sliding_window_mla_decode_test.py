"""Windowed MLA against real padded layer views and an independent mask oracle."""

from types import SimpleNamespace
from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.sliding_window_mla_decode import (
    SlidingWindowMlaDecodeOp,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import CacheGroupType, KVCache
from rtp_llm.utils.model_weight import W


class SlidingWindowMlaDecodeTest(TestCase):
    @torch.inference_mode()
    def test_padded_producer_window_and_graph_replay(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability()[0], 10)
        torch.manual_seed(904)
        kernel_page, dim, heads = 128, 576, 12
        cases = ((1, 1, 256, 1022), (14, 129, 256, 1022),
                 (28, 256, 256, 1022), (77, 255, 256, 1022),
                 (28, 2048, 4096, 1048574))
        for rows, window, page, context in cases:
            with self.subTest(rows=rows, window=window, context=context):
                queries = 1 if rows == 1 else 7
                batch = rows // queries
                pages = max(6, (context + queries + 3 + page - 1) // page)
                config = AttentionConfigs()
                config.head_num = heads
                config.kv_lora_rank, config.rope_head_dim = 512, 64
                config.nope_head_dim, config.v_head_dim = 128, 128
                config.tokens_per_block = page
                config.sliding_window = window
                config.kv_cache_dtype = KvCacheDataType.BASE
                config.softmax_extra_scale = 1.0
                weights = [{
                    W.mla_kc: torch.randn(heads, 128, 512, device="cuda", dtype=torch.bfloat16) * 0.025,
                    W.mla_vc: torch.randn(heads, 512, 128, device="cuda", dtype=torch.bfloat16) * 0.025,
                }]
                op = SlidingWindowMlaDecodeOp(config, weights)
                # Use the binding producer, including its layer offset and
                # per-kernel-page padding. Runtime FULL changes only the view.
                raw_stride = page * dim + (page // kernel_page) * 64
                raw = torch.full(
                    (2, batch * pages + 1, raw_stride), float("nan"),
                    dtype=torch.bfloat16, device="cuda",
                )
                owner = KVCache()
                owner.kv_cache_base_by_layer = [raw[0], raw[1]]
                owner.seq_size_per_block = page
                owner.kernel_seq_size_per_block = kernel_page
                owner.use_mla = True
                owner.kv_lora_rank, owner.rope_head_dim = 512, 64
                owner.layer_group_types = [CacheGroupType.FULL, CacheGroupType.FULL]
                cache = owner.get_layer_cache(1)
                view = cache.kv_cache_base
                self.assertEqual(view.stride(), (kernel_page * dim + 64, dim, 1))
                self.assertEqual(view.storage_offset(), raw[1].storage_offset())
                table_storage = torch.full((batch, pages + 3), -1, device="cuda", dtype=torch.int32)
                table = table_storage[:, :pages]
                table.copy_((torch.randperm(batch * pages, device="cuda") + 1).view(batch, pages))
                logical_kv = torch.randn(batch, pages * page, dim, device="cuda", dtype=torch.bfloat16)
                logical = torch.arange(pages * page, device="cuda")
                for request in range(batch):
                    slot = table[request, logical // page].long() * page + logical % page
                    view[slot // kernel_page, slot % kernel_page] = logical_kv[request]
                query = torch.randn(rows, heads, 192, device="cuda", dtype=torch.bfloat16)
                positions = (context + torch.arange(queries, device="cuda", dtype=torch.int32)).repeat(batch)
                requests = torch.arange(batch, device="cuda", dtype=torch.int32).repeat_interleave(queries)
                valid = torch.ones(rows, dtype=torch.bool, device="cuda")
                if rows > 1:
                    valid[-1] = False
                params = SimpleNamespace(
                    positions_d=positions, batch_indice_d=requests,
                    valid_queries=valid, block_table=table,
                )
                op.plan(params)

                def forward():
                    return op.forward(query[..., :128], query[..., 128:], cache, 0)

                def check(output):
                    # Reconstruct keys from the logical producer, independent of
                    # both the packed-window gather and sparse-index conversion.
                    for row, end in enumerate(positions.cpu().tolist()):
                        if row == rows - 1 and rows > 1:
                            torch.testing.assert_close(output[row], torch.zeros_like(output[row]))
                            continue
                        kv = logical_kv[row // queries, max(0, end - window + 1):end + 1].float()
                        key = torch.einsum("tr,hdr->htd", kv[:, :512], weights[0][W.mla_kc].float())
                        scores = torch.einsum("hd,htd->ht", query[row, :, :128].float(), key)
                        scores += query[row, :, 128:].float() @ kv[:, 512:].T
                        latent = (scores * (192**-0.5)).softmax(-1) @ kv[:, :512]
                        expected = torch.einsum("hr,hrv->hv", latent, weights[0][W.mla_vc].float())
                        torch.testing.assert_close(output[row].float(), expected, atol=1e-2, rtol=2e-2)

                for _ in range(3):
                    output = forward()
                check(output)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = forward()
                positions.add_(3)
                graph.replay()
                check(captured)
                self.assertTrue(torch.all(table_storage[:, pages:] == -1))
                # Attention reads never alter padding or reserved physical block0.
                self.assertTrue(torch.isnan(raw[1, 0]).all())
                self.assertTrue(torch.isnan(raw[1].view(-1, kernel_page * dim + 64)[:, -64:]).all())


if __name__ == "__main__":
    main()
