import unittest
from unittest.mock import Mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaFp8Op,
    SparseMlaImpl,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_qk_rope_cat_cache_mla import (
    fused_qk_rope_cat_cache_mla,
)
from rtp_llm.ops.compute_ops import PyAttentionInputs


class CaptureCacheSlotsTest(unittest.TestCase):
    def test_capture_and_live_replay(self):
        torch.manual_seed(42)
        angle = torch.randn(256, 32, device="cuda")
        cos_sin = torch.cat((angle.cos(), angle.sin()), dim=1)
        for batch in (1, 4, 8, 16, 32, 48, 64):
            for mode in ("decode", "target", "draft"):
                for cache_type in ("auto", "fp8_ds_mla"):
                    with self.subTest(batch=batch, mode=mode, cache=cache_type):
                        self._check_replay(batch, mode, cache_type, cos_sin)

    def _check_replay(self, batch, mode, cache_type, cos_sin):
        width = 1 if mode == "decode" else 4
        rows = batch * width
        inputs = PyAttentionInputs()
        inputs.is_prefill = mode != "decode"
        inputs.is_target_verify = mode == "target"
        inputs.is_draft_extend = mode == "draft"
        inputs.input_lengths = torch.full(
            (batch,), width, dtype=torch.int32, device="cuda"
        )
        inputs.prefix_lengths = (
            torch.full((batch,), 62, dtype=torch.int32, device="cuda")
            if width > 1
            else torch.empty(0, dtype=torch.int32)
        )
        inputs.sequence_lengths = torch.full((batch,), 62, dtype=torch.int32)
        inputs.sequence_lengths_plus_1_d = torch.full(
            (batch,), 63, dtype=torch.int32, device="cuda"
        )
        table = torch.zeros((batch, 4), dtype=torch.int32, device="cuda")
        inputs.kv_cache_block_id_device = table
        inputs.kv_cache_kernel_block_id_device = table
        inputs.kv_cache_block_id_host = table.cpu()
        inputs.kv_cache_kernel_block_id_host = table.cpu()
        impl = SparseMlaImpl.__new__(SparseMlaImpl)
        impl.seq_size_per_block = 64
        impl.fmha_impl = Mock(spec=SparseMlaFp8Op)
        impl._refresh_paged_mqa_schedule_metadata = Mock()
        impl._cuda_dag_indexer_metadata = None
        impl.is_cuda_graph = True
        impl.create_params(inputs)
        params = impl.fmha_params
        self.assertIs(impl.rope_params, params)
        self.assertTrue(bool((params.slot_mapping == -1).all()))
        slot_ptr = params.slot_mapping.data_ptr()
        fp8 = cache_type == "fp8_ds_mla"
        cache = torch.zeros(
            (batch * 4, 64, 656 if fp8 else 576),
            dtype=torch.uint8 if fp8 else torch.bfloat16,
            device="cuda",
        )
        q = torch.randn(rows, 8, 256, device="cuda", dtype=torch.bfloat16)
        ck = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
        kpe = torch.randn(rows, 64, device="cuda", dtype=torch.bfloat16)

        def run(query, key, target):
            fused_qk_rope_cat_cache_mla(
                query,
                ck,
                key,
                target,
                params.slot_mapping,
                params.positions_d,
                cos_sin,
                512,
                64,
                False,
                cache_type,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(q, kpe, cache)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run(q, kpe, cache)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(bool((cache == 0).all()))
        for prefix in (62, 127, 62):
            table.copy_(
                torch.arange(batch * 4, device="cuda", dtype=torch.int32).view(batch, 4)
            )
            inputs.prefix_lengths.fill_(prefix)
            inputs.sequence_lengths_plus_1_d.fill_(prefix + 1)
            impl.prepare_cuda_graph(inputs)
            self.assertEqual(slot_ptr, params.slot_mapping.data_ptr())
            expected_slots = (
                torch.arange(batch, device="cuda")[:, None] * 256
                + prefix
                + torch.arange(width, device="cuda")[None, :]
            ).flatten()
            torch.testing.assert_close(
                params.slot_mapping, expected_slots, rtol=0, atol=0
            )
            q.copy_(torch.randn_like(q))
            ck.copy_(torch.randn_like(ck))
            kpe.copy_(torch.randn_like(kpe))
            expected_q, expected_kpe = q.clone(), kpe.clone()
            expected_cache = torch.zeros_like(cache)
            run(expected_q, expected_kpe, expected_cache)
            cache.zero_()
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(bool((cache != 0).any()))
            torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
            torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
            torch.testing.assert_close(kpe, expected_kpe, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
