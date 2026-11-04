import unittest
from unittest.mock import Mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaFp8Op,
    SparseMlaImpl,
)
from rtp_llm.models_py.modules.hybrid.hy4_cmp_kernels.ops import indexer_k_fused
from rtp_llm.ops.compute_ops import PyAttentionInputs


def packed_scales(rows):
    scales = torch.empty((48, (rows + 3) // 4 * 4), device="cuda", dtype=torch.int32).T[
        :rows
    ]
    scales.fill_(0x7F7F7F7F)
    return scales


class CaptureCacheSlotsTest(unittest.TestCase):
    def test_capture_and_live_replay(self):
        torch.manual_seed(42)
        weight = torch.randn(128, 6144, device="cuda").to(torch.float8_e4m3fn)
        weight_scales = packed_scales(128)
        gamma = torch.ones(128, device="cuda", dtype=torch.bfloat16)
        beta = torch.zeros_like(gamma)
        angle = torch.randn(256, 32, device="cuda")
        cos_sin = torch.cat((angle.cos(), angle.sin()), dim=1)
        for batch in (1, 4, 8, 16, 32, 48, 64):
            for mode in ("decode", "target", "draft"):
                with self.subTest(batch=batch, mode=mode):
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
                    inputs.sequence_lengths = torch.full(
                        (batch,), 62, dtype=torch.int32
                    )
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
                    impl.is_cuda_graph = False
                    impl.create_params(inputs)
                    cache = torch.full(
                        (batch * 4, 64, 132), 173, dtype=torch.uint8, device="cuda"
                    )
                    if batch > 1:
                        with self.assertRaisesRegex(
                            ValueError, "duplicate live cache slots"
                        ):
                            indexer_k_fused.validate_metadata(
                                impl.fmha_params.positions_d,
                                cos_sin,
                                impl.fmha_params.slot_mapping,
                                cache,
                            )
                    impl.is_cuda_graph = True
                    impl.create_params(inputs)
                    params = impl.fmha_params
                    self.assertIs(impl.rope_params, params)
                    self.assertTrue(bool((params.slot_mapping == -1).all()))
                    slot_ptr = params.slot_mapping.data_ptr()
                    x = torch.randn(rows, 6144, device="cuda").to(torch.float8_e4m3fn)
                    tensors = (
                        x,
                        packed_scales(rows),
                        weight,
                        weight_scales,
                        gamma,
                        beta,
                        params.positions_d,
                        cos_sin,
                        params.slot_mapping,
                        cache,
                    )
                    out = indexer_k_fused.allocate_outputs(rows)
                    plan = indexer_k_fused.prepare_plan(*tensors, out=out)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        indexer_k_fused.run(*tensors, out=out, plan=plan)
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        indexer_k_fused.run(*tensors, out=out, plan=plan)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(bool((cache == 173).all()))
                    for prefix in (62, 127, 62):
                        table.copy_(
                            torch.arange(
                                batch * 4, device="cuda", dtype=torch.int32
                            ).view(batch, 4)
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
                        cache.fill_(173)
                        indexer_k_fused.run(*tensors, out=out, plan=plan)
                        expected_cache = cache.clone()
                        expected_key = out.key_bf16.clone()
                        self.assertTrue(bool((cache != 173).any()))
                        cache.fill_(173)
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            cache, expected_cache, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            out.key_bf16, expected_key, rtol=0, atol=0
                        )
                    print(f"PASS batch={batch} mode={mode} rows={rows}", flush=True)


if __name__ == "__main__":
    unittest.main()
