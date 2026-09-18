"""Replay V4.1 state writes, index scoring and candidate filtering on CUDA.

Run only on an idle GPU. The test mutates both token positions and input
features between replays, comparing the emitted keys/selections with eager.
"""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    INDEXER_KV,
    SWA_KV,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class AttentionV41CudaGraphTest(unittest.TestCase):
    def test_real_verify_metadata_construction_and_replay(self):
        from rtp_llm.models_py.modules.dsv4.decode.forward import (
            build_metadata_eager,
            decode_metadata_compress_ratios,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImplConfigFP8,
            DSv4DecodeFmhaImplFP8,
        )

        device = torch.device("cuda")
        args = SimpleNamespace(
            v41_config={"kv_source_layer_ids": [2, 8, 14, 20]},
            compress_ratios=[0, 0, 2, 1],
            n_layers=4,
            max_seq_len=256,
            window_size=128,
            head_dim=512,
            index_topk=512,
        )
        specs = {
            SWA_KV: (256, 128, 3),
            CSA_KV: (64, 128, 3),
            HCA_KV: (128, 128, 3),
            INDEXER_KV: (128, 128, 3),
            CSA_STATE: (8, 128, 3),
        }
        inputs = SimpleNamespace(
            is_target_verify=True,
            input_lengths=torch.tensor([6, 6]),
            prefix_lengths=torch.tensor([1, 127], dtype=torch.int32, device=device),
            kv_cache_kernel_block_id_device_by_group=[
                torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32, device=device)
                for _ in specs
            ],
        )
        config = DSv4DecodeFmhaImplConfigFP8(
            max_batch_size=2,
            q_len=6,
            window_size=128,
            head_dim=512,
            max_seq_len=256,
            compress_ratios=decode_metadata_compress_ratios(args),
            index_topk=512,
            paged_pool_specs=specs,
            group_region_names=list(specs),
        )
        impl = DSv4DecodeFmhaImplFP8(config, device, inputs)
        meta = impl.metadata
        self.assertEqual(meta.slot_mapping_compressed, {})
        self.assertEqual(set(meta.pool_block_tables), set(specs))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_positions = meta.position_ids.clone()
            captured_slots = meta.swa_global_slots.clone()
        for starts in ([3, 129], [250, 245]):
            inputs.prefix_lengths.copy_(
                torch.tensor(starts, dtype=torch.int32, device=device)
            )
            impl.prepare_cuda_graph(inputs)
            graph.replay()
            eager = build_metadata_eager(
                args,
                inputs,
                device,
                specs,
                kv_cache=SimpleNamespace(group_region_names=list(specs)),
                fp8_kv_cache=True,
            )
            torch.testing.assert_close(captured_positions, eager.position_ids)
            torch.testing.assert_close(captured_slots, eager.swa_global_slots)
            torch.testing.assert_close(
                captured_positions.view(2, 6),
                inputs.prefix_lengths[:, None]
                + torch.arange(6, device=device, dtype=torch.int32),
            )

    def test_compression_indexing_and_candidates_replay(self):
        device = torch.device("cuda")
        B, S, dim = 2, 6, 32
        attn = AttentionV41FP8.__new__(AttentionV41FP8)
        torch.nn.Module.__init__(attn)
        attn.layer_id = attn.kv_source_layer_id = attn.index_source_layer_id = 2
        attn.is_index_source = attn.is_kv_source = True
        attn.compress_ratio = 2
        attn.head_dim, attn.rope_head_dim = 512, 64
        attn.index_head_dim, attn.index_n_heads, attn.index_topk = 128, 4, 4
        attn.eps = 1e-6
        attn._rope_max_seq_len = 64
        attn._cp_ctx = None
        attn.v41_config = {
            "candidate_source_layer_id": 2,
            "candidate_topk_blocks": 2,
            "candidate_block_size": 4,
        }
        attn._shared_attention = {"layers": {2: attn}}
        torch.manual_seed(817)
        attn.global_wkv = (
            torch.randn(512, dim, device=device, dtype=torch.bfloat16) * 0.1
        )
        attn.global_wgate = (
            torch.randn(512, dim, device=device, dtype=torch.bfloat16) * 0.1
        )
        attn.global_norm = torch.ones(512, device=device, dtype=torch.bfloat16)
        attn.index_wk = (
            torch.randn(128, 512, device=device, dtype=torch.bfloat16) * 0.05
        )
        attn.index_k_norm = torch.ones(128, device=device, dtype=torch.bfloat16)
        attn.index_weights = (
            torch.randn(4, dim, device=device, dtype=torch.bfloat16) * 0.1
        )
        attn.index_wq = torch.nn.Linear(
            16, 4 * 128, bias=False, device=device, dtype=torch.bfloat16
        )
        attn.freqs_cis = torch.polar(
            torch.ones(64, 32, device=device), torch.randn(64, 32, device=device)
        )
        pools = {
            CSA_KV: torch.zeros(3, 64 * 288, device=device, dtype=torch.uint8),
            INDEXER_KV: torch.zeros(3, 128 * 68, device=device, dtype=torch.uint8),
            CSA_STATE: torch.zeros(3, 8 * 1024, device=device, dtype=torch.float32),
        }
        attn._pool_spec = {
            CSA_KV: (torch.uint8, 288),
            INDEXER_KV: (torch.uint8, 68),
            CSA_STATE: (torch.float32, 1024),
        }
        attn._kv_cache = SimpleNamespace(
            seq_size_per_block=128,
            kernel_seq_size_per_block=128,
            group_region_names=[CSA_KV, INDEXER_KV, CSA_STATE],
            get_layer_cache=lambda layer, region: SimpleNamespace(
                kv_cache_base=pools[int(region)]
            ),
        )
        attn._block_tables_by_type = {
            region: torch.tensor([[1], [2]], device=device, dtype=torch.int32)
            for region in pools
        }
        x = torch.randn(B, S, dim, device=device, dtype=torch.bfloat16)
        qr = torch.randn(B, S, 16, device=device, dtype=torch.bfloat16)
        starts = torch.tensor([10, 18], device=device)
        positions = (starts[:, None] + torch.arange(S, device=device)).reshape(-1)
        req = torch.arange(B, device=device).repeat_interleave(S)

        def forward():
            attn._begin_forward()
            attn._produce_global_decode(x, positions, req, starts)
            topk = attn._select_indices_decode(x, qr, positions)
            return (
                attn._shared_attention["global"][2].clone(),
                topk.clone(),
                attn._shared_attention["candidates"].clone(),
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward()
        for offset in (1, 6):
            x.normal_()
            qr.normal_()
            starts.copy_(torch.tensor([10 + offset, 18 + offset], device=device))
            positions.copy_(
                (starts[:, None] + torch.arange(S, device=device)).reshape(-1)
            )
            for pool in pools.values():
                pool.zero_()
            expected = forward()
            for pool in pools.values():
                pool.zero_()
            graph.replay()
            torch.cuda.synchronize()
            for actual, reference in zip(captured, expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
