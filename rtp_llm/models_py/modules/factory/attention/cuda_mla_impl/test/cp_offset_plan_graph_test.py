import os
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.cp_topk_offsets_triton import (
    offset_cp_topk,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
    SparseMlaFp8CPOp,
)
from rtp_llm.ops import ParallelismConfig


class CpOffsetPlanGraphTest(TestCase):
    @patch.dict(os.environ, {"USE_GATHER_PATH": "1", "RTP_CP_FUSED_TOPK_OFFSETS": "1"})
    def test_graph_reads_refreshed_per_request_offsets_without_changing_address(
        self,
    ) -> None:
        cp_info = SimpleNamespace(
            prefill_cp_chunk_lengths=[2, 2],
            prefill_qkv_padding_mask=torch.ones(4, dtype=torch.int32),
            prefill_qkv_restore_indice=torch.arange(4, dtype=torch.long),
            prefill_actual_input_lengths_cpu=torch.tensor([5, 7], dtype=torch.int32),
        )
        inputs = SimpleNamespace(
            context_parallel_info=cp_info,
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            is_prefill=True,
        )
        params = SimpleNamespace(
            fill_cp_plan_params=lambda *_args: None,
            cp_kv_restore_unpad_indices=torch.arange(4, device="cuda"),
            cp_total_global_ids=torch.arange(4, device="cuda"),
            cp_total_local_ids=torch.arange(4, device="cuda"),
            cp_cu_kv_seqlens_global=torch.tensor(
                [0, 5, 12], dtype=torch.int32, device="cuda"
            ),
            cp_total_kv_len=12,
            batch_indice_d=torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda"),
            positions_d=torch.arange(4, dtype=torch.int32, device="cuda"),
            prefill_ragged_kv_len_indptr_d=torch.tensor(
                [0, 5, 12], dtype=torch.int32, device="cuda"
            ),
        )
        op = SparseMlaFp8CPOp(
            num_heads=64,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=512,
            page_size=64,
            softmax_extra_scale=1.0,
            top_k=128,
            parallelism_config=ParallelismConfig(),
            use_cuda_graph=True,
        )
        op.kv_cache_sharded = True
        blocks = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
        op.plan(params, blocks, inputs)
        address = op.precomputed_kv_offsets.data_ptr()
        indices = torch.tensor(
            [[0, -1], [1, 2], [-2, 3], [4, 0]], device="cuda", dtype=torch.int32
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            offset_cp_topk(indices, op.precomputed_kv_offsets)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = offset_cp_topk(indices, op.precomputed_kv_offsets)

        for request_ids, lengths in (([1, 0, 1, 0], [7, 5]), ([0, 1, 0, 1], [3, 9])):
            cp_info.prefill_actual_input_lengths_cpu.copy_(
                torch.tensor(lengths, dtype=torch.int32)
            )
            params.batch_indice_d.copy_(
                torch.tensor(request_ids, device="cuda", dtype=torch.int32)
            )
            params.prefill_ragged_kv_len_indptr_d.copy_(
                torch.tensor(
                    [0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32
                )
            )
            params.cp_cu_kv_seqlens_global = params.prefill_ragged_kv_len_indptr_d
            op.plan(params, blocks, inputs)
            self.assertEqual(op.precomputed_kv_offsets.data_ptr(), address)
            expected_offsets = torch.tensor(
                [0 if rid == 0 else lengths[0] for rid in request_ids],
                device="cuda",
                dtype=torch.int32,
            )
            torch.testing.assert_close(
                op.precomputed_kv_offsets, expected_offsets, atol=0, rtol=0
            )

            graph.replay()

            expected = (indices + expected_offsets[:, None]).masked_fill(
                indices < 0, -1
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    main()
