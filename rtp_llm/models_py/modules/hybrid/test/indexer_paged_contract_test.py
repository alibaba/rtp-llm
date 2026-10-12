import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.modules.base.cuda import indexer_op
from rtp_llm.models_py.modules.hybrid.indexer import Indexer


class IndexerPagedContractTest(unittest.TestCase):
    def make_indexer(self, *, fuse=True, cp=False):
        module = Indexer.__new__(Indexer)
        nn.Module.__init__(module)
        module.parallelism_config = SimpleNamespace(
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: cp)
        )
        module._fuse_logits_head_gate = fuse
        module._prefill_cp_fused_quant_enabled = Mock(return_value=True)
        q, scales = torch.zeros(4, 2, 128), torch.ones(4, 2, 1)
        module._fused_forward_decode = Mock(return_value=(q, scales))
        module._fused_forward_prefill_cp = Mock(return_value=(q, scales))
        module._get_q_k_bf16 = Mock(return_value=(q, q))
        module._quantize_q_k = Mock(return_value=(q, scales))
        module._get_logits_head_gate = Mock(return_value=scales)
        module._compute_topk = Mock(return_value=torch.zeros(4, 2))
        return module

    def call_indexer(self, module, inputs, cp_params=None):
        return module(
            torch.zeros(4, 256),
            torch.zeros(4, 128),
            object(),
            object(),
            inputs,
            use_fast_path=False,
            cp_params=cp_params,
        )

    def test_decode_verify_and_extend_share_fused_qk_path(self):
        for phase in ("decode", "target_verify", "draft_extend"):
            with self.subTest(phase=phase):
                inputs = SimpleNamespace(
                    is_prefill=phase != "decode",
                    is_target_verify=phase == "target_verify",
                    is_draft_extend=phase == "draft_extend",
                )
                module = self.make_indexer(cp=True)
                self.call_indexer(module, inputs)
                module._fused_forward_decode.assert_called_once()
                module._fused_forward_prefill_cp.assert_not_called()
                module._get_q_k_bf16.assert_not_called()

    def test_prefill_keeps_its_cp_fused_qk_path(self):
        module = self.make_indexer(cp=True)
        self.call_indexer(
            module,
            SimpleNamespace(is_prefill=True),
            SimpleNamespace(full_rope_pos_ids=torch.arange(4)),
        )
        module._fused_forward_prefill_cp.assert_called_once()
        module._fused_forward_decode.assert_not_called()

    def test_explicitly_disabled_fusion_keeps_reference_path(self):
        module = self.make_indexer(fuse=False)
        self.call_indexer(
            module,
            SimpleNamespace(is_prefill=True, is_target_verify=True),
        )
        module._get_q_k_bf16.assert_called_once()
        module._quantize_q_k.assert_called_once()
        module._fused_forward_decode.assert_not_called()

    def test_reference_decode_uses_local_positions_with_cp_configured(self):
        module = self.make_indexer(cp=True)
        module.index_n_heads, module.index_head_dim = 2, 128
        q, k = torch.zeros(4, 256), torch.zeros(4, 128)
        module.wq_b, module.wk = Mock(return_value=q), Mock(return_value=k)
        module.k_norm = Mock(side_effect=lambda value: value)
        module.indexer_op = Mock()
        module.indexer_op.apply_rope_and_rotate_q_k.return_value = (
            q.reshape(4, 2, 128),
            k,
        )
        positions = torch.arange(4)
        Indexer._get_q_k_bf16(
            module, q, k, SimpleNamespace(positions_d=positions), cp_params=None
        )
        module.indexer_op.apply_rope_and_rotate_q_k.assert_called_once()
        self.assertIs(
            module.indexer_op.apply_rope_and_rotate_q_k.call_args.args[2], positions
        )
        module.indexer_op.apply_rope_and_rotate_q_k_cp.assert_not_called()

    def make_operator(self):
        operator = indexer_op.IndexerOp.__new__(indexer_op.IndexerOp)
        nn.Module.__init__(operator)
        operator.index_n_heads = 2
        operator.index_head_dim = 128
        operator.index_topk = 2
        operator.blocksize = 64
        operator.block_size = 128
        operator._paged_topk_backend = "dsv4_persistent"
        operator._paged_topk_op = Mock(side_effect=lambda *args: args[2].fill_(0))
        return operator

    def test_stable_topk_equal_scores_padding_and_graph_replay(self):
        operator = indexer_op.IndexerOp(1, 128, 512, 0, stable_topk=True)
        self.assertEqual(operator._paged_topk_backend, "topk_v3_tie_break")
        self.assertEqual(operator._prefill_topk_backend, "topk_v3_tie_break")
        width = 4099
        scores = (
            torch.arange(width, device="cuda", dtype=torch.float32)
            .remainder(17)
            .expand(4, -1)
            .contiguous()
        )
        starts = torch.zeros(4, device="cuda", dtype=torch.int32)
        ends = torch.tensor([0, 128, 512, width], device="cuda", dtype=torch.int32)
        columns = torch.arange(width, device="cuda").expand(4, -1)
        masked = torch.where(columns < ends.unsqueeze(1), scores, -torch.inf)
        expected = torch.argsort(masked, dim=1, descending=True, stable=True)[
            :, :512
        ].to(torch.int32)
        expected = torch.where(expected < ends.unsqueeze(1), expected, -1)
        expected = indexer_op._canonicalize_topk_indices(expected)
        output = torch.empty(4, 512, device="cuda", dtype=torch.int32)
        for _ in range(5):
            operator._run_stable_topk(scores, starts, ends, output, width)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operator._run_stable_topk(scores, starts, ends, output, width)
        for _ in range(10):
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def check_decode_phase(self, phase):
        operator = self.make_operator()
        tokens_per_request = 1 if phase == "decode" else 3
        rows = 2 * tokens_per_request
        block_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        lengths = torch.arange(1, rows + 1, dtype=torch.int32)
        params = SimpleNamespace(
            expanded_seq_lens=lengths,
            batch_indice_d=torch.arange(2, dtype=torch.int32).repeat_interleave(
                tokens_per_request
            ),
            kvlen_d=lengths,
            schedule_metadata=torch.ones(1, dtype=torch.int32),
        )
        inputs = SimpleNamespace(
            is_target_verify=phase == "verify",
            is_draft_extend=phase == "draft_extend",
            kv_cache_kernel_block_id_device=block_table,
            decode_cu_seqlens_device=torch.tensor([0, 1, 2], dtype=torch.int32),
        )
        cache = SimpleNamespace(
            seq_size_per_block=64,
            kv_cache_base=torch.zeros(5, 64 * 132, dtype=torch.uint8),
        )
        logits = torch.zeros(rows, 128)
        kernel = Mock(return_value=logits)
        with (
            patch.object(
                indexer_op, "deep_gemm", SimpleNamespace(fp8_paged_mqa_logits=kernel)
            ),
            patch.object(
                indexer_op, "_get_topk_workspace", return_value=torch.empty(0)
            ),
        ):
            result = operator._get_topk_paged(
                torch.zeros(rows, 2, 128), torch.ones(rows, 2, 1), cache, params, inputs
            )

        args = kernel.call_args.args
        self.assertEqual(args[1].shape, (5, 64, 1, 132))
        self.assertEqual(args[1].data_ptr(), cache.kv_cache_base.data_ptr())
        self.assertEqual(args[1].dtype, torch.uint8)
        torch.testing.assert_close(args[3], lengths.reshape(-1, 1))
        torch.testing.assert_close(
            args[4], block_table.repeat_interleave(tokens_per_request, dim=0)
        )
        self.assertEqual(result.shape, (rows, 2))
        torch.testing.assert_close(operator._paged_topk_op.call_args.args[1], lengths)

    def test_decode_uses_validated_packed_cache(self):
        self.check_decode_phase("decode")

    def test_target_verify_repeats_request_block_tables(self):
        self.check_decode_phase("verify")

    def test_draft_extend_repeats_request_block_tables(self):
        self.check_decode_phase("draft_extend")

    def test_ragged_draft_extend_maps_tokens_with_empty_padded_requests(self):
        operator = self.make_operator()
        table = torch.arange(12, dtype=torch.int32).reshape(4, 3)
        mapping = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2], dtype=torch.int32)
        lengths = torch.arange(10, dtype=torch.int32) + 64
        params = SimpleNamespace(
            batch_indice_d=mapping,
            expanded_seq_lens=lengths,
            schedule_metadata=torch.ones(1, dtype=torch.int32),
        )
        inputs = SimpleNamespace(
            is_target_verify=False,
            is_draft_extend=True,
            kv_cache_kernel_block_id_device=table,
        )
        cache = SimpleNamespace(
            seq_size_per_block=64,
            kv_cache_base=torch.zeros(16, 64 * 132, dtype=torch.uint8),
        )
        kernel = Mock(return_value=torch.zeros(10, 192))
        with (
            patch.object(
                indexer_op, "deep_gemm", SimpleNamespace(fp8_paged_mqa_logits=kernel)
            ),
            patch.object(
                indexer_op, "_get_topk_workspace", return_value=torch.empty(0)
            ),
        ):
            result = operator._get_topk_paged(
                torch.zeros(10, 2, 128), torch.ones(10, 2, 1), cache, params, inputs
            )
        torch.testing.assert_close(kernel.call_args.args[4], table[mapping.long()])
        torch.testing.assert_close(kernel.call_args.args[3], lengths.reshape(-1, 1))
        self.assertEqual(result.shape, (10, 2))

    def test_cache_geometry_mismatch_is_rejected_before_kernel(self):
        cache = SimpleNamespace(
            seq_size_per_block=32,
            kv_cache_base=torch.zeros(5, 64 * 132, dtype=torch.uint8),
        )
        with self.assertRaisesRegex(RuntimeError, "page geometry mismatch"):
            self.make_operator()._indexer_cache_view(cache)


if __name__ == "__main__":
    unittest.main()
