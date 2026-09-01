import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

# Importing the package performs device-specific implementation registration.
import rtp_llm.models_py.modules.factory.attention  # noqa: F401, E402
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    DECODE_MLA_IMPS,
    PREFILL_MLA_IMPS,
    _supports_sparse_prefill_dense_fast_path,
)
from rtp_llm.ops import KvCacheDataType


class SparseMlaRegistryTest(unittest.TestCase):
    def test_base_sparse_mla_registration_is_not_coupled_to_cp(self) -> None:
        prefill_names = {impl.__name__ for impl in PREFILL_MLA_IMPS}
        decode_names = {impl.__name__ for impl in DECODE_MLA_IMPS}

        self.assertIn("SparseMlaImpl", prefill_names)
        self.assertIn("SparseMlaImpl", decode_names)

    def test_fp8_nope_cache_stays_on_sparse_prefill(self) -> None:
        glm53_fp8 = SimpleNamespace(
            kv_cache_dtype=KvCacheDataType.FP8,
            rope_head_dim=0,
        )
        ds_fp8 = SimpleNamespace(
            kv_cache_dtype=KvCacheDataType.FP8,
            rope_head_dim=64,
        )
        glm53_bf16 = SimpleNamespace(
            kv_cache_dtype=KvCacheDataType.BASE,
            rope_head_dim=0,
        )

        self.assertFalse(_supports_sparse_prefill_dense_fast_path(glm53_fp8))
        self.assertTrue(_supports_sparse_prefill_dense_fast_path(ds_fp8))
        self.assertTrue(_supports_sparse_prefill_dense_fast_path(glm53_bf16))

    def test_tp_page_rr_prefill_selects_sharded_sparse_mla(self) -> None:
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            SparseMlaCpImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            SparseMlaImpl,
        )

        page_rr_tp8 = SimpleNamespace(
            tp_size=8,
            prefill_cp_config=SimpleNamespace(
                is_enabled=lambda: False,
                kv_cache_sharded=True,
            ),
        )
        plain_tp8 = SimpleNamespace(
            tp_size=8,
            prefill_cp_config=SimpleNamespace(
                is_enabled=lambda: False,
                kv_cache_sharded=False,
            ),
        )

        self.assertFalse(SparseMlaImpl.support_parallelism_config(page_rr_tp8))
        self.assertTrue(SparseMlaCpImpl.support_parallelism_config(page_rr_tp8))
        self.assertTrue(SparseMlaImpl.support_parallelism_config(plain_tp8))

    def test_tp_page_rr_does_not_all_gather_current_token_projection(self) -> None:
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            SparseMlaFp8CPOp,
        )

        op = object.__new__(SparseMlaFp8CPOp)
        op.sequence_parallel = False
        op.kv_cache_sharded = True
        op.kv_restore_unpad_indices = torch.arange(3, dtype=torch.int64)
        op.kv_cache_write_op = Mock()
        op.write_cache_store_impl = None
        op.attn_inputs = SimpleNamespace(is_prefill=False)
        op.mla_params = object()
        op.sharded_slot_mapping = torch.tensor([2, -1, 7], dtype=torch.int64)
        op.total_local_ids = torch.empty(0, dtype=torch.int64)
        op._gather = None

        compressed_kv = torch.randn(3, 4)
        k_pe = torch.empty(3, 0)
        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "flashmla_sparse_cp_impl.all_gather"
        ) as gather:
            actual = op.forward(
                torch.empty(3, 1, 4),
                compressed_kv,
                k_pe,
                None,
                torch.zeros(3, dtype=torch.int32),
                object(),
            )

        self.assertIsNone(actual)
        gather.assert_not_called()
        write_args = op.kv_cache_write_op.forward.call_args
        self.assertIs(write_args.args[0], compressed_kv)
        self.assertIs(write_args.args[1], k_pe)
        self.assertIs(
            write_args.kwargs["slot_mapping_override"], op.sharded_slot_mapping
        )

    def test_tp_page_rr_routes_sparse_attention_through_local_tp_kernel(self) -> None:
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            SparseMlaFp8CPOp,
            _GatherWorkspace,
        )

        op = object.__new__(SparseMlaFp8CPOp)
        op._gather = _GatherWorkspace(
            workspace_starts=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([3], dtype=torch.int32),
            total_kv_len=3,
            batch_size=1,
        )
        op.precomputed_req_ids = torch.zeros(2, dtype=torch.int64)
        op.kv_cache_sharded = True
        op.kernel_top_k = 128
        op._allocate_fused_kv = Mock(return_value=torch.randn(3, 4))
        op._gather_sharded_kv_cache = Mock()
        op._prepare_cp_local_topk_indices = Mock(
            return_value=torch.tensor([[0, 1], [1, -1]], dtype=torch.int64)
        )
        expected = torch.randn(2, 8, 4)
        op._forward_sparse_prefill = Mock(return_value=expected)

        q = torch.randn(2, 8, 6)
        actual = op._attend_gather(
            q,
            SimpleNamespace(kv_cache_base=torch.empty(0)),
            torch.empty(2, 1, 2, dtype=torch.int64),
        )

        torch.testing.assert_close(actual, expected)
        op._forward_sparse_prefill.assert_called_once()
        sparse_args = op._forward_sparse_prefill.call_args.args
        self.assertIs(sparse_args[0], q)
        self.assertEqual(tuple(sparse_args[1].shape), (3, 1, 4))
        self.assertEqual(tuple(sparse_args[2].shape), (2, 1, 128))
        self.assertEqual(sparse_args[2][1, 0, :4].tolist(), [1, -1, -1, -1])

    def test_tp_page_rr_local_kernel_does_not_gather_query_heads(self) -> None:
        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            SparseMlaFp8CPOp,
        )

        op = object.__new__(SparseMlaFp8CPOp)
        op.sequence_parallel = False
        op.attn_tp_size = 8
        op.num_heads = 8
        op.qk_rope_head_dim = 0
        op.kv_lora_rank = 4
        op.scale = 0.25
        op._tp_attention_sink = None

        q = torch.randn(2, 8, 4)
        kv = torch.randn(3, 1, 4)
        indices = torch.tensor([[[0, 1]], [[1, -1]]], dtype=torch.int32)
        expected = torch.randn_like(q)

        with patch.object(
            tilelang_kernels, "tilelang_available", return_value=True
        ), patch.object(
            tilelang_kernels, "sparse_attn", return_value=expected.unsqueeze(0)
        ) as sparse_attn, patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "flashmla_sparse_impl.all_gather"
        ) as gather:
            actual = op._forward_sparse_prefill(q, kv, indices)

        torch.testing.assert_close(actual, expected)
        gather.assert_not_called()
        sparse_attn.assert_called_once()
        args = sparse_attn.call_args.args
        self.assertEqual(tuple(args[0].shape), (1, 2, 8, 4))
        self.assertEqual(tuple(args[1].shape), (1, 3, 4))
        self.assertTrue(torch.isneginf(args[2]).all().item())
        self.assertEqual(tuple(args[3].shape), (1, 2, 2))
        self.assertEqual(args[4], 0.25)

    @patch("torch.cuda.current_device", return_value=0)
    def test_tp_page_rr_operator_keeps_attention_tp_geometry(self, _) -> None:
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
            SparseMlaFp8CPOp,
        )

        parallelism = SimpleNamespace(
            tp_size=8,
            tp_rank=5,
            get_attn_tp_size=lambda: 8,
            get_attn_tp_rank=lambda: 5,
            prefill_cp_config=SimpleNamespace(
                is_enabled=lambda: False,
                kv_cache_sharded=True,
            ),
        )
        op = SparseMlaFp8CPOp(
            num_heads=8,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            qk_nope_head_dim=256,
            page_size=64,
            softmax_extra_scale=1.0,
            top_k=2051,
            parallelism_config=parallelism,
            indexer_top_k=512,
            indexer_group_size=4,
        )

        self.assertEqual(op.attn_tp_size, 8)
        self.assertEqual(op.attn_tp_rank, 5)
        self.assertFalse(op.sequence_parallel)
        self.assertTrue(op.kv_cache_sharded)


if __name__ == "__main__":
    unittest.main()
