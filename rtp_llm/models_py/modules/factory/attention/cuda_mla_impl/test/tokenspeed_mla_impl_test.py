"""CPU contract tests for the TokenSpeed MLA framework adapter."""

from types import SimpleNamespace
from unittest import TestCase, main, mock

import torch

from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    _page_rr_cache_group,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_impl import (
    TokenSpeedMlaDecodeImpl,
    _TokenSpeedDecodeMetadata,
)
from rtp_llm.ops import HybridAttentionConfig, HybridAttentionType, ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import PyAttentionInputs

# ``PyAttentionInputs`` binds both layer-to-group maps as ``torch::Tensor``, and
# production assigns an undefined tensor when no map exists, so "absent" must be
# expressed as an empty tensor here rather than ``None``.
_ABSENT_LAYER_MAP = torch.empty(0, dtype=torch.int32)


class PageRrFactoryContractTest(TestCase):
    def setUp(self):
        self.inputs = PyAttentionInputs()
        self.inputs.kv_cache_kernel_block_id_device_by_group = [
            torch.empty((1, 4), dtype=torch.int32),
            torch.empty((1, 1), dtype=torch.int32),
        ]
        self.inputs.kv_cache_layer_to_group = torch.tensor(
            [1, 1, 0], dtype=torch.int32
        )
        self.parallel = ParallelismConfig()
        self.parallel.tp_size = 8
        self.hybrid = HybridAttentionConfig()
        self.hybrid.enable_hybrid_attention = True
        self.hybrid.enable_independent_kv_cache_pools = True
        self.hybrid.hybrid_attention_types = [
            HybridAttentionType.LINEAR,
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
        ]

    def test_role_flag_selects_group_zero_for_model_layer_two(self):
        for role in (RoleType.DECODE, RoleType.PREFILL, RoleType.PDFUSION):
            for sharded in (False, True):
                with self.subTest(role=role, sharded=sharded):
                    self.parallel.role_type = role
                    self.parallel.decode_cp_kv_cache_sharded = (
                        sharded if role == RoleType.DECODE else not sharded
                    )
                    self.parallel.prefill_cp_config.kv_cache_sharded = (
                        not sharded if role == RoleType.DECODE else sharded
                    )
                    for host_mirror in (
                        _ABSENT_LAYER_MAP,
                        self.inputs.kv_cache_layer_to_group,
                    ):
                        self.inputs.kv_cache_layer_to_group_host = host_mirror
                        self.assertEqual(
                            _page_rr_cache_group(
                                self.inputs, self.parallel, self.hybrid
                            ),
                            0 if sharded else None,
                        )

    def test_single_group_does_not_need_a_layer_map(self):
        self.parallel.role_type = RoleType.DECODE
        self.parallel.decode_cp_kv_cache_sharded = True
        self.inputs.kv_cache_layer_to_group = _ABSENT_LAYER_MAP
        self.inputs.kv_cache_kernel_block_id_device_by_group = (
            self.inputs.kv_cache_kernel_block_id_device_by_group[:1]
        )
        self.assertEqual(_page_rr_cache_group(self.inputs, self.parallel, None), 0)
        self.inputs.kv_cache_kernel_block_id_device_by_group = []
        self.assertEqual(_page_rr_cache_group(self.inputs, self.parallel, None), 0)
        self.assertIsNone(_page_rr_cache_group(self.inputs, None, None))
        self.parallel.tp_size = 1
        self.assertIsNone(_page_rr_cache_group(self.inputs, self.parallel, None))

    def test_physical_full_and_independent_swa_stay_distinct(self):
        self.parallel.role_type = RoleType.DECODE
        self.parallel.decode_cp_kv_cache_sharded = True
        for kind in (HybridAttentionType.LINEAR, HybridAttentionType.SLIDING_WINDOW):
            self.hybrid.hybrid_attention_types = [kind]
            self.assertIsNone(
                _page_rr_cache_group(self.inputs, self.parallel, self.hybrid)
            )
        self.hybrid.hybrid_attention_types = [
            HybridAttentionType.NONE, HybridAttentionType.SLIDING_WINDOW
        ]
        with self.assertRaisesRegex(ValueError, "independently allocated SWA"):
            _page_rr_cache_group(self.inputs, self.parallel, self.hybrid)
        self.hybrid.enable_independent_kv_cache_pools = False
        self.assertEqual(
            _page_rr_cache_group(self.inputs, self.parallel, self.hybrid), 1
        )


class TokenSpeedMlaGraphAdapterTest(TestCase):
    def test_prepare_cuda_graph_uses_fixed_capacity_plan(self) -> None:
        impl = object.__new__(TokenSpeedMlaDecodeImpl)
        impl.prepare = mock.Mock()
        inputs = SimpleNamespace()

        impl.prepare_cuda_graph(inputs)

        impl.prepare.assert_called_once_with(inputs, forbid_realloc=True)


class TokenSpeedMlaMetadataContractTest(TestCase):
    def test_graph_metadata_rejects_capacity_growth(self) -> None:
        metadata = _TokenSpeedDecodeMetadata(
            token_per_block=64,
            max_bs=1,
            max_context_len=64,
            use_cuda_graph=True,
            device=torch.device("cpu"),
        )
        too_many_rows = SimpleNamespace(
            qo_indptr_h=torch.arange(3, dtype=torch.int32),
            kvlen_h=torch.tensor([1, 1], dtype=torch.int32),
        )
        too_many_blocks = SimpleNamespace(
            qo_indptr_h=torch.arange(2, dtype=torch.int32),
            kvlen_h=torch.tensor([65], dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "too small for batch 2"):
            metadata.plan(too_many_rows)
        with self.assertRaisesRegex(ValueError, "needs 2 blocks, has 1"):
            metadata.plan(too_many_blocks)


if __name__ == "__main__":
    main()
