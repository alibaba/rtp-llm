import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import INDEXER_KV, INDEXER_STATE
from rtp_llm.models_py.modules.dsv4.decode.forward import build_metadata_eager
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    build_block_tables_batched,
    build_block_tables_batched_from_group_ids,
    resolve_block_table_group_ids,
)

PAGED_POOL_SPECS = {
    INDEXER_KV: (4, 128, 1),
    INDEXER_STATE: (4, 128, 1),
}
PAGED_TABLE_GROUP_IDS = {
    INDEXER_KV: 5,
    INDEXER_STATE: 6,
}


def _block_tables_by_group():
    return [
        torch.full((2, 1), group_id, dtype=torch.int32)
        for group_id in range(7)
    ]


class TestKVCacheGroupMapping(unittest.TestCase):

    def test_eager_uses_cached_physical_groups_without_layer_cache_scan(self):
        by_group = _block_tables_by_group()
        attn_inputs = SimpleNamespace(
            is_target_verify=False,
            sequence_lengths=torch.tensor([1, 2], dtype=torch.int32),
            kv_cache_kernel_block_id_device_by_group=by_group,
        )
        v4_args = SimpleNamespace(
            max_seq_len=64,
            window_size=8,
            head_dim=32,
            compress_ratios=[4],
            n_layers=1,
            index_topk=4,
        )

        with patch(
            "rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata."
            "build_decode_metadata",
            side_effect=lambda **kwargs: kwargs,
        ):
            metadata = build_metadata_eager(
                v4_args,
                attn_inputs,
                torch.device("cpu"),
                PAGED_POOL_SPECS,
                paged_table_group_ids=PAGED_TABLE_GROUP_IDS,
            )

        self.assertIs(metadata["paged_block_tables"][INDEXER_KV], by_group[5])
        self.assertIs(metadata["paged_block_tables"][INDEXER_STATE], by_group[6])

    def test_cached_group_ids_reject_incomplete_physical_block_tables(self):
        complete_tables = _block_tables_by_group()
        for by_group in (
            complete_tables[:6],
            complete_tables[:6] + [torch.empty((0, 1), dtype=torch.int32)],
        ):
            with self.subTest(block_table_count=len(by_group)):
                attn_inputs = SimpleNamespace(
                    kv_cache_kernel_block_id_device_by_group=by_group
                )
                with self.assertRaisesRegex(RuntimeError, "physical group 6"):
                    build_block_tables_batched_from_group_ids(
                        PAGED_TABLE_GROUP_IDS, attn_inputs
                    )

    def test_compact_mtp_without_physical_group_api_fails_fast(self):
        kv_cache = SimpleNamespace(
            group_region_names=[0, INDEXER_KV, INDEXER_STATE],
            layer_region_to_group_id=[[0, -1, -1, 1, 2, -1, -1, -1]],
        )

        with self.assertRaisesRegex(RuntimeError, "physical group"):
            resolve_block_table_group_ids(kv_cache)

    def test_initialized_dsv4_without_kv_cache_fails_fast(self):
        from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model

        model = SimpleNamespace(v4=object(), kv_cache=None)

        with self.assertRaisesRegex(RuntimeError, "kv_cache is None"):
            DeepSeekV4Model._cache_paged_decode_layout(model)

    def test_mtp_uses_layer_cache_physical_group_ids(self):
        layer_caches = [
            SimpleNamespace(
                region_name=INDEXER_KV,
                group_id=1,
                physical_group_id=5,
            ),
            SimpleNamespace(
                region_name=INDEXER_STATE,
                group_id=2,
                physical_group_id=6,
            ),
        ]
        kv_cache = SimpleNamespace(
            group_region_names=[
                0,
                0,
                INDEXER_KV,
                INDEXER_STATE,
                0,
                INDEXER_KV,
                INDEXER_STATE,
            ],
            layer_region_to_group_id=[[0, -1, -1, 1, 2, -1, -1, -1]],
            get_layer_caches=lambda _layer_id: layer_caches,
        )
        by_group = _block_tables_by_group()
        # These are the exact main INDEXER_KV kernel-page ids observed in the
        # cores. Interpreting local group 2 as draft INDEXER_STATE would trap.
        by_group[2] = torch.tensor([[192], [224]], dtype=torch.int32)
        attn_inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device_by_group=by_group
        )

        block_tables = build_block_tables_batched(kv_cache, attn_inputs)

        self.assertIs(block_tables[INDEXER_KV], by_group[5])
        self.assertIs(block_tables[INDEXER_STATE], by_group[6])

    def test_fp8_graph_snapshot_indexes_global_tables(self):
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImplConfigFP8,
            DSv4DecodeFmhaImplFP8,
        )

        config = DSv4DecodeFmhaImplConfigFP8(
            max_batch_size=2,
            q_len=1,
            window_size=8,
            head_dim=32,
            max_seq_len=64,
            compress_ratios=[4],
            index_topk=4,
            paged_pool_specs=PAGED_POOL_SPECS,
            paged_table_group_ids=PAGED_TABLE_GROUP_IDS,
        )
        impl = object.__new__(DSv4DecodeFmhaImplFP8)
        impl.config = config
        impl._paged_entries_per_block = {
            attn_type: spec[0] for attn_type, spec in PAGED_POOL_SPECS.items()
        }
        by_group = _block_tables_by_group()
        attn_inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device_by_group=by_group
        )

        block_tables = impl._extract_paged_block_tables(attn_inputs)

        self.assertIs(block_tables[INDEXER_KV], by_group[5])
        self.assertIs(block_tables[INDEXER_STATE], by_group[6])


if __name__ == "__main__":
    unittest.main()
