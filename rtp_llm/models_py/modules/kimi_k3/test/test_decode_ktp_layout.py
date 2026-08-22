import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.kimi_k3.decode_ktp import (
    DecodeOwnerLayout,
    KdaParallelContext,
    build_owner_attention_inputs,
)
from rtp_llm.models.kimi_k3.kimi_k3_weight import _KimiK3KDAWeight
from rtp_llm.model_loader.linear_attn_weight import LinearAttnConfig
from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.ops import LinearAttentionConfig, ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.utils.model_weight import W, identity


class DecodeOwnerLayoutTest(unittest.TestCase):
    def test_fixed_partition_covers_each_request_once(self):
        owners = [DecodeOwnerLayout.fixed(32, 8, rank) for rank in range(8)]
        self.assertEqual([owner.local_batch for owner in owners], [4] * 8)
        covered = [idx for owner in owners for idx in range(owner.start, owner.stop)]
        self.assertEqual(covered, list(range(32)))

    def test_rejects_non_divisible_or_too_small_batch(self):
        for batch in (1, 7, 9, 15):
            with self.assertRaisesRegex(ValueError, "BS divisible by TP"):
                DecodeOwnerLayout.fixed(batch, 8, 0)

    def test_owner_attention_metadata_keeps_order_and_group_rows(self):
        batch = 16
        attention = PyAttentionInputs()
        attention.is_prefill = False
        attention.is_target_verify = False
        attention.is_cuda_graph = False
        attention.cache_store_inputs = None
        attention.input_lengths = torch.ones(batch, dtype=torch.int32)
        attention.input_lengths_host = torch.arange(
            100, 100 + batch, dtype=torch.int32
        )
        attention.prefix_lengths = torch.arange(batch, dtype=torch.int32)
        attention.prefix_lengths_host = attention.prefix_lengths.clone()
        attention.sequence_lengths = torch.arange(100, 100 + batch, dtype=torch.int32)
        attention.sequence_lengths_host = attention.sequence_lengths.clone()
        attention.sequence_lengths_plus_1_d = attention.sequence_lengths + 1
        attention.kv_cache_block_id_host = torch.arange(
            2 * batch * 3, dtype=torch.int32
        ).reshape(2, batch, 3)
        group = torch.arange(batch * 4, dtype=torch.int32).reshape(batch, 4)
        attention.kv_cache_block_id_host_by_group = [group]
        attention.kv_cache_kernel_block_id_host_by_group = [group]
        attention.kv_cache_kernel_block_id_device_by_group = [group]

        layout = DecodeOwnerLayout.fixed(batch, 8, 3)
        local = build_owner_attention_inputs(
            attention,
            layout,
            device=torch.device("cpu"),
            global_query_tokens=batch,
        )

        self.assertEqual((layout.start, layout.stop), (6, 8))
        self.assertEqual(local.prefix_lengths_host.tolist(), [6, 7])
        self.assertEqual(local.sequence_lengths_host.tolist(), [106, 107])
        self.assertEqual(local.cu_seqlens.tolist(), [0, 1, 2])
        self.assertEqual(local.kv_cache_block_id_host.shape, (2, 2, 3))
        self.assertTrue(
            torch.equal(local.kv_cache_block_id_host_by_group[0], group[6:8])
        )

    def test_rejects_q_len_greater_than_one(self):
        attention = PyAttentionInputs()
        attention.is_prefill = False
        attention.is_target_verify = False
        attention.is_cuda_graph = False
        attention.cache_store_inputs = None
        attention.input_lengths_host = torch.arange(100, 108, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "q_len=1"):
            build_owner_attention_inputs(
                attention,
                DecodeOwnerLayout.fixed(8, 8, 7),
                device=torch.device("cpu"),
                global_query_tokens=9,
            )


def _target_parallelism(rank: int = 3) -> ParallelismConfig:
    config = ParallelismConfig()
    config.tp_size = 1
    config.dp_size = 8
    config.ep_size = 8
    config.ktp_size = 8
    config.world_size = 8
    config.world_rank = rank
    config.tp_rank = 0
    config.dp_rank = rank
    config.ktp_rank = rank
    config.ep_rank = rank
    config.role_type = RoleType.DECODE
    return config


class KdaParallelContextTest(unittest.TestCase):
    def test_target_topology_builds_independent_ktp_view(self):
        config = _target_parallelism()
        with patch.dict(os.environ, {"KIMI_K3_DECODE_KTP": "1"}):
            context = KdaParallelContext.from_parallelism(config)
        self.assertEqual((context.size, context.rank), (8, 3))
        self.assertEqual(context.group, Group.KTP)
        kda_config = context.parallelism_config(config)
        self.assertEqual((kda_config.tp_size, kda_config.tp_rank), (8, 3))
        self.assertEqual((config.tp_size, config.tp_rank), (1, 0))

    def test_disabled_path_keeps_existing_tp_contract(self):
        config = ParallelismConfig()
        config.tp_size = 8
        config.tp_rank = 5
        config.role_type = RoleType.DECODE
        with patch.dict(os.environ, {"KIMI_K3_DECODE_KTP": "0"}):
            context = KdaParallelContext.from_parallelism(config)
        self.assertEqual((context.size, context.rank, context.group), (8, 5, Group.TP))

    def test_rejects_mixed_topology_mismatch(self):
        for field, value in (
            ("tp_size", 2),
            ("dp_size", 4),
            ("ep_size", 4),
            ("ktp_size", 4),
            ("ktp_rank", 2),
        ):
            config = _target_parallelism()
            setattr(config, field, value)
            with patch.dict(os.environ, {"KIMI_K3_DECODE_KTP": "1"}):
                with self.assertRaisesRegex(ValueError, "TP1/DP=EP=KTP=WORLD"):
                    KdaParallelContext.from_parallelism(config)

    def test_process_group_is_separate_from_world_and_tp(self):
        config = _target_parallelism()
        ktp_group = object()
        saved = collective_torch._group_map
        collective_torch._group_map = {}
        try:
            with patch.dict(os.environ, {"KIMI_K3_DECODE_KTP": "1"}), patch.object(
                collective_torch.torch.distributed,
                "new_group",
                return_value=ktp_group,
            ) as new_group, patch.object(
                collective_torch.torch.distributed, "barrier"
            ):
                collective_torch._create_process_groups(config, "nccl", None)
            self.assertIs(collective_torch._group_map[Group.KTP], ktp_group)
            self.assertNotIn(Group.TP, collective_torch._group_map)
            new_group.assert_called_once()
            self.assertEqual(new_group.call_args.kwargs["ranks"], list(range(8)))
        finally:
            collective_torch._group_map = saved

    def test_disabled_switch_does_not_create_ktp_group(self):
        config = _target_parallelism()
        saved = collective_torch._group_map
        collective_torch._group_map = {}
        try:
            with patch.dict(os.environ, {"KIMI_K3_DECODE_KTP": "0"}), patch.object(
                collective_torch.torch.distributed, "new_group"
            ) as new_group:
                collective_torch._create_process_groups(config, "nccl", None)
            new_group.assert_not_called()
            self.assertNotIn(Group.KTP, collective_torch._group_map)
        finally:
            collective_torch._group_map = saved

    def test_kda_weight_uses_ktp_shard_not_global_tp1(self):
        linear = LinearAttentionConfig()
        linear.linear_num_key_heads = 64
        linear.linear_num_value_heads = 64
        linear.linear_key_head_dim = 128
        linear.linear_value_head_dim = 128
        weight = _KimiK3KDAWeight(
            W.linear_attn_alog,
            [],
            identity,
            LinearAttnConfig(linear),
            data_type=torch.float32,
            kda_tp_size=8,
            kda_tp_rank=3,
        )

        class FakeLoadConfig(SimpleNamespace):
            def model_copy(self, *, update):
                values = vars(self).copy()
                values.update(update)
                return FakeLoadConfig(**values)

        loaded = weight._split(
            torch.arange(64, dtype=torch.float32),
            FakeLoadConfig(tp_size=1, tp_rank=0),
        )[W.linear_attn_alog]
        self.assertEqual(loaded.tolist(), list(range(24, 32)))


if __name__ == "__main__":
    unittest.main()
