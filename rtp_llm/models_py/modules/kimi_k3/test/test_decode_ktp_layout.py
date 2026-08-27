import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.engine_config import _reject_legacy_k3_mla_cache_tp
from rtp_llm.models.kimi_k3.decode_ktp import (
    DecodeOwnerLayout,
    KdaParallelContext,
    KtpBatchPlan,
    KtpStepDescriptor,
    build_ktp_cuda_graph_attention_inputs,
    build_ktp_attention_inputs,
    build_owner_attention_inputs,
    rendezvous_ktp_step,
)
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models.kimi_k3.kimi_k3_weight import _KimiK3KDAWeight
from rtp_llm.model_loader.linear_attn_weight import LinearAttnConfig
from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.kimi_k3.kda import KimiK3KDA
from rtp_llm.models_py.modules.kimi_k3.moe import (
    KimiK3LatentMoE,
    _validate_k3_mega_parallelism,
)
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


def _target_parallelism(rank: int = 3, world_size: int = 8) -> ParallelismConfig:
    config = ParallelismConfig()
    config.tp_size = 1
    config.dp_size = world_size
    config.ep_size = world_size
    config.ktp_size = world_size
    config.world_size = world_size
    config.world_rank = rank
    config.tp_rank = 0
    config.dp_rank = rank
    config.ktp_rank = rank
    config.ep_rank = rank
    config.role_type = RoleType.DECODE
    return config


class KdaParallelContextTest(unittest.TestCase):
    def test_legacy_mla_cache_tp_environment_fails_fast(self):
        with patch.dict(os.environ, {"KIMI_K3_MLA_CACHE_TP": "1"}):
            with self.assertRaisesRegex(ValueError, "retired 576/TP"):
                _reject_legacy_k3_mla_cache_tp()
        with patch.dict(os.environ, {"KIMI_K3_MLA_CACHE_TP": "0"}):
            _reject_legacy_k3_mla_cache_tp()

    def test_mega_moe_accepts_decode_dp8_ktp8_owner_local_tokens(self):
        with patch.dict(os.environ, {"KIMI_K3_DECODE_KTP": "1"}):
            mode = _validate_k3_mega_parallelism(
                attn_tp_size=1,
                dp_size=8,
                ep_size=8,
                ktp_size=8,
                world_size=8,
                local_expert_count=112,
                role_type=RoleType.DECODE,
            )
        self.assertEqual(mode, "dp8_ep8_ktp8")

        with patch.dict(os.environ, {"KIMI_K3_DECODE_KTP": "1"}):
            mode16 = _validate_k3_mega_parallelism(
                attn_tp_size=1,
                dp_size=16,
                ep_size=16,
                ktp_size=16,
                world_size=16,
                local_expert_count=56,
                role_type=RoleType.DECODE,
            )
        self.assertEqual(mode16, "dp16_ep16_ktp16")

        fake_moe = SimpleNamespace(
            attn_tp_size=1,
            dp_size=8,
            ep_size=8,
            ktp_size=8,
            local_expert_count=112,
            parallelism_config=SimpleNamespace(role_type=RoleType.DECODE),
        )
        with patch.dict(os.environ, {"KIMI_K3_DECODE_KTP": "1"}):
            self.assertEqual(
                KimiK3LatentMoE._mega_parallel_mode(fake_moe, 8),
                "dp8_ep8_ktp8",
            )

    def test_mega_moe_rejects_dp8_without_ktp_contract(self):
        with patch.dict(os.environ, {"KIMI_K3_DECODE_KTP": "0"}):
            with self.assertRaisesRegex(RuntimeError, "TP1/DP=EP=KTP=WORLD"):
                _validate_k3_mega_parallelism(
                    attn_tp_size=1,
                    dp_size=8,
                    ep_size=8,
                    ktp_size=8,
                    world_size=8,
                    local_expert_count=112,
                    role_type=RoleType.DECODE,
                )

    def test_mega_preconditions_use_decode_dp_ktp_topology_contract(self):
        for width in (8, 16):
            fake_moe = SimpleNamespace(
                attn_tp_size=1,
                dp_size=width,
                ep_size=width,
                ktp_size=width,
                local_expert_count=896 // width,
                parallelism_config=SimpleNamespace(role_type=RoleType.DECODE),
            )
            fake_moe._mega_parallel_mode = lambda world_size, moe=fake_moe: (
                KimiK3LatentMoE._mega_parallel_mode(moe, world_size)
            )
            with patch.dict(os.environ, {"KIMI_K3_DECODE_KTP": "1"}), patch(
                "torch.cuda.is_available", return_value=True
            ), patch(
                "torch.cuda.get_device_capability", return_value=(10, 0)
            ), patch(
                "torch.distributed.is_initialized", return_value=True
            ), patch(
                "torch.distributed.get_world_size", return_value=width
            ):
                KimiK3LatentMoE._validate_mega_preconditions(
                    fake_moe, "K3 DeepGEMM MegaMoE SE"
                )

        invalid_moe = SimpleNamespace(
            attn_tp_size=1,
            dp_size=4,
            ep_size=4,
            ktp_size=4,
            local_expert_count=224,
            parallelism_config=SimpleNamespace(role_type=RoleType.DECODE),
        )
        invalid_moe._mega_parallel_mode = lambda world_size: (
            KimiK3LatentMoE._mega_parallel_mode(invalid_moe, world_size)
        )
        with patch.dict(os.environ, {"KIMI_K3_DECODE_KTP": "1"}), patch(
            "torch.cuda.is_available", return_value=True
        ), patch(
            "torch.cuda.get_device_capability", return_value=(10, 0)
        ), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch(
            "torch.distributed.get_world_size", return_value=4
        ):
            with self.assertRaisesRegex(RuntimeError, "K3 DeepGEMM MegaMoE SE"):
                KimiK3LatentMoE._validate_mega_preconditions(
                    invalid_moe, "K3 DeepGEMM MegaMoE SE"
                )

    def test_target_topology_builds_independent_ktp_view(self):
        config = _target_parallelism()
        with patch.dict(os.environ, {"KIMI_K3_DECODE_KTP": "1"}):
            context = KdaParallelContext.from_parallelism(config)
        self.assertEqual((context.size, context.rank), (8, 3))
        self.assertEqual(context.group, Group.KTP)
        kda_config = context.parallelism_config(config)
        self.assertEqual((kda_config.tp_size, kda_config.tp_rank), (8, 3))
        self.assertEqual((config.tp_size, config.tp_rank), (1, 0))

        config16 = _target_parallelism(rank=15, world_size=16)
        with patch.dict(os.environ, {"KIMI_K3_DECODE_KTP": "1"}):
            context16 = KdaParallelContext.from_parallelism(config16)
        self.assertEqual((context16.size, context16.rank), (16, 15))

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

    def test_world16_process_group_contains_all_ktp_ranks(self):
        config = _target_parallelism(rank=8, world_size=16)
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
            new_group.assert_called_once()
            self.assertEqual(new_group.call_args.kwargs["ranks"], list(range(16)))
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


class KtpBatchRendezvousTest(unittest.TestCase):
    @staticmethod
    def _descriptors(
        local_batches, *, bucket=4, step_epoch=7
    ) -> tuple[KtpStepDescriptor, ...]:
        return tuple(
            KtpStepDescriptor.build(
                rank=rank,
                step_epoch=step_epoch,
                local_batch=local_batch,
                bucket=bucket,
                request_ids=range(rank * 100, rank * 100 + local_batch),
                generation_epochs=[3] * local_batch,
            )
            for rank, local_batch in enumerate(local_batches)
        )

    def test_rendezvous_supports_empty_and_uneven_dp_batches(self):
        local_batches = [1, 2, 3, 4, 1, 2, 3, 4]
        descriptors = self._descriptors(local_batches)
        gathered = torch.cat(
            [descriptor.pack(torch.device("cpu")) for descriptor in descriptors]
        )
        context = KdaParallelContext(8, 3, Group.KTP)
        with patch(
            "rtp_llm.models.kimi_k3.decode_ktp.all_gather",
            return_value=gathered,
        ):
            plan = rendezvous_ktp_step(
                context,
                step_epoch=7,
                local_batch=4,
                fixed_bucket=4,
                device=torch.device("cpu"),
                request_ids=range(300, 304),
                generation_epochs=[3] * 4,
            )
        self.assertEqual(plan.valid_rows, sum(local_batches))
        self.assertEqual(plan.physical_rows, 32)
        self.assertEqual(plan.local_batch, 4)
        self.assertEqual(
            [key for key in plan.request_keys if key is not None][0], (0, 3)
        )
        self.assertEqual(plan.valid_mask(torch.device("cpu")).sum().item(), 20)

    def test_partial_and_all_idle_ranks_still_have_physical_rows(self):
        partial = KtpBatchPlan(
            self._descriptors([0, 1, 0, 1, 0, 1, 0, 1], bucket=1), rank=0
        )
        idle = KtpBatchPlan(self._descriptors([0] * 8, bucket=1), rank=5)
        self.assertEqual((partial.valid_rows, partial.physical_rows), (4, 8))
        self.assertEqual((idle.valid_rows, idle.physical_rows), (0, 8))
        self.assertFalse(idle.valid_mask(torch.device("cpu")).any())

    def test_cuda_graph_plan_uses_one_full_fixed_bucket_on_every_rank(self):
        plan = KtpBatchPlan.for_cuda_graph(
            KdaParallelContext(8, 5, Group.KTP), fixed_bucket=4
        )
        self.assertEqual(plan.rank, 5)
        self.assertEqual(plan.local_batch, 4)
        self.assertEqual(plan.valid_rows, 32)
        self.assertEqual(plan.physical_rows, 32)
        self.assertTrue(plan.valid_mask(torch.device("cpu")).all())

    def test_compact_and_expand_never_execute_padding_rows(self):
        plan = KtpBatchPlan(
            self._descriptors([0, 1, 0, 2, 0, 1, 0, 2], bucket=2), rank=3
        )
        physical = torch.arange(16, dtype=torch.float32).reshape(16, 1)
        compact = plan.compact_valid_rows(physical)
        self.assertEqual(compact.flatten().tolist(), [2.0, 6.0, 7.0, 10.0, 14.0, 15.0])

        expanded = plan.expand_valid_rows(compact)
        self.assertEqual(tuple(expanded.shape), (16, 1))
        self.assertTrue(
            torch.equal(expanded[list(plan.valid_physical_indices)], compact)
        )
        padding = expanded[~plan.valid_mask(torch.device("cpu"))]
        self.assertTrue(torch.equal(padding, torch.zeros_like(padding)))

    def test_all_idle_compacts_to_zero_rows_and_expands_zero_padding(self):
        plan = KtpBatchPlan(self._descriptors([0] * 8, bucket=2), rank=5)
        physical = torch.ones(16, 3)
        compact = plan.compact_valid_rows(physical)
        self.assertEqual(tuple(compact.shape), (0, 3))
        expanded = plan.expand_valid_rows(compact)
        self.assertEqual(tuple(expanded.shape), (16, 3))
        self.assertFalse(expanded.any())

    def test_activation_padding_all_gather_and_reduce_scatter_trim(self):
        plan = KtpBatchPlan(
            self._descriptors([1, 2, 3, 4, 1, 2, 3, 4]), rank=2
        )
        local = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        rank_rows = [
            torch.full((4, 2), float(rank), dtype=torch.float32)
            for rank in range(8)
        ]
        with patch(
            "rtp_llm.models.kimi_k3.decode_ktp.all_gather",
            return_value=torch.cat(rank_rows),
        ) as gather:
            global_rows = plan.all_gather_rows(local)
        submitted = gather.call_args.args[0]
        self.assertEqual(tuple(submitted.shape), (4, 2))
        self.assertTrue(torch.equal(submitted[:3], local))
        self.assertTrue(torch.equal(submitted[3], torch.zeros(2)))
        self.assertEqual(tuple(global_rows.shape), (32, 2))

        local_physical = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        with patch(
            "rtp_llm.models.kimi_k3.decode_ktp.reduce_scatter",
            return_value=local_physical,
        ):
            trimmed = plan.reduce_scatter_rows(torch.zeros(32, 2))
        self.assertTrue(torch.equal(trimmed, local_physical[:3]))

    def test_mismatched_step_epoch_fails_after_descriptor_collective(self):
        descriptors = list(self._descriptors([1] * 8, bucket=1))
        descriptors[6] = KtpStepDescriptor.build(
            rank=6,
            step_epoch=8,
            local_batch=1,
            bucket=1,
            request_ids=[600],
            generation_epochs=[3],
        )
        with self.assertRaisesRegex(ValueError, "different decode step epochs"):
            KtpBatchPlan(tuple(descriptors), rank=0)

    def test_fake_stream_contributes_only_padding(self):
        descriptor = KtpStepDescriptor.build(
            rank=4,
            step_epoch=9,
            local_batch=1,
            bucket=2,
            request_ids=[44],
            generation_epochs=[1],
            is_fake=True,
        )
        self.assertEqual(descriptor.local_batch, 0)
        self.assertEqual(descriptor.request_ids, (-1, -1))
        self.assertEqual(descriptor.valid_mask, (False, False))

    def test_kda_decode_output_uses_ktp_reduce_scatter(self):
        module = KimiK3KDA.__new__(KimiK3KDA)
        torch.nn.Module.__init__(module)
        module.weights = {
            W.linear_attn_norm_w: torch.ones(2),
            W.linear_attn_out_w: torch.eye(2),
        }
        module.eps = 1e-6
        module.projection_size = 2
        module.attn_tp_size = 8
        module.attn_tp_rank = 3
        module.collective_group = Group.KTP
        output = torch.ones(1, 8, 1, 2)
        output_gate = torch.ones(1, 8, 1, 2)
        expected = torch.full((1, 2), 7.0)
        hidden = SimpleNamespace(is_cuda=True)
        with patch(
            "rtp_llm.models_py.modules.kimi_k3.kda.module.reduce_scatter_padded",
            return_value=expected,
        ) as reduce_scatter, patch(
            "rtp_llm.models_py.modules.kimi_k3.kda.module.all_reduce"
        ) as all_reduce:
            actual = module._project_output(
                output,
                output_gate,
                is_target_verify=False,
                sequence_parallel=True,
                hidden_states=hidden,
                mode="decode",
            )
        self.assertIs(actual, expected)
        self.assertEqual(reduce_scatter.call_args.kwargs["group"], Group.KTP)
        all_reduce.assert_not_called()

    def test_kda_shadow_snapshot_selects_group_one_for_all_global_rows(self):
        plan = KtpBatchPlan(self._descriptors([1] * 8, bucket=1), rank=0)
        attention = PyAttentionInputs()
        attention.is_prefill = False
        attention.is_fake_stream = False
        attention.input_lengths = torch.ones(1, dtype=torch.int32)
        attention.sequence_lengths = torch.tensor([100], dtype=torch.int32)
        attention.kda_shadow_group_id = 1
        attention.kda_shadow_keys_host = torch.tensor(
            [[rank * 100, 3] for rank in range(8)], dtype=torch.int64
        )
        physical = torch.arange(8 * 3, dtype=torch.int32).reshape(8, 3)
        kernel = physical + 100
        attention.kda_shadow_block_ids_host = physical
        attention.kda_shadow_kernel_block_ids_host = kernel

        with patch.object(
            KtpBatchPlan,
            "all_gather_rows",
            side_effect=[
                torch.ones(8, dtype=torch.int32),
                torch.arange(100, 108, dtype=torch.int32),
            ],
        ):
            result = build_ktp_attention_inputs(
                attention, plan, device=torch.device("cpu")
            )

        self.assertEqual(tuple(result.sequence_lengths_plus_1_d.shape), (8,))
        self.assertEqual(len(result.kv_cache_kernel_block_id_device_by_group), 2)
        self.assertEqual(
            tuple(result.kv_cache_kernel_block_id_device_by_group[0].shape), (0, 0)
        )
        self.assertTrue(
            torch.equal(result.kv_cache_kernel_block_id_device_by_group[1], kernel),
            msg=(
                f"actual={result.kv_cache_kernel_block_id_device_by_group[1].tolist()} "
                f"expected={kernel.tolist()}"
            ),
        )
        self.assertEqual(select_block_map_for_layer(result, 0, 1), 1)
        self.assertTrue(torch.equal(result.kv_cache_kernel_block_id_device, kernel))

    def test_cuda_graph_metadata_captures_kda_table_all_gather(self):
        plan = KtpBatchPlan.for_cuda_graph(
            KdaParallelContext(8, 2, Group.KTP), fixed_bucket=4
        )
        attention = PyAttentionInputs()
        attention.is_prefill = False
        attention.input_lengths = torch.ones(4, dtype=torch.int32)
        attention.sequence_lengths = torch.arange(100, 104, dtype=torch.int32)
        local_table = torch.arange(12, dtype=torch.int32).reshape(4, 3)
        attention.kv_cache_kernel_block_id_device_by_group = [
            torch.zeros(4, 3, dtype=torch.int32),
            local_table,
        ]
        global_input = torch.ones(32, dtype=torch.int32)
        global_sequence = torch.arange(100, 132, dtype=torch.int32)
        global_table = torch.arange(96, dtype=torch.int32).reshape(32, 3)
        with patch.object(
            KtpBatchPlan,
            "all_gather_rows",
            side_effect=[global_input, global_sequence, global_table],
        ):
            result = build_ktp_cuda_graph_attention_inputs(
                attention,
                plan,
                device=torch.device("cpu"),
                kda_group_id=1,
            )

        self.assertEqual(tuple(result.sequence_lengths_plus_1_d.shape), (32,))
        self.assertEqual(result.total_tokens, 32)
        self.assertTrue(
            torch.equal(result.kv_cache_kernel_block_id_device_by_group[1], global_table)
        )
        self.assertTrue(torch.equal(result.kv_cache_kernel_block_id_device, global_table))


if __name__ == "__main__":
    unittest.main()
