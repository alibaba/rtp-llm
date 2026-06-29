import dataclasses
import unittest
from types import SimpleNamespace
from typing import NamedTuple

import torch

from rtp_llm.models_py.modules.dsv4.prefill_graph import StaticTensorMirror
from rtp_llm.models_py.modules.dsv4.prefill_graph import StaticPrefillGraphInputs
from rtp_llm.models_py.modules.dsv4.prefill_graph import StaticMetadataBuffers
from rtp_llm.models_py.modules.dsv4.prefill_graph import StaticPrefillMetaBuckets
from rtp_llm.models_py.modules.dsv4.prefill_graph import StaticPrefillGraphState
from rtp_llm.models_py.modules.dsv4.prefill_graph import StaticPrefillGraphStateManager
from rtp_llm.models_py.modules.dsv4.prefill_graph import (
    exact_static_prefill_layer_loop_args,
)
from rtp_llm.models_py.modules.dsv4.prefill_graph import PrefillGraphKey
from rtp_llm.models_py.modules.dsv4.prefill_graph import PrefillGraphDecision
from rtp_llm.models_py.modules.dsv4.prefill_graph import PrefillGraphRequest
from rtp_llm.models_py.modules.dsv4.prefill_graph import analyze_prefill_capture_surface
from rtp_llm.models_py.modules.dsv4.prefill_graph import select_prefill_graph_key
from rtp_llm.models_py.modules.dsv4.prefill_graph import (
    try_update_static_prefill_graph_state,
)
from rtp_llm.models_py.modules.dsv4.prefill_graph import with_static_state_invariants


class TinyMeta(NamedTuple):
    block_table: torch.Tensor
    positions: torch.Tensor
    scale: int


class NestedMeta(NamedTuple):
    tiny: TinyMeta
    enabled: bool
    optional: object


class MetaWithWorkspace(NamedTuple):
    workspace: object
    tensor: torch.Tensor


class MetaWithOpaque(NamedTuple):
    opaque: object
    tensor: torch.Tensor


class MetaWithReader(NamedTuple):
    cmp_reader: object
    tensor: torch.Tensor


@dataclasses.dataclass
class TinyCPContext:
    cp_info: object
    positions: torch.Tensor


class MetaWithCPContext(NamedTuple):
    cp_ctx: TinyCPContext
    tensor: torch.Tensor


@dataclasses.dataclass
class TinyContext:
    lengths: torch.Tensor
    tag: str


class StaticTensorMirrorTest(unittest.TestCase):
    def test_prefill_graph_key_selects_smallest_token_bucket(self):
        decision = select_prefill_graph_key(
            PrefillGraphRequest(token_count=512, batch_size=1, cp_size=8),
            enabled=True,
            token_buckets=(512, 4096, 8192),
        )

        self.assertTrue(decision.enabled)
        self.assertEqual(
            decision.key,
            PrefillGraphKey(token_bucket=512, batch_bucket=1, cp_size=8),
        )
        self.assertEqual(decision.reason, "ok")

        decision = select_prefill_graph_key(
            PrefillGraphRequest(token_count=513, batch_size=1, cp_size=8),
            enabled=True,
            token_buckets="512,4096,8192",
        )

        self.assertTrue(decision.enabled)
        self.assertEqual(decision.key.token_bucket, 4096)

    def test_prefill_graph_key_falls_back_when_disabled_or_overflow(self):
        disabled = select_prefill_graph_key(
            PrefillGraphRequest(token_count=512, batch_size=1, cp_size=8),
            enabled=False,
            token_buckets=(512,),
        )
        self.assertFalse(disabled.enabled)
        self.assertEqual(disabled.reason, "disabled")

        overflow = select_prefill_graph_key(
            PrefillGraphRequest(token_count=513, batch_size=1, cp_size=8),
            enabled=True,
            token_buckets=(512,),
        )
        self.assertFalse(overflow.enabled)
        self.assertEqual(overflow.reason, "token_overflow")

    def test_prefill_graph_key_rejects_unsupported_modes(self):
        base = dict(token_count=512, batch_size=1, cp_size=8)
        cases = [
            (dict(prepare_hidden=True), "prepare_hidden"),
            (dict(cache_store=True), "cache_store"),
            (dict(mtp_hidden=True), "mtp_hidden"),
            (dict(prefix_length=1), "prefix_reuse"),
            (dict(max_prefix_length=1), "prefix_reuse"),
            (dict(cp_size=4), "cp_size"),
            (dict(batch_size=2), "batch_overflow"),
        ]
        for overrides, reason in cases:
            with self.subTest(reason=reason, overrides=overrides):
                request = PrefillGraphRequest(**{**base, **overrides})
                decision = select_prefill_graph_key(
                    request,
                    enabled=True,
                    token_buckets=(512, 4096),
                    batch_buckets=(1,),
                    fixed_cp_size=8,
                    allow_prefix_reuse=False,
                )
                self.assertFalse(decision.enabled)
                self.assertEqual(decision.reason, reason)

    def test_prefill_graph_key_can_bucket_batch_and_prefix_when_enabled(self):
        decision = select_prefill_graph_key(
            PrefillGraphRequest(
                token_count=400,
                batch_size=2,
                cp_size=8,
                prefix_length=32,
                max_prefix_length=32,
            ),
            enabled=True,
            token_buckets=(512,),
            batch_buckets=(1, 4),
            prefix_buckets=(0, 64),
            reuse_buckets=(0, 64),
            fixed_cp_size=8,
            allow_prefix_reuse=True,
        )

        self.assertTrue(decision.enabled)
        self.assertEqual(decision.key.batch_bucket, 4)
        self.assertEqual(decision.key.token_bucket, 512)
        self.assertEqual(decision.key.prefix_bucket, 64)
        self.assertEqual(decision.key.reuse_bucket, 64)

    def test_prefill_graph_key_rejects_prefix_overflow_when_allowed(self):
        decision = select_prefill_graph_key(
            PrefillGraphRequest(
                token_count=400,
                batch_size=1,
                cp_size=8,
                prefix_length=65,
                max_prefix_length=65,
            ),
            enabled=True,
            token_buckets=(512,),
            prefix_buckets=(0, 64),
            reuse_buckets=(0, 64),
            allow_prefix_reuse=True,
        )

        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "prefix_overflow")

    def test_prefill_graph_key_rejects_unknown_prefix_without_sync(self):
        decision = select_prefill_graph_key(
            PrefillGraphRequest(
                token_count=400,
                batch_size=1,
                cp_size=8,
                prefix_unknown=True,
            ),
            enabled=True,
            token_buckets=(512,),
        )

        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "prefix_unknown")

    def test_prefill_graph_key_malformed_buckets_fail_closed(self):
        for kwargs in (
            {"token_buckets": "512,bad"},
            {"token_buckets": "-1"},
            {"token_buckets": (512,), "batch_buckets": "0"},
            {"token_buckets": (512,), "prefix_buckets": "-1"},
        ):
            with self.subTest(kwargs=kwargs):
                decision = select_prefill_graph_key(
                    PrefillGraphRequest(token_count=1, batch_size=1, cp_size=8),
                    enabled=True,
                    **kwargs,
                )
                self.assertFalse(decision.enabled)
                self.assertEqual(decision.reason, "invalid_buckets")

    def test_same_shape_update_reuses_tensor_addresses(self):
        mirror = StaticTensorMirror()
        first = TinyMeta(
            block_table=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            positions=torch.tensor([0, 1, 2], dtype=torch.int64),
            scale=7,
        )
        static_first = mirror.update(first)
        block_ptr = static_first.block_table.data_ptr()
        pos_ptr = static_first.positions.data_ptr()

        second = TinyMeta(
            block_table=torch.tensor([[9, 8], [7, 6]], dtype=torch.int32),
            positions=torch.tensor([5, 6, 7], dtype=torch.int64),
            scale=7,
        )
        static_second = mirror.update(second)

        self.assertIs(static_first, static_second)
        self.assertEqual(static_second.block_table.data_ptr(), block_ptr)
        self.assertEqual(static_second.positions.data_ptr(), pos_ptr)
        self.assertTrue(torch.equal(static_second.block_table, second.block_table))
        self.assertTrue(torch.equal(static_second.positions, second.positions))

    def test_shape_change_rebuilds_static_tensors(self):
        mirror = StaticTensorMirror()
        first = {"bt": torch.ones((1, 2), dtype=torch.int32)}
        static_first = mirror.update(first)
        first_ptr = static_first["bt"].data_ptr()

        second = {"bt": torch.ones((2, 2), dtype=torch.int32)}
        static_second = mirror.update(second)

        self.assertIsNot(static_first, static_second)
        self.assertNotEqual(static_second["bt"].data_ptr(), first_ptr)
        self.assertEqual(tuple(static_second["bt"].shape), (2, 2))

    def test_mapping_same_shape_update_copies_values(self):
        mirror = StaticTensorMirror()
        first = {"bt": torch.tensor([[1, 2]], dtype=torch.int32)}
        static_first = mirror.update(first)
        ptr = static_first["bt"].data_ptr()

        second = {"bt": torch.tensor([[7, 8]], dtype=torch.int32)}
        static_second = mirror.update(second)

        self.assertIs(static_first, static_second)
        self.assertEqual(static_second["bt"].data_ptr(), ptr)
        self.assertTrue(torch.equal(static_second["bt"], second["bt"]))

    def test_scalar_change_rebuilds_metadata_object(self):
        mirror = StaticTensorMirror()
        first = TinyMeta(torch.ones(2), torch.arange(2), scale=1)
        static_first = mirror.update(first)
        second = TinyMeta(torch.zeros(2), torch.arange(2), scale=2)
        static_second = mirror.update(second)

        self.assertIsNot(static_first, static_second)
        self.assertEqual(static_second.scale, 2)

    def test_dataclass_fields_are_mirrored(self):
        mirror = StaticTensorMirror()
        first = TinyContext(torch.tensor([1, 2], dtype=torch.int32), "cp8")
        static_first = mirror.update(first)
        ptr = static_first.lengths.data_ptr()
        second = TinyContext(torch.tensor([8, 9], dtype=torch.int32), "cp8")
        static_second = mirror.update(second)

        self.assertIs(static_first, static_second)
        self.assertEqual(static_second.lengths.data_ptr(), ptr)
        self.assertTrue(torch.equal(static_second.lengths, second.lengths))

    def test_opaque_objects_rejected_by_default(self):
        mirror = StaticTensorMirror()

        class Opaque:
            pass

        with self.assertRaises(TypeError):
            mirror.update({"opaque": Opaque()})

    def test_opaque_objects_allowed_for_debug_identity_mode(self):
        mirror = StaticTensorMirror(allow_opaque_objects=True)

        class Opaque:
            pass

        opaque = Opaque()
        static = mirror.update({"opaque": opaque})
        self.assertIs(static["opaque"], opaque)
        self.assertIs(mirror.update({"opaque": opaque}), static)

    def test_static_prefill_graph_inputs_reuses_addresses_and_updates_values(self):
        buffers = StaticPrefillGraphInputs(
            token_cap=4,
            batch_cap=2,
            block_cap=3,
            device="cpu",
            block_table_keys=[0, 4],
        )
        pos_ptr = buffers.position_ids.data_ptr()
        bt_ptr = buffers.block_tables_by_type[4].data_ptr()

        buffers.update(
            position_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
            req_id_per_token=torch.tensor([0, 0, 1], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 2, 3], dtype=torch.int64),
            input_lengths=torch.tensor([2, 1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0, 5], dtype=torch.int32),
            block_tables_by_type={4: torch.tensor([[9, 8], [7, 6]], dtype=torch.int64)},
            seq_len_full=3,
            prefix_length=0,
        )
        buffers.update(
            position_ids=torch.tensor([5, 6, 7], dtype=torch.int64),
            req_id_per_token=torch.tensor([0, 1, 1], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1, 3], dtype=torch.int64),
            input_lengths=torch.tensor([1, 2], dtype=torch.int32),
            prefix_lengths=torch.tensor([1, 2], dtype=torch.int32),
            block_tables_by_type={4: torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)},
            seq_len_full=3,
            prefix_length=1,
        )

        self.assertEqual(buffers.position_ids.data_ptr(), pos_ptr)
        self.assertEqual(buffers.block_tables_by_type[4].data_ptr(), bt_ptr)
        self.assertTrue(torch.equal(buffers.position_ids[:3], torch.tensor([5, 6, 7])))
        self.assertTrue(
            torch.equal(
                buffers.block_tables_by_type[4],
                torch.tensor([[1, 2, 0], [3, 4, 0]], dtype=torch.int32),
            )
        )
        self.assertEqual(int(buffers.scalar_i64[0]), 3)
        self.assertEqual(int(buffers.scalar_i64[3]), 1)

    def test_static_prefill_graph_inputs_rejects_over_capacity(self):
        buffers = StaticPrefillGraphInputs(
            token_cap=2,
            batch_cap=1,
            block_cap=2,
            device="cpu",
            block_table_keys=[0],
        )

        with self.assertRaises(ValueError):
            buffers.update(
                position_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
                req_id_per_token=torch.tensor([0, 0, 0], dtype=torch.int32),
                cu_seqlens=torch.tensor([0, 3], dtype=torch.int64),
                input_lengths=torch.tensor([3], dtype=torch.int32),
                prefix_lengths=torch.tensor([0], dtype=torch.int32),
                block_tables_by_type={},
                seq_len_full=3,
                prefix_length=0,
            )

    def test_static_prefill_graph_inputs_rejects_mismatched_prefix_lengths(self):
        buffers = StaticPrefillGraphInputs(
            token_cap=4,
            batch_cap=2,
            block_cap=2,
            device="cpu",
            block_table_keys=[0],
        )

        with self.assertRaises(ValueError):
            buffers.update(
                position_ids=torch.tensor([0, 1], dtype=torch.int64),
                req_id_per_token=torch.tensor([0, 0], dtype=torch.int32),
                cu_seqlens=torch.tensor([0, 2], dtype=torch.int64),
                input_lengths=torch.tensor([2], dtype=torch.int32),
                prefix_lengths=torch.tensor([0, 1], dtype=torch.int32),
                block_tables_by_type={},
                seq_len_full=2,
                prefix_length=0,
            )

    def test_static_prefill_graph_inputs_rejects_inconsistent_lengths(self):
        buffers = StaticPrefillGraphInputs(
            token_cap=4,
            batch_cap=2,
            block_cap=2,
            device="cpu",
            block_table_keys=[0],
        )

        with self.assertRaisesRegex(ValueError, "cu_seqlens"):
            buffers.update(
                position_ids=torch.tensor([0, 1], dtype=torch.int64),
                req_id_per_token=torch.tensor([0, 0], dtype=torch.int32),
                cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
                input_lengths=torch.tensor([2], dtype=torch.int32),
                prefix_lengths=torch.tensor([0], dtype=torch.int32),
                block_tables_by_type={},
                seq_len_full=2,
                prefix_length=0,
            )

        with self.assertRaisesRegex(ValueError, "input_lengths"):
            buffers.update(
                position_ids=torch.tensor([0, 1], dtype=torch.int64),
                req_id_per_token=torch.tensor([0, 0], dtype=torch.int32),
                cu_seqlens=torch.tensor([0, 2], dtype=torch.int64),
                input_lengths=torch.tensor([1], dtype=torch.int32),
                prefix_lengths=torch.tensor([0], dtype=torch.int32),
                block_tables_by_type={},
                seq_len_full=2,
                prefix_length=0,
            )

        with self.assertRaisesRegex(ValueError, "seq_len_full"):
            buffers.update(
                position_ids=torch.tensor([0, 1], dtype=torch.int64),
                req_id_per_token=torch.tensor([0, 0], dtype=torch.int32),
                cu_seqlens=torch.tensor([0, 2], dtype=torch.int64),
                input_lengths=torch.tensor([2], dtype=torch.int32),
                prefix_lengths=torch.tensor([0], dtype=torch.int32),
                block_tables_by_type={},
                seq_len_full=1,
                prefix_length=0,
            )

    def test_static_prefill_graph_inputs_rejects_bad_boundaries_and_req_ids(self):
        buffers = StaticPrefillGraphInputs(
            token_cap=4,
            batch_cap=2,
            block_cap=2,
            device="cpu",
            block_table_keys=[0],
        )
        base = dict(
            position_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
            req_id_per_token=torch.tensor([0, 0, 1], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 2, 3], dtype=torch.int64),
            input_lengths=torch.tensor([2, 1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0, 0], dtype=torch.int32),
            block_tables_by_type={},
            seq_len_full=3,
            prefix_length=0,
        )

        bad_start = {**base, "cu_seqlens": torch.tensor([1, 2, 3])}
        with self.assertRaisesRegex(ValueError, "cu_seqlens\\[0\\]"):
            buffers.update(**bad_start)

        bad_monotonic = {**base, "cu_seqlens": torch.tensor([0, 3, 2])}
        with self.assertRaisesRegex(ValueError, "monotonic"):
            buffers.update(**bad_monotonic)

        bad_diff = {
            **base,
            "cu_seqlens": torch.tensor([0, 1, 3]),
            "input_lengths": torch.tensor([2, 1], dtype=torch.int32),
        }
        with self.assertRaisesRegex(ValueError, "diffs"):
            buffers.update(**bad_diff)

        bad_range = {
            **base,
            "req_id_per_token": torch.tensor([0, 0, 2], dtype=torch.int32),
        }
        with self.assertRaisesRegex(ValueError, "batch range"):
            buffers.update(**bad_range)

        bad_boundary = {
            **base,
            "req_id_per_token": torch.tensor([0, 1, 1], dtype=torch.int32),
        }
        with self.assertRaisesRegex(ValueError, "boundaries"):
            buffers.update(**bad_boundary)

    def test_static_prefill_graph_inputs_rejects_unknown_block_table_key(self):
        buffers = StaticPrefillGraphInputs(
            token_cap=4,
            batch_cap=1,
            block_cap=2,
            device="cpu",
            block_table_keys=[0],
        )

        with self.assertRaises(ValueError):
            buffers.update(
                position_ids=torch.tensor([0], dtype=torch.int64),
                req_id_per_token=torch.tensor([0], dtype=torch.int32),
                cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
                input_lengths=torch.tensor([1], dtype=torch.int32),
                prefix_lengths=torch.tensor([0], dtype=torch.int32),
                block_tables_by_type={4: torch.tensor([[1]], dtype=torch.int32)},
                seq_len_full=1,
                prefix_length=0,
            )

    def test_static_prefill_graph_inputs_rejects_block_table_row_mismatch(self):
        buffers = StaticPrefillGraphInputs(
            token_cap=4,
            batch_cap=2,
            block_cap=2,
            device="cpu",
            block_table_keys=[0],
        )

        with self.assertRaisesRegex(ValueError, "must match batch_size"):
            buffers.update(
                position_ids=torch.tensor([0, 1], dtype=torch.int64),
                req_id_per_token=torch.tensor([0, 0], dtype=torch.int32),
                cu_seqlens=torch.tensor([0, 2], dtype=torch.int64),
                input_lengths=torch.tensor([2], dtype=torch.int32),
                prefix_lengths=torch.tensor([0], dtype=torch.int32),
                block_tables_by_type={0: torch.tensor([[1], [2]], dtype=torch.int32)},
                seq_len_full=2,
                prefix_length=0,
            )

    def test_static_metadata_buffers_update_tensor_and_scalar_leaves(self):
        buffers = StaticMetadataBuffers(device="cpu")
        first = NestedMeta(
            tiny=TinyMeta(
                block_table=torch.tensor([[1, 2]], dtype=torch.int32),
                positions=torch.tensor([0, 1], dtype=torch.int64),
                scale=3,
            ),
            enabled=True,
            optional=None,
        )
        buffers.update(first)
        bt_ptr = buffers.tensors["tiny.block_table"].data_ptr()
        pos_ptr = buffers.tensors["tiny.positions"].data_ptr()
        scale_ptr = buffers.scalar_i64["tiny.scale"].data_ptr()

        second = NestedMeta(
            tiny=TinyMeta(
                block_table=torch.tensor([[7, 8]], dtype=torch.int32),
                positions=torch.tensor([5, 6], dtype=torch.int64),
                scale=3,
            ),
            enabled=True,
            optional=None,
        )
        buffers.update(second)

        self.assertEqual(buffers.tensors["tiny.block_table"].data_ptr(), bt_ptr)
        self.assertEqual(buffers.tensors["tiny.positions"].data_ptr(), pos_ptr)
        self.assertEqual(buffers.scalar_i64["tiny.scale"].data_ptr(), scale_ptr)
        self.assertTrue(
            torch.equal(
                buffers.tensors["tiny.block_table"],
                torch.tensor([[7, 8]], dtype=torch.int32),
            )
        )
        self.assertEqual(int(buffers.scalar_i64["tiny.scale"]), 3)
        self.assertEqual(int(buffers.scalar_i64["enabled"]), 1)

    def test_static_metadata_buffers_rejects_scalar_value_change(self):
        buffers = StaticMetadataBuffers(device="cpu")
        first = TinyMeta(torch.ones(1), torch.arange(1), 1)
        buffers.update(first)
        with self.assertRaisesRegex(ValueError, "scalar values"):
            buffers.update(TinyMeta(torch.ones(1), torch.arange(1), 2))

    def test_static_metadata_buffers_rejects_shape_change(self):
        buffers = StaticMetadataBuffers(device="cpu")
        buffers.update(TinyMeta(torch.ones((1, 2)), torch.arange(2), 1))
        with self.assertRaises(ValueError):
            buffers.update(TinyMeta(torch.ones((2, 2)), torch.arange(2), 1))

    def test_static_metadata_buffers_rejects_opaque_object(self):
        buffers = StaticMetadataBuffers(device="cpu")

        class Opaque:
            pass

        with self.assertRaises(TypeError):
            buffers.update(
                NestedMeta(TinyMeta(torch.ones(1), torch.arange(1), 1), True, Opaque())
            )

    def test_static_metadata_buffers_rejects_replay_varying_float_or_string(self):
        buffers = StaticMetadataBuffers(device="cpu")

        with self.assertRaisesRegex(TypeError, "float"):
            buffers.update({"scale": 1.25})
        with self.assertRaisesRegex(TypeError, "str"):
            buffers.update({"tag": "dynamic"})

    def test_static_prefill_meta_buckets_update_ratio_metadata(self):
        buckets = StaticPrefillMetaBuckets(device="cpu")
        first = {
            0: NestedMeta(
                tiny=TinyMeta(torch.tensor([1, 2]), torch.tensor([0, 1]), 3),
                enabled=True,
                optional=None,
            )
        }
        buckets.update(first)
        tensor_ptr = buckets.by_ratio[0].tensors["tiny.block_table"].data_ptr()
        scalar_ptr = buckets.by_ratio[0].scalar_i64["tiny.scale"].data_ptr()

        second = {
            0: NestedMeta(
                tiny=TinyMeta(torch.tensor([7, 8]), torch.tensor([5, 6]), 3),
                enabled=True,
                optional=None,
            )
        }
        buckets.update(second)

        self.assertEqual(
            buckets.by_ratio[0].tensors["tiny.block_table"].data_ptr(), tensor_ptr
        )
        self.assertEqual(
            buckets.by_ratio[0].scalar_i64["tiny.scale"].data_ptr(), scalar_ptr
        )
        self.assertTrue(
            torch.equal(
                buckets.by_ratio[0].tensors["tiny.block_table"], torch.tensor([7, 8])
            )
        )
        self.assertEqual(int(buckets.by_ratio[0].scalar_i64["tiny.scale"]), 3)

    def test_static_prefill_meta_buckets_materializes_static_tensor_leaves(self):
        buckets = StaticPrefillMetaBuckets(device="cpu")
        first = {
            4: NestedMeta(
                tiny=TinyMeta(torch.tensor([1, 2]), torch.tensor([0, 1]), 7),
                enabled=True,
                optional=None,
            )
        }
        buckets.update(first)
        materialized_first = buckets.materialize(first)
        block_ptr = materialized_first[4].tiny.block_table.data_ptr()
        pos_ptr = materialized_first[4].tiny.positions.data_ptr()

        second = {
            4: NestedMeta(
                tiny=TinyMeta(torch.tensor([9, 8]), torch.tensor([5, 6]), 7),
                enabled=True,
                optional=None,
            )
        }
        buckets.update(second)
        materialized_second = buckets.materialize(second)

        self.assertEqual(materialized_second[4].tiny.block_table.data_ptr(), block_ptr)
        self.assertEqual(materialized_second[4].tiny.positions.data_ptr(), pos_ptr)
        self.assertTrue(
            torch.equal(materialized_second[4].tiny.block_table, torch.tensor([9, 8]))
        )
        self.assertTrue(
            torch.equal(materialized_second[4].tiny.positions, torch.tensor([5, 6]))
        )

    def test_static_prefill_meta_materialize_preserves_skipped_owner(self):
        workspace = object()
        meta = {128: MetaWithWorkspace(workspace=workspace, tensor=torch.tensor([1]))}
        buckets = StaticPrefillMetaBuckets(device="cpu")
        buckets.update(meta)

        materialized = buckets.materialize(meta)

        self.assertIs(materialized[128].workspace, workspace)
        self.assertIsNot(materialized[128].tensor, meta[128].tensor)
        self.assertTrue(torch.equal(materialized[128].tensor, meta[128].tensor))

    def test_static_workspace_materialization_reaches_static_bound(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=2, batch_bucket=1, cp_size=8),
            local_token_bucket=2,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=1,
            block_table_keys=(4,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=1,
            block_table_keys=(4,),
        )
        meta = {4: MetaWithWorkspace(workspace=object(), tensor=torch.tensor([1]))}
        state.update(
            input_ids=torch.tensor([1, 2], dtype=torch.int64),
            hidden=torch.ones((2, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0, 1], dtype=torch.int64),
            req_id_per_token=torch.tensor([0, 0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int64),
            input_lengths=torch.tensor([2], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={4: torch.tensor([[1]], dtype=torch.int32)},
            seq_len_full=2,
            prefix_length=0,
            meta_by_ratio=meta,
            workspace_config=dict(
                q_rows=2,
                q_dim=3,
                reserve_cp=True,
                cp_rows=2,
                main_w=2,
                idx_w=2,
                swa_w=2,
                align_bytes=1,
            ),
        )

        materialized = state.meta.materialize(meta, workspace=state.workspace)
        report = analyze_prefill_capture_surface(
            state,
            input_ids=state.input_ids,
            hidden=state.hidden,
            position_ids=state.request.position_ids,
            req_id_per_token=state.request.req_id_per_token,
            cu_seqlens=state.request.cu_seqlens,
            input_lengths=state.request.input_lengths,
            prefix_lengths=state.request.prefix_lengths,
            block_tables_by_type=state.request.block_tables_by_type,
            meta_by_ratio=materialized,
        )

        self.assertIs(materialized[4].workspace, state.workspace)
        self.assertTrue(report.static_bound)
        self.assertEqual(report.live_not_static, ())
        self.assertEqual(report.missing_static, ())
        self.assertEqual(report.skipped_critical, ())

    def test_static_prefill_meta_materialize_drops_cp_info_owner(self):
        cp_info = object()
        meta = {
            4: MetaWithCPContext(
                cp_ctx=TinyCPContext(cp_info=cp_info, positions=torch.tensor([0, 1])),
                tensor=torch.tensor([1]),
            )
        }
        buckets = StaticPrefillMetaBuckets(device="cpu")
        buckets.update(meta)

        materialized = buckets.materialize(meta)

        self.assertIsNone(materialized[4].cp_ctx.cp_info)
        self.assertIsNot(materialized[4].cp_ctx.positions, meta[4].cp_ctx.positions)
        self.assertTrue(
            torch.equal(materialized[4].cp_ctx.positions, meta[4].cp_ctx.positions)
        )

    def test_static_prefill_meta_materialize_drops_stateless_local_reader(self):
        class LocalPoolReader:
            pass

        reader = LocalPoolReader()
        meta = {4: MetaWithReader(cmp_reader=reader, tensor=torch.tensor([1]))}
        buckets = StaticPrefillMetaBuckets(device="cpu")
        buckets.update(meta)

        materialized = buckets.materialize(meta)

        self.assertIsNone(materialized[4].cmp_reader)
        self.assertIsNot(materialized[4].tensor, meta[4].tensor)
        self.assertTrue(torch.equal(materialized[4].tensor, meta[4].tensor))

    def test_static_prefill_meta_materialize_preserves_stateful_reader(self):
        class CPShardedPoolReader:
            pass

        reader = CPShardedPoolReader()
        meta = {4: MetaWithReader(cmp_reader=reader, tensor=torch.tensor([1]))}
        buckets = StaticPrefillMetaBuckets(device="cpu")
        buckets.update(meta)

        materialized = buckets.materialize(meta)

        self.assertIs(materialized[4].cmp_reader, reader)
        self.assertIsNot(materialized[4].tensor, meta[4].tensor)

    def test_static_prefill_meta_buckets_reject_ratio_set_change(self):
        buckets = StaticPrefillMetaBuckets(device="cpu")
        buckets.update({0: TinyMeta(torch.ones(1), torch.arange(1), 1)})
        with self.assertRaises(ValueError):
            buckets.update(
                {
                    0: TinyMeta(torch.ones(1), torch.arange(1), 1),
                    4: TinyMeta(torch.ones(1), torch.arange(1), 1),
                }
            )

    def test_static_prefill_graph_state_updates_same_buffers_for_two_requests(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=2, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=3,
            block_table_keys=(0, 4),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=3,
            block_table_keys=(0, 4),
        )
        ptrs = {
            "input_ids": state.input_ids.data_ptr(),
            "hidden": state.hidden.data_ptr(),
            "output_hidden": state.output_hidden.data_ptr(),
            "positions": state.request.position_ids.data_ptr(),
            "block4": state.request.block_tables_by_type[4].data_ptr(),
        }

        state.update(
            input_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
            hidden=torch.ones((3, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
            req_id_per_token=torch.tensor([0, 0, 1], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 2, 3], dtype=torch.int64),
            input_lengths=torch.tensor([2, 1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0, 0], dtype=torch.int32),
            block_tables_by_type={4: torch.tensor([[9, 8], [7, 6]], dtype=torch.int32)},
            seq_len_full=3,
            prefix_length=0,
            meta_by_ratio={
                0: TinyMeta(
                    torch.tensor([[1, 2]], dtype=torch.int32),
                    torch.tensor([0, 1], dtype=torch.int64),
                    3,
                )
            },
        )
        first_signature = state.pointer_signature()
        first_inventory_names = {item.name for item in state.pointer_inventory()}
        self.assertTrue(state.pointer_stable)
        self.assertIn("input_ids", first_inventory_names)
        self.assertIn("hidden", first_inventory_names)
        self.assertIn("request.block_tables_by_type.4", first_inventory_names)
        self.assertIn("meta.0.tensor.block_table", first_inventory_names)
        meta_ptr = state.meta.by_ratio[0].tensors["block_table"].data_ptr()

        state.update(
            input_ids=torch.tensor([4, 5, 6], dtype=torch.int64),
            hidden=torch.full((3, 2, 3), 2.0, dtype=torch.bfloat16),
            position_ids=torch.tensor([5, 6, 7], dtype=torch.int64),
            req_id_per_token=torch.tensor([0, 1, 1], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1, 3], dtype=torch.int64),
            input_lengths=torch.tensor([1, 2], dtype=torch.int32),
            prefix_lengths=torch.tensor([1, 2], dtype=torch.int32),
            block_tables_by_type={4: torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)},
            seq_len_full=3,
            prefix_length=2,
            meta_by_ratio={
                0: TinyMeta(
                    torch.tensor([[7, 8]], dtype=torch.int32),
                    torch.tensor([5, 6], dtype=torch.int64),
                    3,
                )
            },
        )

        self.assertEqual(state.input_ids.data_ptr(), ptrs["input_ids"])
        self.assertEqual(state.hidden.data_ptr(), ptrs["hidden"])
        self.assertEqual(state.output_hidden.data_ptr(), ptrs["output_hidden"])
        self.assertEqual(state.request.position_ids.data_ptr(), ptrs["positions"])
        self.assertEqual(
            state.request.block_tables_by_type[4].data_ptr(), ptrs["block4"]
        )
        self.assertEqual(
            state.meta.by_ratio[0].tensors["block_table"].data_ptr(), meta_ptr
        )
        self.assertTrue(torch.equal(state.input_ids[:3], torch.tensor([4, 5, 6])))
        self.assertTrue(
            torch.equal(state.request.position_ids[:3], torch.tensor([5, 6, 7]))
        )
        self.assertTrue(
            torch.equal(
                state.request.block_tables_by_type[4],
                torch.tensor([[1, 2, 0], [3, 4, 0]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                state.meta.by_ratio[0].tensors["block_table"],
                torch.tensor([[7, 8]], dtype=torch.int32),
            )
        )
        self.assertEqual(int(state.request.scalar_i64[3]), 2)
        self.assertEqual(int(state.meta.by_ratio[0].scalar_i64["scale"]), 3)
        self.assertTrue(state.valid)
        self.assertTrue(state.pointer_stable)
        self.assertEqual(state.pointer_signature(), first_signature)

    def test_static_prefill_graph_state_tracks_cuda_graph_lifetime(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=1, batch_bucket=1, cp_size=8),
            local_token_bucket=1,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=1,
            block_table_keys=(),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=1,
            block_table_keys=(),
        )
        state.update(
            input_ids=torch.tensor([1], dtype=torch.int64),
            hidden=torch.ones((1, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0], dtype=torch.int64),
            req_id_per_token=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
            input_lengths=torch.tensor([1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={},
            seq_len_full=1,
            prefix_length=0,
        )

        graph = object()
        state.mark_cuda_graph_captured(graph)

        self.assertTrue(state.cuda_graph_ready)
        self.assertIs(state.cuda_graph, graph)
        self.assertEqual(state.graph_capture_count, 1)
        self.assertIsNone(state.graph_capture_error)

        state.reset_cuda_graph("recapture")

        self.assertFalse(state.cuda_graph_ready)
        self.assertIsNone(state.cuda_graph)
        self.assertEqual(state.graph_capture_error, "recapture")
        self.assertEqual(state.graph_capture_count, 0)
        self.assertEqual(state.graph_replay_count, 0)

    def test_static_prefill_graph_state_detects_pointer_inventory_drift(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        common = dict(
            input_ids=torch.tensor([1], dtype=torch.int64),
            hidden=torch.ones((1, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0], dtype=torch.int64),
            req_id_per_token=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
            input_lengths=torch.tensor([1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={},
            seq_len_full=1,
            prefix_length=0,
        )
        state.update(**common)
        self.assertTrue(state.pointer_stable)
        no_meta_signature = state.pointer_signature()

        state.update(
            **common,
            meta_by_ratio={0: TinyMeta(torch.ones(1), torch.arange(1), 1)},
        )

        self.assertFalse(state.valid)
        self.assertFalse(state.pointer_stable)
        self.assertNotEqual(state.pointer_signature(), no_meta_signature)

    def test_capture_surface_report_detects_live_non_static_tensors(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        live = dict(
            input_ids=torch.tensor([1], dtype=torch.int64),
            hidden=torch.ones((1, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0], dtype=torch.int64),
            req_id_per_token=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
            input_lengths=torch.tensor([1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={0: torch.tensor([[1]], dtype=torch.int32)},
            meta_by_ratio=None,
        )
        state.update(
            **live,
            seq_len_full=1,
            prefix_length=0,
        )

        report = analyze_prefill_capture_surface(state, **live)

        self.assertFalse(report.static_bound)
        self.assertIn("input_ids", report.live_not_static)
        self.assertIn("hidden", report.live_not_static)
        self.assertEqual(report.missing_static, ())
        self.assertEqual(report.skipped_critical, ())

    def test_capture_surface_report_accepts_static_tensors(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state.update(
            input_ids=torch.tensor([1], dtype=torch.int64),
            hidden=torch.ones((1, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0], dtype=torch.int64),
            req_id_per_token=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
            input_lengths=torch.tensor([1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={0: torch.tensor([[1]], dtype=torch.int32)},
            seq_len_full=1,
            prefix_length=0,
        )

        report = analyze_prefill_capture_surface(
            state,
            input_ids=state.input_ids,
            hidden=state.hidden,
            position_ids=state.request.position_ids,
            req_id_per_token=state.request.req_id_per_token,
            cu_seqlens=state.request.cu_seqlens,
            input_lengths=state.request.input_lengths,
            prefix_lengths=state.request.prefix_lengths,
            block_tables_by_type=state.request.block_tables_by_type,
            meta_by_ratio=None,
        )

        self.assertTrue(report.static_bound)
        self.assertEqual(report.live_tensor_count, report.static_bound_count)
        self.assertEqual(report.live_not_static, ())
        self.assertEqual(report.missing_static, ())
        self.assertEqual(report.skipped_critical, ())

    def test_capture_surface_report_rejects_shape_stride_alias(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state.update(
            input_ids=torch.tensor([1], dtype=torch.int64),
            hidden=torch.ones((1, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0], dtype=torch.int64),
            req_id_per_token=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
            input_lengths=torch.tensor([1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={0: torch.tensor([[1]], dtype=torch.int32)},
            seq_len_full=1,
            prefix_length=0,
        )

        report = analyze_prefill_capture_surface(
            state,
            input_ids=state.input_ids[:1],
            hidden=state.hidden[:1],
            position_ids=state.request.position_ids[:1],
            req_id_per_token=state.request.req_id_per_token[:1],
            cu_seqlens=state.request.cu_seqlens[:2],
            input_lengths=state.request.input_lengths[:1],
            prefix_lengths=state.request.prefix_lengths[:1],
            block_tables_by_type={0: state.request.block_tables_by_type[0][:1]},
            meta_by_ratio=None,
        )

        self.assertFalse(report.static_bound)
        self.assertIn("input_ids", report.live_not_static)
        self.assertIn("hidden", report.live_not_static)

    def test_capture_surface_report_tracks_skipped_critical_owners(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state.update(
            input_ids=torch.tensor([1], dtype=torch.int64),
            hidden=torch.ones((1, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0], dtype=torch.int64),
            req_id_per_token=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
            input_lengths=torch.tensor([1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={0: torch.tensor([[1]], dtype=torch.int32)},
            seq_len_full=1,
            prefix_length=0,
            meta_by_ratio={0: MetaWithWorkspace(object(), torch.ones(1))},
        )

        report = analyze_prefill_capture_surface(
            state,
            input_ids=state.input_ids,
            hidden=state.hidden,
            position_ids=state.request.position_ids,
            req_id_per_token=state.request.req_id_per_token,
            cu_seqlens=state.request.cu_seqlens,
            input_lengths=state.request.input_lengths,
            prefix_lengths=state.request.prefix_lengths,
            block_tables_by_type=state.request.block_tables_by_type,
            meta_by_ratio={0: MetaWithWorkspace(object(), torch.ones(1))},
        )

        self.assertFalse(report.static_bound)
        self.assertIn("meta.0.tensor.workspace", report.skipped_critical)

    def test_capture_surface_report_tracks_unknown_opaque_objects(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state.update(
            input_ids=torch.tensor([1], dtype=torch.int64),
            hidden=torch.ones((1, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0], dtype=torch.int64),
            req_id_per_token=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
            input_lengths=torch.tensor([1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={0: torch.tensor([[1]], dtype=torch.int32)},
            seq_len_full=1,
            prefix_length=0,
        )

        report = analyze_prefill_capture_surface(
            state,
            input_ids=state.input_ids,
            hidden=state.hidden,
            position_ids=state.request.position_ids,
            req_id_per_token=state.request.req_id_per_token,
            cu_seqlens=state.request.cu_seqlens,
            input_lengths=state.request.input_lengths,
            prefix_lengths=state.request.prefix_lengths,
            block_tables_by_type=state.request.block_tables_by_type,
            meta_by_ratio={0: MetaWithOpaque(object(), torch.ones(1))},
        )

        self.assertFalse(report.static_bound)
        self.assertIn("meta.0.tensor.opaque", report.skipped_critical)

    def test_static_prefill_graph_state_rejects_hidden_shape_change(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )

        with self.assertRaisesRegex(ValueError, "hidden tail shape"):
            state.update(
                input_ids=torch.tensor([1], dtype=torch.int64),
                hidden=torch.ones((1, 2, 4), dtype=torch.bfloat16),
                position_ids=torch.tensor([0], dtype=torch.int64),
                req_id_per_token=torch.tensor([0], dtype=torch.int32),
                cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
                input_lengths=torch.tensor([1], dtype=torch.int32),
                prefix_lengths=torch.tensor([0], dtype=torch.int32),
                block_tables_by_type={},
                seq_len_full=1,
                prefix_length=0,
            )

    def test_static_prefill_graph_state_rejects_input_position_length_mismatch(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )

        with self.assertRaisesRegex(ValueError, "position_ids"):
            state.update(
                input_ids=torch.tensor([1, 2], dtype=torch.int64),
                hidden=torch.ones((2, 2, 3), dtype=torch.bfloat16),
                position_ids=torch.tensor([0], dtype=torch.int64),
                req_id_per_token=torch.tensor([0], dtype=torch.int32),
                cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
                input_lengths=torch.tensor([1], dtype=torch.int32),
                prefix_lengths=torch.tensor([0], dtype=torch.int32),
                block_tables_by_type={},
                seq_len_full=1,
                prefix_length=0,
            )

    def test_static_prefill_graph_state_marks_invalid_after_failed_update(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        good = dict(
            input_ids=torch.tensor([1], dtype=torch.int64),
            hidden=torch.ones((1, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0], dtype=torch.int64),
            req_id_per_token=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
            input_lengths=torch.tensor([1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={},
            seq_len_full=1,
            prefix_length=0,
        )
        state.update(**good)
        self.assertTrue(state.valid)

        with self.assertRaisesRegex(ValueError, "position_ids"):
            state.update(
                **{
                    **good,
                    "input_ids": torch.tensor([1, 2], dtype=torch.int64),
                    "hidden": torch.ones((2, 2, 3), dtype=torch.bfloat16),
                }
            )
        self.assertFalse(state.valid)

        state.update(**good)
        self.assertTrue(state.valid)
        with self.assertRaisesRegex(TypeError, "float"):
            state.update(**good, meta_by_ratio={0: {"bad": 1.25}})
        self.assertFalse(state.valid)

    def test_static_prefill_graph_state_manager_reuses_state_by_key(self):
        manager = StaticPrefillGraphStateManager(device="cpu")
        key = PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8)
        first = manager.get_or_create(
            key,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        second = manager.get_or_create(
            key,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )

        self.assertIs(first, second)
        with self.assertRaisesRegex(ValueError, "hidden_shape_tail"):
            manager.get_or_create(
                key,
                hidden_shape_tail=(2, 4),
                hidden_dtype=torch.bfloat16,
                block_cap=2,
                block_table_keys=(0,),
            )
        with self.assertRaisesRegex(ValueError, "block_cap"):
            manager.get_or_create(
                key,
                hidden_shape_tail=(2, 3),
                hidden_dtype=torch.bfloat16,
                block_cap=3,
                block_table_keys=(0,),
            )
        with self.assertRaisesRegex(ValueError, "block_table_keys"):
            manager.get_or_create(
                key,
                hidden_shape_tail=(2, 3),
                hidden_dtype=torch.bfloat16,
                block_cap=2,
                block_table_keys=(0, 4),
            )

    def test_static_prefill_graph_state_uses_local_token_bucket(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=512, batch_bucket=1, cp_size=8),
            local_token_bucket=64,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(0,),
        )

        self.assertEqual(tuple(state.hidden.shape), (64, 2, 3))
        self.assertEqual(tuple(state.input_ids.shape), (64,))

    def test_try_update_static_prefill_graph_state_success_and_fail_closed(self):
        class FakeV4:
            pass

        v4 = FakeV4()
        decision = select_prefill_graph_key(
            PrefillGraphRequest(token_count=3, batch_size=2, cp_size=8),
            enabled=True,
            token_buckets=(4,),
            batch_buckets=(2,),
        )
        updated = try_update_static_prefill_graph_state(
            v4,
            decision,
            input_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
            hidden=torch.ones((3, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
            req_id_per_token=torch.tensor([0, 0, 1], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 2, 3], dtype=torch.int64),
            input_lengths=torch.tensor([2, 1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0, 0], dtype=torch.int32),
            block_tables_by_type={4: torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)},
            seq_len_full=3,
            prefix_length=0,
            meta_by_ratio={0: TinyMeta(torch.ones(1), torch.arange(1), 1)},
            block_cap=2,
        )

        self.assertTrue(updated.enabled)
        self.assertEqual(updated.key.local_token_bucket, 3)
        self.assertEqual(updated.key.hidden_shape_tail, (2, 3))
        self.assertEqual(updated.key.block_cap, 2)
        self.assertEqual(updated.key.block_table_keys, (4,))
        self.assertIsNone(v4._last_prefill_graph_state_error)
        self.assertTrue(v4._last_prefill_graph_state.valid)

        failed = try_update_static_prefill_graph_state(
            v4,
            decision,
            input_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
            hidden=torch.ones((3, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
            req_id_per_token=torch.tensor([0, 1, 1], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 2, 3], dtype=torch.int64),
            input_lengths=torch.tensor([2, 1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0, 0], dtype=torch.int32),
            block_tables_by_type={4: torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)},
            seq_len_full=3,
            prefix_length=0,
            meta_by_ratio={0: TinyMeta(torch.ones(1), torch.arange(1), 1)},
            block_cap=2,
        )

        self.assertFalse(failed.enabled)
        self.assertEqual(failed.reason, "static_update_failed")
        self.assertIsNone(v4._last_prefill_graph_state)
        self.assertIn("req_id_per_token", v4._last_prefill_graph_state_error)

        disabled = try_update_static_prefill_graph_state(
            v4,
            PrefillGraphDecision(False, updated.key, "disabled_for_test"),
            input_ids=torch.tensor([1], dtype=torch.int64),
            hidden=torch.ones((1, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0], dtype=torch.int64),
            req_id_per_token=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
            input_lengths=torch.tensor([1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={},
            seq_len_full=1,
            prefix_length=0,
            meta_by_ratio=None,
            block_cap=1,
        )
        self.assertFalse(disabled.enabled)
        self.assertIsNone(v4._last_prefill_graph_state)
        self.assertEqual(v4._last_prefill_graph_state_error, "disabled_for_test")

    def test_try_update_static_prefill_graph_state_catches_runtime_error(self):
        class FakeManager:
            def get_or_create(self, *args, **kwargs):
                raise RuntimeError("simulated allocator failure")

        class FakeV4:
            pass

        v4 = FakeV4()
        v4._dsv4_static_prefill_graph_state_manager = FakeManager()
        decision = PrefillGraphDecision(
            True, PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8), "ok"
        )
        failed = try_update_static_prefill_graph_state(
            v4,
            decision,
            input_ids=torch.tensor([1], dtype=torch.int64),
            hidden=torch.ones((1, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0], dtype=torch.int64),
            req_id_per_token=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
            input_lengths=torch.tensor([1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={},
            seq_len_full=1,
            prefix_length=0,
            meta_by_ratio=None,
            block_cap=1,
        )

        self.assertFalse(failed.enabled)
        self.assertEqual(failed.reason, "static_update_failed")
        self.assertIn("simulated allocator failure", v4._last_prefill_graph_state_error)

    def test_try_update_static_prefill_graph_state_rejects_pointer_drift(self):
        class FakeV4:
            pass

        v4 = FakeV4()
        decision = select_prefill_graph_key(
            PrefillGraphRequest(token_count=1, batch_size=1, cp_size=8),
            enabled=True,
            token_buckets=(4,),
            batch_buckets=(1,),
        )
        common = dict(
            input_ids=torch.tensor([1], dtype=torch.int64),
            hidden=torch.ones((1, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.tensor([0], dtype=torch.int64),
            req_id_per_token=torch.tensor([0], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int64),
            input_lengths=torch.tensor([1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={},
            seq_len_full=1,
            prefix_length=0,
            block_cap=1,
        )

        first = try_update_static_prefill_graph_state(
            v4,
            decision,
            meta_by_ratio=None,
            **common,
        )
        self.assertTrue(first.enabled)
        self.assertTrue(v4._last_prefill_graph_state.valid)
        self.assertTrue(v4._last_prefill_graph_state.pointer_stable)
        v4._last_prefill_graph_state.mark_cuda_graph_captured(object())
        self.assertTrue(v4._last_prefill_graph_state.cuda_graph_ready)

        drifted = try_update_static_prefill_graph_state(
            v4,
            decision,
            meta_by_ratio={0: TinyMeta(torch.ones(1), torch.arange(1), 1)},
            **common,
        )

        self.assertFalse(drifted.enabled)
        self.assertEqual(drifted.reason, "pointer_drift")
        self.assertEqual(v4._last_prefill_graph_state_error, "pointer_drift")
        self.assertFalse(v4._last_prefill_graph_state.valid)
        self.assertFalse(v4._last_prefill_graph_state.pointer_stable)
        self.assertIsNone(v4._last_prefill_graph_state.cuda_graph)
        self.assertEqual(
            v4._last_prefill_graph_state.graph_capture_error,
            "pointer_drift",
        )

    def test_exact_static_prefill_layer_loop_args_returns_static_tensors(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=3, batch_bucket=2, cp_size=8),
            local_token_bucket=3,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(4,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(4,),
        )
        input_ids = torch.tensor([1, 2, 3], dtype=torch.int64)
        hidden = torch.arange(18, dtype=torch.float32).view(3, 2, 3).to(torch.bfloat16)
        positions = torch.tensor([0, 1, 2], dtype=torch.int64)
        cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int64)
        block_tables = {4: torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)}
        state.update(
            input_ids=input_ids,
            hidden=hidden,
            position_ids=positions,
            req_id_per_token=torch.tensor([0, 0, 1], dtype=torch.int32),
            cu_seqlens=cu_seqlens,
            input_lengths=torch.tensor([2, 1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0, 0], dtype=torch.int32),
            block_tables_by_type=block_tables,
            seq_len_full=3,
            prefix_length=0,
            meta_by_ratio=None,
        )

        args = exact_static_prefill_layer_loop_args(
            state,
            input_ids=input_ids.clone(),
            hidden=hidden.clone(),
            position_ids=positions.clone(),
            cu_seqlens=cu_seqlens.clone(),
            block_tables_by_type={4: block_tables[4].clone()},
        )

        self.assertIs(args.input_ids, state.input_ids)
        self.assertIs(args.hidden, state.hidden)
        self.assertIs(args.position_ids, state.request.position_ids)
        self.assertIs(args.cu_seqlens, state.request.cu_seqlens)
        self.assertIs(args.block_tables_by_type[4], state.request.block_tables_by_type[4])
        self.assertTrue(torch.equal(args.input_ids, input_ids))
        self.assertTrue(torch.equal(args.position_ids, positions))
        self.assertTrue(torch.equal(args.block_tables_by_type[4], block_tables[4]))

    def test_exact_static_prefill_layer_loop_args_rejects_padded_shapes(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(4,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(4,),
        )
        state.update(
            input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            hidden=torch.ones((4, 2, 3), dtype=torch.bfloat16),
            position_ids=torch.arange(4, dtype=torch.int64),
            req_id_per_token=torch.zeros(4, dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 4], dtype=torch.int64),
            input_lengths=torch.tensor([4], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={4: torch.ones((1, 4), dtype=torch.int32)},
            seq_len_full=4,
            prefix_length=0,
            meta_by_ratio=None,
        )

        with self.assertRaisesRegex(ValueError, "token_count=3"):
            exact_static_prefill_layer_loop_args(
                state,
                input_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
                hidden=torch.ones((3, 2, 3), dtype=torch.bfloat16),
                position_ids=torch.arange(3, dtype=torch.int64),
                cu_seqlens=torch.tensor([0, 3], dtype=torch.int64),
                block_tables_by_type={4: torch.ones((1, 4), dtype=torch.int32)},
            )
        with self.assertRaisesRegex(ValueError, "block table 4 shape"):
            exact_static_prefill_layer_loop_args(
                state,
                input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
                hidden=torch.ones((4, 2, 3), dtype=torch.bfloat16),
                position_ids=torch.arange(4, dtype=torch.int64),
                cu_seqlens=torch.tensor([0, 4], dtype=torch.int64),
                block_tables_by_type={4: torch.ones((1, 2), dtype=torch.int32)},
            )

    def test_copy_graph_kv_to_live_copies_referenced_blocks_only(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        graph_regions = []
        live_regions = []
        graph_scale_regions = []
        live_scale_regions = []
        for layer in range(2):
            graph_row = [torch.empty((0,)) for _ in range(8)]
            live_row = [torch.empty((0,)) for _ in range(8)]
            graph_scale_row = [torch.empty((0,)) for _ in range(8)]
            live_scale_row = [torch.empty((0,)) for _ in range(8)]
            graph_row[1] = torch.arange(40, dtype=torch.float32).view(5, 8) + 100 * layer
            live_row[1] = torch.full((5, 8), -1.0)
            graph_scale_row[1] = torch.arange(10, dtype=torch.float32).view(5, 2) + 10 * layer
            live_scale_row[1] = torch.full((5, 2), -2.0)
            graph_regions.append(graph_row)
            live_regions.append(live_row)
            graph_scale_regions.append(graph_scale_row)
            live_scale_regions.append(live_scale_row)
        state.graph_kv_cache = SimpleNamespace(
            kv_cache_base_by_layer_region=graph_regions,
            kv_scale_base_by_layer_region=graph_scale_regions,
        )
        state.graph_kv_block_cap = 5
        live_kv_cache = SimpleNamespace(
            kv_cache_base_by_layer_region=live_regions,
            kv_scale_base_by_layer_region=live_scale_regions,
        )

        copied = state.copy_graph_kv_to_live(
            live_kv_cache,
            {1: torch.tensor([[1, 3, 0, -1, 1]], dtype=torch.int32)},
        )

        self.assertEqual(copied, 4)
        for layer in range(2):
            self.assertTrue(torch.equal(live_regions[layer][1][1], graph_regions[layer][1][1]))
            self.assertTrue(torch.equal(live_regions[layer][1][3], graph_regions[layer][1][3]))
            self.assertTrue(torch.all(live_regions[layer][1][0] == -1.0))
            self.assertTrue(torch.all(live_regions[layer][1][2] == -1.0))
            self.assertTrue(
                torch.equal(live_scale_regions[layer][1][1], graph_scale_regions[layer][1][1])
            )
            self.assertTrue(torch.all(live_scale_regions[layer][1][4] == -2.0))

    def test_copy_graph_kv_to_live_skips_layers_without_region_pool(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        graph_regions = [[torch.empty((0,)) for _ in range(8)] for _ in range(2)]
        live_regions = [[torch.empty((0,)) for _ in range(8)] for _ in range(2)]
        graph_regions[1][1] = torch.arange(32, dtype=torch.float32).view(4, 8)
        live_regions[1][1] = torch.full((4, 8), -1.0)
        state.graph_kv_cache = SimpleNamespace(
            kv_cache_base_by_layer_region=graph_regions
        )
        state.graph_kv_block_cap = 4

        copied = state.copy_graph_kv_to_live(
            SimpleNamespace(kv_cache_base_by_layer_region=live_regions),
            {1: torch.tensor([[1, 3]], dtype=torch.int32)},
        )

        self.assertEqual(copied, 1)
        self.assertEqual(graph_regions[0][1].numel(), 0)
        self.assertTrue(torch.equal(live_regions[1][1][1], graph_regions[1][1][1]))
        self.assertTrue(torch.equal(live_regions[1][1][3], graph_regions[1][1][3]))
        self.assertTrue(torch.all(live_regions[1][1][0] == -1.0))

    def test_copy_graph_kv_to_live_skips_layer_region_non_owner(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        graph_regions = [[torch.empty((0,)) for _ in range(8)] for _ in range(2)]
        live_regions = [[torch.empty((0,)) for _ in range(8)] for _ in range(2)]
        # Layer 0 has a stale/non-serving graph tensor for region 1 but does not
        # own that region in live layer_region_to_group_id. Copy-out must skip
        # it instead of treating the global region block table as applying to
        # every layer.
        graph_regions[0][1] = torch.full((4, 8), 9.0)
        graph_regions[1][1] = torch.arange(32, dtype=torch.float32).view(4, 8)
        live_regions[1][1] = torch.full((4, 8), -1.0)
        state.graph_kv_cache = SimpleNamespace(
            kv_cache_base_by_layer_region=graph_regions
        )
        state.graph_kv_block_cap = 4
        live_kv_cache = SimpleNamespace(
            kv_cache_base_by_layer_region=live_regions,
            layer_region_to_group_id=[
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 3, -1, -1, -1, -1, -1, -1],
            ],
        )

        copied = state.copy_graph_kv_to_live(
            live_kv_cache,
            {1: torch.tensor([[1, 3]], dtype=torch.int32)},
        )

        self.assertEqual(copied, 1)
        self.assertEqual(graph_regions[0][1].numel(), 32)
        self.assertTrue(torch.equal(live_regions[1][1][1], graph_regions[1][1][1]))
        self.assertTrue(torch.equal(live_regions[1][1][3], graph_regions[1][1][3]))

    def test_copy_graph_kv_to_live_does_not_overwrite_non_owner_live_tensor(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        graph_regions = [[torch.empty((0,)) for _ in range(8)] for _ in range(2)]
        live_regions = [[torch.empty((0,)) for _ in range(8)] for _ in range(2)]
        graph_regions[0][1] = torch.full((4, 8), 9.0)
        graph_regions[1][1] = torch.arange(32, dtype=torch.float32).view(4, 8)
        live_regions[0][1] = torch.full((4, 8), -7.0)
        live_regions[1][1] = torch.full((4, 8), -1.0)
        state.graph_kv_cache = SimpleNamespace(
            kv_cache_base_by_layer_region=graph_regions
        )
        state.graph_kv_block_cap = 4
        live_kv_cache = SimpleNamespace(
            kv_cache_base_by_layer_region=live_regions,
            layer_region_to_group_id=[
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 3, -1, -1, -1, -1, -1, -1],
            ],
        )

        copied = state.copy_graph_kv_to_live(
            live_kv_cache,
            {1: torch.tensor([[1, 3]], dtype=torch.int32)},
        )

        self.assertEqual(copied, 1)
        self.assertTrue(torch.all(live_regions[0][1] == -7.0))
        self.assertTrue(torch.equal(live_regions[1][1][1], graph_regions[1][1][1]))
        self.assertTrue(torch.equal(live_regions[1][1][3], graph_regions[1][1][3]))

    def test_copy_graph_kv_to_live_rejects_region_without_owner(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        graph_regions = [[torch.empty((0,)) for _ in range(8)]]
        live_regions = [[torch.empty((0,)) for _ in range(8)]]
        graph_regions[0][1] = torch.full((4, 8), 9.0)
        live_regions[0][1] = torch.full((4, 8), -7.0)
        state.graph_kv_cache = SimpleNamespace(
            kv_cache_base_by_layer_region=graph_regions
        )
        state.graph_kv_block_cap = 4
        live_kv_cache = SimpleNamespace(
            kv_cache_base_by_layer_region=live_regions,
            layer_region_to_group_id=[
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ],
        )

        with self.assertRaisesRegex(ValueError, "no owning layers"):
            state.copy_graph_kv_to_live(
                live_kv_cache,
                {1: torch.tensor([[1, 3]], dtype=torch.int32)},
            )
        self.assertTrue(torch.all(live_regions[0][1] == -7.0))

    def test_copy_graph_kv_to_live_rejects_block_overflow(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(1,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=2,
            block_table_keys=(1,),
        )
        row = [[torch.empty((0,)) for _ in range(8)]]
        row[0][1] = torch.zeros((2, 4))
        state.graph_kv_cache = SimpleNamespace(kv_cache_base_by_layer_region=row)
        state.graph_kv_block_cap = 2

        with self.assertRaisesRegex(ValueError, "capacity exceeded"):
            state.copy_graph_kv_to_live(
                SimpleNamespace(kv_cache_base_by_layer_region=row),
                {1: torch.tensor([[2]], dtype=torch.int32)},
            )

    def test_copy_graph_kv_to_live_rejects_missing_live_tensor(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        graph_row = [[torch.empty((0,)) for _ in range(8)]]
        live_row = [[torch.empty((0,)) for _ in range(8)]]
        graph_row[0][1] = torch.zeros((4, 8))
        state.graph_kv_cache = SimpleNamespace(kv_cache_base_by_layer_region=graph_row)
        state.graph_kv_block_cap = 4

        with self.assertRaisesRegex(ValueError, "missing tensor"):
            state.copy_graph_kv_to_live(
                SimpleNamespace(kv_cache_base_by_layer_region=live_row),
                {1: torch.tensor([[1]], dtype=torch.int32)},
            )

    def test_copy_graph_kv_to_live_rejects_region_without_base_copy(self):
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=8),
            local_token_bucket=4,
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 3),
            hidden_dtype=torch.bfloat16,
            block_cap=4,
            block_table_keys=(1,),
        )
        empty_row = [[torch.empty((0,)) for _ in range(8)]]
        state.graph_kv_cache = SimpleNamespace(kv_cache_base_by_layer_region=empty_row)
        state.graph_kv_block_cap = 4

        with self.assertRaisesRegex(ValueError, "no base tensors"):
            state.copy_graph_kv_to_live(
                SimpleNamespace(kv_cache_base_by_layer_region=empty_row),
                {1: torch.tensor([[1]], dtype=torch.int32)},
            )


if __name__ == "__main__":
    unittest.main()
