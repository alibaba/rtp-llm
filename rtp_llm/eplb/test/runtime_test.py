import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.eplb.runtime import EplbLayer, RuntimeExpertBalancer, _ExpertLayoutGroup


def make_config(**updates):
    values = dict(
        eplb_mode=2,
        eplb_update_time=4,
        eplb_stats_window_size=4,
        eplb_rearrange_after_prefill=False,
        eplb_rearrange_p2p_groups=1,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def make_layer(
    physical=(0, 1, 2, 3, 0, 1),
    layer_id=0,
    ep_rank=0,
    ep_size=1,
    device="cpu",
    replica_offset=0,
    weight_width=6,
):
    logical = max(physical) + 1
    inverse = torch.full((logical, len(physical) - logical + 1), -1, dtype=torch.int32)
    counts = torch.zeros(logical, dtype=torch.int32)
    for slot, expert in enumerate(physical):
        inverse[expert, counts[expert]] = slot
        counts[expert] += 1
    local = len(physical) // ep_size
    ids = torch.tensor(physical[ep_rank * local : (ep_rank + 1) * local], device=device)
    weight = (ids[:, None] * 100 + torch.arange(weight_width, device=device)).to(
        torch.int32
    )
    scale = (
        (ids[:, None, None] * 10 + torch.arange(6, device=device).reshape(2, 3))
        .float()
        .transpose(1, 2)
    )
    return EplbLayer(
        layer_id,
        inverse.to(device),
        counts.to(device),
        (weight, scale),
        ep_rank,
        replica_offset,
    )


def routing_oracle(layer, ids):
    """Scalar DeepGEMM uint32 token hash, independent of tensor broadcasting."""
    mapping = layer.logical_to_physical.cpu().tolist()
    replicas = layer.logical_count.cpu().tolist()
    topk = ids.size(-1) if ids.ndim > 1 else 1
    mapped = []
    for position, logical in enumerate(ids.cpu().reshape(-1).tolist()):
        token = position // topk
        seed = (layer.replica_offset + token * 23333) & 0xFFFFFFFF
        mapped.append(-1 if logical < 0 else mapping[logical][seed % replicas[logical]])
    return torch.tensor(mapped, dtype=ids.dtype, device=ids.device).reshape(ids.shape)


class RuntimeTest(unittest.TestCase):
    def test_materialized_layer_initializes_under_meta_device_context(self):
        mapping = torch.tensor([[0, 2], [1, -1]], dtype=torch.int32)
        counts = torch.tensor([2, 1], dtype=torch.int32)
        weight = torch.arange(12).reshape(3, 4)
        ids = torch.tensor([[0, 1, -1], [0, 1, -1]], dtype=torch.int32)
        with torch.device("meta"):
            layer = EplbLayer(0, mapping, counts, (weight,))
            mapped = layer.map_ids(ids)
        self.assertIs(layer.logical_to_physical, mapping)
        self.assertIs(layer.logical_count, counts)
        self.assertIs(layer.weights[0], weight)
        for metadata in (mapping, counts, layer.physical_to_logical):
            self.assertEqual(metadata.device.type, "cpu")
            self.assertFalse(metadata.is_meta)
            self.assertEqual(metadata.dtype, torch.int32)
        self.assertEqual(layer.physical_to_logical.tolist(), [0, 1, 0])
        self.assertEqual(mapped.tolist(), [[0, 1, -1], [2, 1, -1]])

    def test_map_ids_preserves_sentinel_shape_dtype_and_inputs(self):
        layer = make_layer()
        for dtype in (torch.int32, torch.int64):
            ids = torch.tensor(
                [[[0, 1, -1], [0, 1, 3]], [[3, 2, 0], [1, -1, 2]]], dtype=dtype
            )
            saved = ids.clone()
            mapped = layer.map_ids(ids)
            self.assertEqual(mapped.dtype, dtype)
            self.assertEqual(mapped.shape, ids.shape)
            self.assertTrue(torch.equal(ids, saved))
            self.assertTrue(torch.equal(mapped, routing_oracle(layer, ids)))
            self.assertTrue(
                torch.equal(
                    layer.physical_to_logical[mapped[ids >= 0]], ids[ids >= 0].int()
                )
            )

    def test_token_hash_has_fixed_reference_values_across_topk_columns(self):
        layer = make_layer((0, 1, 0, 1, 0, 1))
        ids = torch.tensor([[0, 0, 1]] * 3, dtype=torch.int32)
        self.assertEqual(layer.map_ids(ids).tolist(), [[0, 0, 1], [4, 4, 5], [2, 2, 3]])

    def test_token_hash_wraps_uint32_before_replica_modulo(self):
        layer = make_layer((0, 1, 0, 1, 0, 1), replica_offset=0xFFFFFFFD)
        ids = torch.zeros(6, dtype=torch.int64)
        self.assertEqual(layer.map_ids(ids).tolist(), [2, 4, 2, 0, 4, 2])
        self.assertTrue(torch.equal(layer.map_ids(ids), routing_oracle(layer, ids)))

    def test_one_dimensional_ids_are_a_sequence_of_tokens(self):
        layer = make_layer((0, 1, 0, 1, 0, 1))
        ids = torch.zeros(6, dtype=torch.int32)
        self.assertEqual(layer.map_ids(ids).tolist(), [0, 4, 2, 0, 4, 2])

    def test_tp_storage_rank_does_not_change_shared_input_replica(self):
        first = make_layer(ep_rank=0, ep_size=2, replica_offset=7)
        second = make_layer(ep_rank=1, ep_size=2, replica_offset=7)
        ids = torch.tensor([[0, 1], [1, 0], [0, 3]], dtype=torch.int32)
        self.assertTrue(torch.equal(first.map_ids(ids), second.map_ids(ids)))
        self.assertTrue(torch.equal(first.map_ids(ids), routing_oracle(first, ids)))

    def test_explicit_offset_rotates_independent_inputs(self):
        first = make_layer(ep_rank=0, ep_size=2, replica_offset=0)
        second = make_layer(ep_rank=1, ep_size=2, replica_offset=1)
        ids = torch.tensor([[0]], dtype=torch.int32)
        self.assertEqual(first.map_ids(ids).item(), 0)
        self.assertEqual(second.map_ids(ids).item(), 4)

    def _group(self, config=None, layers=None, normal=False):
        return _ExpertLayoutGroup(
            layers or [make_layer()], config or make_config(), None, "target", normal
        )

    def test_communication_views_keep_expert_axis_and_executor_storage(self):
        layer = make_layer((0, 1, 2))
        strided_weight = torch.arange(36).reshape(3, 12)[:, ::2]
        layer.weights = (strided_weight, layer.weights[1])
        group = self._group(layers=[layer])
        self.assertEqual(group.weights[0][0].shape, strided_weight.shape)
        self.assertEqual(group.weights[0][0].stride(), strided_weight.stride())
        self.assertEqual(group.weights[0][1].shape, layer.weights[1].mT.shape)
        for original, communication in zip(layer.weights, group.weights[0]):
            self.assertEqual(communication.size(0), layer.num_local_experts)
            self.assertEqual(communication.data_ptr(), original.data_ptr())

    def test_legacy_deepep_uses_p2p_without_symmetric_memory(self):
        group = self._group()
        # The pinned CUDA 12 DeepEP wheel has no physical-domain API.
        # Exercise dependency selection only; no GPU communication is mocked as tested.
        with patch.dict("sys.modules", {"deep_ep": SimpleNamespace()}), patch(
            "rtp_llm.utils.deepep_utils.configure_deepep_nccl_root"
        ), patch("torch.cuda.empty_cache"), self.assertLogs(level="WARNING") as logs:
            group.initialize_shuffle()
        self.assertFalse(group.intranode_shuffle)
        self.assertFalse(hasattr(group, "symmetric_weight_buffers"))
        self.assertIn("using P2P shuffle", logs.output[0])

    def test_first_update_after_quarter_interval_then_each_full_interval(self):
        for interval in (4, 10, 16):
            with self.subTest(interval=interval):
                group = self._group(make_config(eplb_update_time=interval))
                first = max(1, interval // 4)
                counts = torch.ones(1, 6, dtype=torch.int32)
                with patch.object(group, "rearrange", return_value=True) as rearrange:
                    for step in range(1, first + interval + 1):
                        changed = group.step(counts, False, 2, interval)
                        self.assertEqual(changed, step in (first, first + interval))
                    self.assertEqual(rearrange.call_count, 2)

    def test_ring_history_is_retained_across_updates(self):
        group = self._group()
        with patch.object(group, "rearrange", return_value=True) as rearrange:
            for value in (10, 20, 30, 40, 50, 60):
                group.step(torch.full((1, 6), value, dtype=torch.int32), False, 2, 4)
            self.assertEqual(rearrange.call_count, 2)
        self.assertEqual(group.window[:, 0, 0].tolist(), [50, 60, 30, 40])
        self.assertEqual(group.window_index, 2)
        self.assertEqual(group.window.dtype, torch.int32)

    def test_counts_are_borrowed_and_only_selected_layer_rows_are_sampled(self):
        group = self._group(layers=[make_layer(layer_id=2), make_layer(layer_id=5)])
        counts = torch.arange(36, dtype=torch.int32).reshape(6, 6)
        original = counts.clone()
        with patch.object(group, "rearrange", return_value=True):
            group.step(counts, False, 2, 4)
        self.assertTrue(torch.equal(counts, original))
        self.assertTrue(torch.equal(group.window[0], counts[[2, 5]]))

    def test_normal_engine_prefill_is_skipped_by_default(self):
        group = self._group(normal=True)
        counts = torch.ones(1, 6, dtype=torch.int32)
        with patch.object(group, "rearrange", return_value=True) as rearrange:
            self.assertFalse(group.step(counts, True, 2, 4))
            self.assertEqual(group.window_index, 0)
            self.assertEqual(group.steps, 3)
            self.assertEqual(torch.count_nonzero(group.window).item(), 0)
            rearrange.assert_not_called()
            self.assertTrue(group.step(counts, False, 2, 4))
            rearrange.assert_called_once_with()

    def test_normal_prefill_samples_then_defers_rearrange_to_decode(self):
        group = self._group(make_config(eplb_rearrange_after_prefill=True), normal=True)
        with patch.object(group, "rearrange", return_value=True) as rearrange:
            for value in (11, 22):
                self.assertFalse(group.step(torch.full((1, 6), value), True, 2, 4))
            rearrange.assert_not_called()
            self.assertEqual(group.steps, 3)
            self.assertTrue(group.step(torch.full((1, 6), 33), False, 2, 4))
            rearrange.assert_called_once_with()
        self.assertEqual(group.window[:3, 0, 0].tolist(), [11, 22, 33])

    def test_other_engines_include_prefill_in_periodic_schedule(self):
        group = self._group(normal=False)
        with patch.object(group, "rearrange", return_value=True) as rearrange:
            self.assertTrue(group.step(torch.ones(1, 6, dtype=torch.int32), True, 2, 4))
            rearrange.assert_called_once_with()

    def test_stats_samples_without_mutating_weights_or_maps(self):
        layer = make_layer()
        group = self._group(layers=[layer])
        tensors = (
            *layer.weights,
            layer.logical_to_physical,
            layer.logical_count,
            layer.physical_to_logical,
        )
        saved = [(tensor.clone(), tensor.data_ptr()) for tensor in tensors]
        with patch.object(group, "rearrange") as rearrange:
            for value in range(1, 7):
                self.assertFalse(group.step(torch.full((1, 6), value), False, 1, 4))
            rearrange.assert_not_called()
        self.assertEqual(group.window[:, 0, 0].tolist(), [5, 6, 3, 4])
        for tensor, (original, pointer) in zip(tensors, saved):
            self.assertTrue(torch.equal(tensor, original))
            self.assertEqual(tensor.data_ptr(), pointer)

    def test_startup_config_sets_mode_and_initial_schedule(self):
        for mode in (0, 1, 2, 3):
            for interval in (0, 4, 1000):
                with self.subTest(mode=mode, interval=interval):
                    runtime = RuntimeExpertBalancer(
                        [make_layer()],
                        make_config(eplb_mode=mode, eplb_update_time=interval),
                        0,
                        1,
                        0,
                    )
                    self.assertEqual((runtime.mode, runtime.interval), (mode, interval))
                    group = runtime.groups[0]
                    self.assertEqual(group.window_index, 0)
                    self.assertEqual(
                        group.steps, max(0, interval - max(1, interval // 4))
                    )
                    self.assertEqual(torch.count_nonzero(group.window).item(), 0)

    def test_ep_one_explicitly_skips_sampling_and_rearrange(self):
        with patch.object(
            _ExpertLayoutGroup,
            "initialize_shuffle",
            side_effect=AssertionError("EP=1 must not initialize communication"),
        ):
            runtime = RuntimeExpertBalancer([make_layer()], make_config(), 0, 1, 0)
        self.assertIsNone(runtime.prompt_flag)
        group = runtime.groups[0]
        counts = torch.ones(1, 6, dtype=torch.int32)
        with patch.object(group, "rearrange") as rearrange:
            for _ in range(8):
                self.assertFalse(runtime.step(counts))
            rearrange.assert_not_called()
        self.assertEqual(group.window_index, 0)
        self.assertEqual(torch.count_nonzero(group.window).item(), 0)
        self.assertEqual(counts.tolist(), [[1] * 6])

    def test_multiple_layout_groups_all_execute_after_first_reports_changed(self):
        layers = [
            make_layer(layer_id=0),
            make_layer(layer_id=2),
            make_layer(layer_id=5, weight_width=8),
        ]
        runtime = RuntimeExpertBalancer(layers, make_config(), 0, 1, 0)
        self.assertEqual(
            [[layer.layer_id for layer in group.layers] for group in runtime.groups],
            [[0, 2], [5]],
        )
        # Test CPU dispatch with real layout groups, not the CUDA-only EP>1 constructor.
        # rearrange alone is replaced; this does not claim to exercise NCCL.
        schedule = SimpleNamespace(
            mode=2, ep_size=2, interval=4, groups=runtime.groups, prompt_flag=None
        )
        with patch.object(
            runtime.groups[0], "rearrange", return_value=True
        ) as first, patch.object(
            runtime.groups[1], "rearrange", return_value=False
        ) as second:
            self.assertTrue(
                RuntimeExpertBalancer.step(
                    schedule, torch.ones(6, 6, dtype=torch.int32)
                )
            )
            first.assert_called_once_with()
            second.assert_called_once_with()
        for group in runtime.groups:
            self.assertEqual(group.window_index, 1)

    def test_none_and_zero_interval_do_not_sample(self):
        group = self._group()
        counts = torch.ones(1, 6, dtype=torch.int32)
        schedule = SimpleNamespace(
            mode=0, ep_size=2, interval=4, groups=(group,), prompt_flag=None
        )
        with patch.object(group, "rearrange") as rearrange:
            self.assertFalse(RuntimeExpertBalancer.step(schedule, counts))
            self.assertFalse(group.step(counts, False, 2, 0))
            rearrange.assert_not_called()
        self.assertEqual(torch.count_nonzero(group.window).item(), 0)
        RuntimeExpertBalancer(
            [make_layer()],
            make_config(eplb_update_time=0, eplb_stats_window_size=0),
            0,
            1,
            0,
        )

    def test_invalid_layout_and_runtime_config_are_rejected(self):
        layer = make_layer()
        bad = layer.logical_to_physical.clone()
        bad[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "cover every physical slot"):
            EplbLayer(0, bad, layer.logical_count, layer.weights)
        for updates in (
            dict(eplb_stats_window_size=-1),
            dict(eplb_stats_window_size=0),
            dict(eplb_stats_window_size=5),
            dict(eplb_update_time=-1),
            dict(eplb_rearrange_p2p_groups=0),
            dict(eplb_mode=9),
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                RuntimeExpertBalancer([layer], make_config(**updates), 0, 1, 0)
        with self.assertRaisesRegex(ValueError, "EP=WORLD"):
            RuntimeExpertBalancer([make_layer(ep_size=2)], make_config(), 0, 2, 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
