import unittest
from types import SimpleNamespace

import torch

from rtp_llm.eplb.runtime import EplbLayer, RuntimeExpertBalancer, _shuffle_weights


def make_config(**updates):
    values = dict(
        eplb_mode=2,
        eplb_update_time=2,
        eplb_stats_window_size=4,
        eplb_balance_layer_per_step=1,
        balance_method="round",
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
):
    logical = max(physical) + 1
    inverse = torch.full((logical, len(physical) - logical + 1), -1, dtype=torch.int32)
    counts = torch.zeros(logical, dtype=torch.int32)
    for slot, expert in enumerate(physical):
        inverse[expert, counts[expert]] = slot
        counts[expert] += 1
    local = len(physical) // ep_size
    ids = torch.tensor(physical[ep_rank * local : (ep_rank + 1) * local], device=device)
    weight = (ids[:, None] * 100 + torch.arange(6, device=device)).to(torch.int32)
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
        for metadata, dtype in (
            (layer.logical_to_physical, torch.int32),
            (layer.logical_count, torch.int32),
            (layer.physical_to_logical, torch.int64),
        ):
            self.assertEqual(metadata.device.type, "cpu")
            self.assertFalse(metadata.is_meta)
            self.assertEqual(metadata.dtype, dtype)
        self.assertEqual(layer.logical_to_physical.tolist(), [[0, 2], [1, -1]])
        self.assertEqual(layer.logical_count.tolist(), [2, 1])
        self.assertEqual(layer.physical_to_logical.tolist(), [0, 1, 0])
        self.assertEqual(mapped.device.type, "cpu")
        self.assertEqual(mapped.dtype, ids.dtype)
        self.assertEqual(mapped.tolist(), [[0, 1, -1], [2, 1, -1]])

    def test_map_ids_preserves_sentinel_shape_and_dtype(self):
        layer = make_layer()
        ids = torch.tensor([[0, 1, -1], [0, 1, 3]], dtype=torch.int64)
        saved = ids.clone()
        mapped = layer.map_ids(ids)
        self.assertEqual(mapped.dtype, ids.dtype)
        self.assertEqual(mapped.shape, ids.shape)
        self.assertEqual(mapped[0, 2], -1)
        self.assertTrue(torch.equal(ids, saved))
        self.assertTrue(
            torch.equal(layer.physical_to_logical[mapped[ids >= 0]], ids[ids >= 0])
        )
        self.assertNotEqual(mapped[0, 0], mapped[1, 0])

    def test_replica_choice_rotates_when_topk_multiple_of_replica_count(self):
        layer = make_layer()
        mapped = layer.map_ids(
            torch.tensor([[0, 2], [0, 2], [0, 2], [0, 2]], dtype=torch.int32)
        )
        self.assertEqual(mapped[:, 0].tolist(), [0, 4, 0, 4])

    def test_storage_rank_does_not_change_replica_choice(self):
        first = make_layer(ep_rank=0, ep_size=2)
        second = make_layer(ep_rank=1, ep_size=2)
        ids = torch.tensor([[0], [1], [0]], dtype=torch.int32)
        torch.testing.assert_close(first.map_ids(ids), second.map_ids(ids))

    def test_explicit_offset_rotates_independent_inputs(self):
        first = make_layer(ep_rank=0, ep_size=2, replica_offset=0)
        second = make_layer(ep_rank=1, ep_size=2, replica_offset=1)
        ids = torch.tensor([[0]], dtype=torch.int32)
        self.assertNotEqual(first.map_ids(ids).item(), second.map_ids(ids).item())

    def test_staging_protects_cycles_noncontiguous_scales_and_aliases(self):
        layer = make_layer((0, 1, 2), layer_id=0)
        storage = torch.arange(24).reshape(3, 8)
        layer.weights = (*layer.weights, storage[:, :4], storage[:, 4:])
        original = [weight.clone() for weight in layer.weights]
        pointers = [weight.data_ptr() for weight in layer.weights]
        strides = [weight.stride() for weight in layer.weights]
        permutation = torch.tensor([2, 0, 1])
        _shuffle_weights(layer, permutation, 1)
        for weight, old, pointer, stride in zip(
            layer.weights, original, pointers, strides
        ):
            self.assertTrue(torch.equal(weight, old[permutation]))
            self.assertEqual(weight.data_ptr(), pointer)
            self.assertEqual(weight.stride(), stride)

    def test_one_source_can_fill_multiple_new_replicas(self):
        layer = make_layer()
        _shuffle_weights(layer, torch.tensor([2, 0, 1, 3, 2, 2]), 1)
        self.assertEqual(layer.weights[0][:, 0].tolist(), [200, 0, 100, 300, 200, 200])

    def test_counts_are_borrowed_and_non_moe_rows_are_not_sampled(self):
        layer = make_layer(layer_id=2)
        runtime = RuntimeExpertBalancer([layer], make_config(), 0, 1, "target")
        counts = torch.tensor([[9] * 6, [8] * 6, [7] * 6], dtype=torch.int32)
        original = counts.clone()
        runtime.record_step(counts)
        self.assertTrue(torch.equal(counts, original))
        self.assertEqual(runtime._window[0].tolist(), [[7] * 6])
        self.assertEqual(runtime._window.dtype, torch.int64)
        self.assertFalse(runtime.balance())

    def test_real_rebalance_publishes_maps_and_weights_in_place(self):
        layer = make_layer()
        runtime = RuntimeExpertBalancer(
            [layer], make_config(eplb_update_time=1), 0, 1, 0
        )
        original_pointers = [
            tensor.data_ptr()
            for tensor in (
                *layer.weights,
                layer.logical_to_physical,
                layer.logical_count,
            )
        ]
        runtime.record_step(torch.tensor([[1, 1, 1000, 0, 1, 1]], dtype=torch.int32))
        self.assertTrue(runtime.balance())
        self.assertEqual(layer.logical_count.tolist(), [1, 1, 3, 1])
        self.assertTrue(
            torch.equal(layer.weights[0][:, 0], layer.physical_to_logical * 100)
        )
        self.assertEqual(
            original_pointers,
            [
                tensor.data_ptr()
                for tensor in (
                    *layer.weights,
                    layer.logical_to_physical,
                    layer.logical_count,
                )
            ],
        )
        self.assertEqual(runtime._samples, 0)
        self.assertFalse(runtime.balance())

    def test_short_intervals_discard_previous_physical_history(self):
        layer = make_layer()
        runtime = RuntimeExpertBalancer(
            [layer], make_config(eplb_update_time=1, eplb_stats_window_size=10), 0, 1, 0
        )
        for expert in (2, 3, 1):
            counts = (layer.physical_to_logical == expert).to(torch.int32).unsqueeze(
                0
            ) * 1000
            runtime.record_step(counts)
            self.assertTrue(runtime.balance())
            self.assertEqual(layer.logical_count[expert].item(), 3)
            self.assertEqual(runtime._samples, 0)
            self.assertTrue(
                torch.equal(layer.weights[0][:, 0], layer.physical_to_logical * 100)
            )

    def test_ring_window_retains_most_recent_steps(self):
        layer = make_layer()
        runtime = RuntimeExpertBalancer(
            [layer], make_config(eplb_update_time=20, eplb_stats_window_size=2), 0, 1, 0
        )
        for value in (10, 20, 30):
            runtime.record_step(torch.full((1, 6), value, dtype=torch.int64))
        self.assertEqual(runtime._samples, 2)
        self.assertEqual(runtime._window[:, 0, 0].sort().values.tolist(), [20, 30])

    def test_zero_load_is_not_a_fake_applied_update(self):
        layer = make_layer()
        runtime = RuntimeExpertBalancer(
            [layer], make_config(eplb_update_time=1), 0, 1, 0
        )
        old = layer.physical_to_logical.clone()
        runtime.record_step(torch.zeros(1, 6, dtype=torch.int32))
        self.assertFalse(runtime.balance())
        self.assertTrue(torch.equal(old, layer.physical_to_logical))

    def test_dynamic_mode_and_interval_reset_sampled_history(self):
        layer = make_layer()
        runtime = RuntimeExpertBalancer([layer], make_config(), 0, 1, 0)
        counts = torch.ones(1, 6, dtype=torch.int32)
        runtime.record_step(counts)
        runtime.update_config(1, 2)
        self.assertEqual(runtime._samples, 0)
        runtime.record_step(counts)
        self.assertEqual(runtime._samples, 1)
        self.assertEqual(runtime._window[0].tolist(), [[1] * 6])
        self.assertFalse(runtime.balance())
        runtime.update_config(3, 3)
        runtime.record_step(counts)
        runtime.update_config(3, 3)
        self.assertEqual(runtime._samples, 1)
        runtime.update_config(3, 1)
        self.assertEqual(runtime._samples, 0)
        self.assertFalse(runtime.balance())

    def test_stats_preserves_layout_then_all_balances_new_history(self):
        layer = make_layer()
        runtime = RuntimeExpertBalancer(
            [layer], make_config(eplb_mode=1, eplb_update_time=1), 0, 1, 0
        )
        tensors = (
            *layer.weights,
            layer.logical_to_physical,
            layer.logical_count,
            layer.physical_to_logical,
        )
        original = [tensor.clone() for tensor in tensors]
        pointers = [tensor.data_ptr() for tensor in tensors]
        for _ in range(3):
            runtime.record_step(
                torch.tensor([[1, 1, 1000, 0, 1, 1]], dtype=torch.int32)
            )
            self.assertFalse(runtime.balance())
        self.assertEqual(runtime._samples, 3)
        self.assertEqual(runtime._updates, 0)
        for tensor, saved in zip(tensors, original):
            torch.testing.assert_close(tensor, saved, rtol=0, atol=0)

        runtime.update_config(3, 1)
        self.assertEqual(runtime._samples, 0)
        self.assertFalse(runtime.balance())
        runtime.record_step(torch.tensor([[1, 1, 0, 1000, 1, 1]], dtype=torch.int32))
        self.assertTrue(runtime.balance())
        self.assertEqual(layer.logical_count.tolist(), [1, 1, 1, 3])
        self.assertTrue(
            torch.equal(layer.weights[0][:, 0], layer.physical_to_logical * 100)
        )
        self.assertEqual([tensor.data_ptr() for tensor in tensors], pointers)

    def test_none_does_not_sample_or_balance(self):
        runtime = RuntimeExpertBalancer(
            [make_layer()], make_config(eplb_mode=0), 0, 1, 0
        )
        for _ in range(3):
            runtime.record_step(torch.ones(1, 6, dtype=torch.int32))
            self.assertFalse(runtime.balance())
        self.assertEqual(runtime._samples, 0)
        self.assertEqual(runtime._steps, 0)

    def test_layer_selection_uses_actual_rank_load_and_rotates(self):
        layers = [make_layer(layer_id=i) for i in range(3)]
        runtime = RuntimeExpertBalancer(
            layers, make_config(eplb_balance_layer_per_step=2), 0, 1, 0
        )
        loads = torch.tensor([[[8, 0], [4, 4], [6, 2]]])
        self.assertEqual(runtime._select_layers(loads), [0, 1])
        self.assertEqual(runtime._select_layers(loads), [2, 0])
        runtime._method = "most_unbalanced_layer"
        self.assertEqual(runtime._select_layers(loads), [0, 2])
        runtime._method = "random"
        self.assertEqual(runtime._select_layers(loads), runtime._select_layers(loads))

    def test_invalid_layout_and_runtime_config_are_rejected(self):
        layer = make_layer()
        bad = layer.logical_to_physical.clone()
        bad[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "cover every physical slot"):
            EplbLayer(0, bad, layer.logical_count, layer.weights)
        for updates in (
            dict(eplb_stats_window_size=0),
            dict(eplb_update_time=0),
            dict(eplb_balance_layer_per_step=0),
            dict(balance_method="unknown"),
            dict(eplb_mode=9),
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                RuntimeExpertBalancer([layer], make_config(**updates), 0, 1, 0)
        with self.assertRaises(ValueError):
            RuntimeExpertBalancer([make_layer(ep_size=2)], make_config(), 0, 2, 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
