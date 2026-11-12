import unittest
from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.models_py.utils.expert_load_monitor import (
    bind_model_expert_load,
    collect_eplb_layers,
)


class RoutedLayer(nn.Module):
    def __init__(self, layer_id, mapped=True):
        super().__init__()
        self.expert_load_layer_id = layer_id
        self.plan = SimpleNamespace(layer_id=layer_id) if mapped else None
        self.context = None

    def get_eplb_layer(self):
        return self.plan

    def bind_expert_load(self, context):
        self.context = context

    def forward(self, physical_ids):
        self.context.expert_token_counts.add_(
            torch.bincount(physical_ids, minlength=4).to(torch.int32)
        )


class ModelBindingTest(unittest.TestCase):
    def test_discovery_is_independent_of_model_name_and_nesting(self):
        for name in ("GLM5", "HY4", "AnotherMoeModel"):
            model = type(name, (nn.Module,), {})()
            first = RoutedLayer(1)
            second = RoutedLayer(3)
            model.layers = nn.ModuleList([nn.Identity(), nn.Sequential(first), second])
            model.shared_reference = first
            self.assertEqual(collect_eplb_layers(model), [first.plan, second.plan])

    def test_monitor_only_uses_actual_layer_ids_and_original_storage(self):
        model = nn.Sequential(RoutedLayer(1, mapped=False), RoutedLayer(3, mapped=False))
        counts = torch.zeros((4, 4), dtype=torch.int32)
        with torch.inference_mode():
            bind_model_expert_load(model, counts)
        self.assertFalse(model[0].context.expert_token_counts.is_inference())
        self.assertEqual(
            model[0].context.expert_token_counts.data_ptr(), counts[1].data_ptr()
        )
        self.assertEqual(torch.count_nonzero(counts), 0)
        model[0](torch.tensor([0, 1, 1]))
        model[1](torch.tensor([2, 2, 3]))
        self.assertEqual(
            counts.tolist(), [[0] * 4, [1, 2, 0, 0], [0] * 4, [0, 0, 2, 1]]
        )
        counts.zero_()
        model[1](torch.tensor([3]))
        self.assertEqual(counts[3].tolist(), [0, 0, 0, 1])

    def test_runtime_rejects_missing_maps_and_duplicate_ids(self):
        for model in (
            nn.Sequential(RoutedLayer(0, False)),
            nn.Sequential(RoutedLayer(1), RoutedLayer(1)),
        ):
            with self.assertRaises(ValueError):
                collect_eplb_layers(model)
        with self.assertRaises(NotImplementedError):
            collect_eplb_layers(nn.Sequential(nn.Identity()))

    def test_invalid_buffer_rejected_before_any_layer_is_bound(self):
        model = nn.Sequential(RoutedLayer(0), RoutedLayer(3))
        for counts in (
            torch.zeros(2, 4, dtype=torch.int32),
            torch.zeros(4, 4),
            torch.zeros(4, dtype=torch.int32),
            torch.zeros(4, 4, dtype=torch.int32).T,
        ):
            with self.assertRaises(ValueError):
                bind_model_expert_load(model, counts)
            self.assertIsNone(model[0].context)
            self.assertIsNone(model[1].context)


if __name__ == "__main__":
    unittest.main()
