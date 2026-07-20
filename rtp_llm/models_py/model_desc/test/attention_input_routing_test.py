import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from rtp_llm.models_py.model_desc.block_map import (
    get_fmha_params,
    get_group_tags_for_layers,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase


class FakeKVCache:
    def __init__(self, layer_tags: list[list[str]]):
        self.layer_tags = layer_tags

    def get_layer_cache_groups(self, layer_idx: int):
        return [SimpleNamespace(tag=tag) for tag in self.layer_tags[layer_idx]]


class RoutingModel(GptModelBase):
    def __init__(self, fmha_group_tags: list[str] | None):
        nn.Module.__init__(self)
        self.config = object()
        self.parallelism_config = object()
        self.weight = object()
        self.fmha_config = object()
        self.fmha_group_tags = fmha_group_tags

    def _get_fmha_group_tags(self) -> list[str] | None:
        return self.fmha_group_tags


class AttentionInputRoutingTest(unittest.TestCase):
    def test_fmha_params_preserves_only_single_group_fast_path(self):
        params = object()
        impl = SimpleNamespace(fmha_params=params)

        self.assertIs(get_fmha_params(impl), params)
        self.assertIsNone(get_fmha_params({"full": impl, "aux": impl}))

    def test_get_group_tags_for_model_selected_layers(self):
        kv_cache = FakeKVCache([["full"], ["linear0"], ["linear1"], ["full", "aux"]])

        self.assertEqual(get_group_tags_for_layers(kv_cache, [0, 3]), ["full", "aux"])

    def test_prepare_fmha_impl_only_for_model_selected_tags(self):
        inputs_by_tag = {
            "full": object(),
            "linear0": object(),
            "linear1": object(),
        }
        inputs = SimpleNamespace(attention_inputs=inputs_by_tag)
        model = RoutingModel(["full"])

        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
            side_effect=lambda _config, _parallelism_config, _weight, group_inputs, _fmha_config, _is_cuda_graph: (
                group_inputs
            ),
        ) as factory:
            fmha_impl = model.prepare_fmha_impl(inputs, is_cuda_graph=True)

        self.assertEqual(fmha_impl, {"full": inputs_by_tag["full"]})
        factory.assert_called_once()

    def test_default_model_prepares_every_tag(self):
        inputs_by_tag = {"group0": object(), "group1": object()}
        inputs = SimpleNamespace(attention_inputs=inputs_by_tag)
        model = RoutingModel(None)

        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
            side_effect=lambda _config, _parallelism_config, _weight, group_inputs, _fmha_config, _is_cuda_graph: (
                group_inputs
            ),
        ) as factory:
            fmha_impl = model.prepare_fmha_impl(inputs)

        self.assertEqual(fmha_impl, inputs_by_tag)
        self.assertEqual(factory.call_count, 2)


if __name__ == "__main__":
    unittest.main()
