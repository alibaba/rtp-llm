import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.block_map import get_group_tags_for_layers
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3_next import (
    Qwen3NextGatedDeltaNetDecode,
    Qwen3NextMetadata,
    _maybe_write_cp_cache_store,
    _write_cp_cache_store,
)


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
    def test_qwen3_next_cuda_graph_uses_narrow_block_map_view(self):
        block_map = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        attention_inputs = SimpleNamespace(
            is_cuda_graph=True,
            kv_cache_kernel_block_id_device=block_map,
        )
        decode = object.__new__(Qwen3NextGatedDeltaNetDecode)

        narrowed = decode._get_fla_block_map(attention_inputs)

        self.assertEqual(narrowed.shape, (3, 1))
        self.assertEqual(narrowed.stride(0), block_map.stride(0))
        self.assertEqual(narrowed[:, 0].tolist(), [0, 4, 8])

    def test_qwen3_next_non_graph_keeps_full_block_map(self):
        block_map = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        attention_inputs = SimpleNamespace(
            is_cuda_graph=False,
            kv_cache_kernel_block_id_device=block_map,
        )
        decode = object.__new__(Qwen3NextGatedDeltaNetDecode)

        self.assertIs(decode._get_fla_block_map(attention_inputs), block_map)

    def test_cp_cache_store_uses_each_layer_tag_metadata(self):
        expected = {}
        layer_inputs = {}
        for index, tag in enumerate(("full", "linear0", "linear1"), start=1):
            actual_lengths = torch.tensor([index], dtype=torch.int32)
            prefix_lengths = torch.tensor([index + 10], dtype=torch.int32)
            block_ids = torch.tensor([[index + 20]], dtype=torch.int32)
            cache_store_inputs = SimpleNamespace(tag=tag)
            kv_cache = SimpleNamespace(tag=tag)
            layer_inputs[tag] = (
                SimpleNamespace(
                    context_parallel_info=SimpleNamespace(
                        prefill_actual_input_lengths_cpu=actual_lengths
                    ),
                    prefix_lengths=prefix_lengths,
                    kv_cache_block_id=block_ids,
                    cache_store_inputs=cache_store_inputs,
                ),
                kv_cache,
            )
            expected[tag] = (
                actual_lengths,
                prefix_lengths,
                block_ids,
                cache_store_inputs,
                kv_cache,
            )

        with patch(
            "rtp_llm.models_py.model_desc.qwen3_next.compute_ops.write_cache_store"
        ) as write_cache_store:
            for tag in ("full", "linear0", "linear1"):
                _write_cp_cache_store(*layer_inputs[tag])

        self.assertEqual(write_cache_store.call_count, 3)
        for call, tag in zip(
            write_cache_store.call_args_list, ("full", "linear0", "linear1")
        ):
            for actual, wanted in zip(call.args, expected[tag]):
                self.assertIs(actual, wanted)

    def test_cp_cache_store_skips_layer_without_store_inputs(self):
        attention_inputs = SimpleNamespace(cache_store_inputs=None)

        with patch(
            "rtp_llm.models_py.model_desc.qwen3_next.compute_ops.write_cache_store"
        ) as write_cache_store:
            _write_cp_cache_store(attention_inputs, SimpleNamespace(tag="linear0"))

        write_cache_store.assert_not_called()

    def test_cp_cache_store_requires_context_parallel_metadata(self):
        attention_inputs = SimpleNamespace(
            cache_store_inputs=SimpleNamespace(tag="linear0"),
            context_parallel_info=None,
        )

        with self.assertRaisesRegex(
            RuntimeError, "CP cache store requires context_parallel_info"
        ):
            _write_cp_cache_store(attention_inputs, SimpleNamespace(tag="linear0"))

    def test_non_cp_linear_attention_does_not_write_cache_store(self):
        attention_inputs = SimpleNamespace(
            cache_store_inputs=SimpleNamespace(tag="linear0"),
            context_parallel_info=SimpleNamespace(
                prefill_actual_input_lengths_cpu=torch.tensor([1], dtype=torch.int32)
            ),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            kv_cache_block_id=torch.tensor([[1]], dtype=torch.int32),
        )

        with patch(
            "rtp_llm.models_py.model_desc.qwen3_next.compute_ops.write_cache_store"
        ) as write_cache_store:
            _maybe_write_cp_cache_store(
                attention_inputs,
                SimpleNamespace(tag="linear0"),
                Qwen3NextMetadata(),
            )

        write_cache_store.assert_not_called()

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
