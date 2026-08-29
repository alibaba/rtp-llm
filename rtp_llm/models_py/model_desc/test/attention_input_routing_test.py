import unittest
from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.block_map import get_group_tags_for_layers
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel
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


class DuplicateSparseTagMapping(Mapping[str, object]):
    def __init__(self):
        self.values = {"default": object(), "indexer_kv": object()}

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("default", "indexer_kv", "indexer_kv"))

    def __len__(self) -> int:
        return 3


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
    def test_generic_sparse_mla_prepares_only_exact_semantic_groups(self):
        model = object.__new__(GenericMoeModel)
        model.__dict__["config"] = SimpleNamespace(
            attn_config=SimpleNamespace(is_sparse=True, use_mla=True)
        )

        self.assertEqual(model._get_fmha_group_tags(), ["default", "indexer_kv"])

    def test_generic_dense_mla_keeps_scalar_group_selection(self):
        model = object.__new__(GenericMoeModel)
        model.__dict__["config"] = SimpleNamespace(
            attn_config=SimpleNamespace(is_sparse=False, use_mla=True)
        )

        self.assertIsNone(model._get_fmha_group_tags())

    def test_generic_sparse_non_mla_keeps_scalar_group_selection(self):
        model = object.__new__(GenericMoeModel)
        model.__dict__["config"] = SimpleNamespace(
            attn_config=SimpleNamespace(is_sparse=True, use_mla=False)
        )

        self.assertIsNone(model._get_fmha_group_tags())

    def test_generic_sparse_mla_rejects_invalid_raw_tags_before_factory(self):
        model = object.__new__(GenericMoeModel)
        model.__dict__.update(
            config=SimpleNamespace(
                attn_config=SimpleNamespace(is_sparse=True, use_mla=True)
            ),
            parallelism_config=object(),
            weight=object(),
            fmha_config=object(),
        )
        invalid_mappings = (
            {"default": object(), "indexer_kv": object(), "extra": object()},
            {"default": object()},
            {"default": object(), "wrong": object()},
            DuplicateSparseTagMapping(),
        )

        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl"
        ) as factory:
            for attention_inputs in invalid_mappings:
                with self.subTest(tags=list(attention_inputs)):
                    with self.assertRaisesRegex(RuntimeError, "exactly.*tags"):
                        model.prepare_fmha_impl(
                            SimpleNamespace(attention_inputs=attention_inputs)
                        )

        factory.assert_not_called()

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
        layer_inputs = {}
        for tag in ("full", "linear0", "linear1"):
            cache_store_inputs = SimpleNamespace(tag=tag)
            kv_cache = SimpleNamespace(tag=tag)
            cache_store_writer = Mock()
            layer_inputs[tag] = (
                SimpleNamespace(
                    cache_store_inputs=cache_store_inputs,
                    cache_store_writer=cache_store_writer,
                ),
                kv_cache,
            )

        for tag in ("full", "linear0", "linear1"):
            attention_inputs, kv_cache = layer_inputs[tag]
            _write_cp_cache_store(attention_inputs, kv_cache)
            attention_inputs.cache_store_writer.write.assert_called_once_with(
                attention_inputs.cache_store_inputs, kv_cache
            )

    def test_cp_cache_store_skips_layer_without_store_inputs(self):
        cache_store_writer = Mock()
        attention_inputs = SimpleNamespace(
            cache_store_inputs=None, cache_store_writer=cache_store_writer
        )

        _write_cp_cache_store(attention_inputs, SimpleNamespace(tag="linear0"))

        cache_store_writer.write.assert_not_called()

    def test_cp_cache_store_skips_layer_without_writer(self):
        attention_inputs = SimpleNamespace(
            cache_store_inputs=SimpleNamespace(tag="linear0"),
            cache_store_writer=None,
        )

        _write_cp_cache_store(attention_inputs, SimpleNamespace(tag="linear0"))

    def test_non_cp_linear_attention_does_not_write_cache_store(self):
        attention_inputs = SimpleNamespace(
            cache_store_inputs=SimpleNamespace(tag="linear0"),
            cache_store_writer=Mock(),
            context_parallel_info=SimpleNamespace(
                prefill_actual_input_lengths_cpu=torch.tensor([1], dtype=torch.int32)
            ),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            kv_cache_block_id=torch.tensor([[1]], dtype=torch.int32),
        )

        _maybe_write_cp_cache_store(
            attention_inputs,
            SimpleNamespace(tag="linear0"),
            Qwen3NextMetadata(),
        )

        attention_inputs.cache_store_writer.write.assert_not_called()

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
