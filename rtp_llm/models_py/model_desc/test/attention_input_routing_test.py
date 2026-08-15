import unittest
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
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    resolve_mla_use_fast_path,
)
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention
from rtp_llm.ops.compute_ops import PyAttentionInputs


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
    def test_mla_fast_path_uses_bound_total_kv_length(self):
        for is_prefill, max_length, cp_enabled, expected in (
            (True, 4, False, True),
            (True, 5, False, False),
            (False, 4, False, False),
            (True, 4, True, False),
        ):
            with self.subTest(
                is_prefill=is_prefill,
                max_length=max_length,
                cp_enabled=cp_enabled,
            ):
                attention_inputs = PyAttentionInputs()
                attention_inputs.is_prefill = is_prefill
                attention_inputs.context_total_kv_length = max_length
                parallelism = SimpleNamespace(
                    prefill_cp_config=SimpleNamespace(is_enabled=lambda: cp_enabled)
                )
                self.assertEqual(
                    resolve_mla_use_fast_path(
                        SimpleNamespace(indexer_topk=4),
                        attention_inputs,
                        parallelism,
                    ),
                    expected,
                )

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
        group_inputs = {
            "full": PyAttentionInputs(),
            "linear0": PyAttentionInputs(),
            "linear1": PyAttentionInputs(),
        }
        inputs = SimpleNamespace(attention_inputs=group_inputs)
        model = RoutingModel(["full"])

        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
            side_effect=lambda _config, _parallelism_config, _weight, group_inputs, _fmha_config, _is_cuda_graph: (
                group_inputs
            ),
        ) as factory:
            fmha_impl = model.prepare_fmha_impl(inputs, is_cuda_graph=True)

        self.assertEqual(fmha_impl, {"full": group_inputs["full"]})
        factory.assert_called_once()

    def test_default_model_prepares_every_tag(self):
        group_inputs = {"group0": PyAttentionInputs(), "group1": PyAttentionInputs()}
        inputs = SimpleNamespace(attention_inputs=group_inputs)
        model = RoutingModel(None)

        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
            side_effect=lambda _config, _parallelism_config, _weight, group_inputs, _fmha_config, _is_cuda_graph: (
                group_inputs
            ),
        ) as factory:
            fmha_impl = model.prepare_fmha_impl(inputs)

        self.assertEqual(fmha_impl, group_inputs)
        self.assertEqual(factory.call_count, 2)

    def test_sparse_model_prepares_group_local_impls_with_one_path_decision(self):
        group_inputs = {
            "default": PyAttentionInputs(),
            "indexer_kv": PyAttentionInputs(),
        }
        inputs = SimpleNamespace(attention_inputs=group_inputs)
        model = object.__new__(GenericMoeModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            attn_config=SimpleNamespace(is_sparse=True),
            getAttentionConfigs=lambda _tp_size: object(),
        )
        model.parallelism_config = SimpleNamespace(get_attn_tp_size=lambda: 1)
        model.weight = object()
        model.fmha_config = object()

        def return_tagged_inputs(
            _config,
            _parallelism,
            _weight,
            tagged_inputs,
            _fmha,
            _graph,
            **_kwargs,
        ):
            return tagged_inputs

        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.resolve_mla_use_fast_path",
            return_value=False,
        ) as resolve_fast_path, patch(
            "rtp_llm.models_py.model_desc.generic_moe.AttnImplFactory.get_fmha_impl",
            side_effect=return_tagged_inputs,
        ) as factory:
            fmha_impl = model.prepare_fmha_impl(inputs, is_cuda_graph=True)

        self.assertEqual(fmha_impl, group_inputs)
        resolve_fast_path.assert_called_once()
        self.assertEqual(factory.call_count, 2)
        for call in factory.call_args_list:
            self.assertFalse(call.kwargs["mla_use_fast_path"])

    def test_sparse_model_builds_fmha_routes_once_and_looks_up_each_layer_cache(self):
        model = object.__new__(GenericMoeModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(attn_config=SimpleNamespace(is_sparse=True))
        model.layer_num = 2
        model.kv_cache = object()
        model.embed_tokens = Mock(return_value=torch.ones((1, 2)))
        layers = [Mock(), Mock()]
        for layer in layers:
            layer.return_value = SimpleNamespace(
                hidden_states=torch.ones((1, 2)),
                residual=torch.zeros((1, 2)),
            )
        model.layers = layers
        model.norm = Mock(return_value=(torch.ones((1, 2)), None))
        fmha_impl = {"default": object(), "indexer_kv": object()}

        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.select_fmha_impl_for_tag",
            side_effect=lambda routes, tag: routes[tag],
        ) as select_route, patch(
            "rtp_llm.models_py.model_desc.generic_moe.get_layer_caches_for_tags",
            side_effect=lambda _cache, layer_id, tags: {
                tag: f"{tag}-{layer_id}" for tag in tags
            },
        ) as get_layer_caches:
            model.forward(SimpleNamespace(input_ids=torch.tensor([1])), fmha_impl)

        self.assertEqual(select_route.call_count, 2)
        self.assertEqual(get_layer_caches.call_count, 2)
        self.assertEqual(
            [call.args[1] for call in get_layer_caches.call_args_list], [0, 1]
        )
        self.assertIs(layers[0].call_args.args[2], layers[1].call_args.args[2])
        self.assertEqual(
            layers[0].call_args.kwargs["kv_cache"],
            {"default": "default-0", "indexer_kv": "indexer_kv-0"},
        )
        self.assertEqual(
            layers[1].call_args.kwargs["kv_cache"],
            {"default": "default-1", "indexer_kv": "indexer_kv-1"},
        )

    def test_sparse_indexer_publishes_group_cache_after_kernel_write(self):
        attention = object.__new__(MlaAttention)
        nn.Module.__init__(attention)
        attention.q_lora_rank = 0

        for topk_indices in (torch.tensor([3]), None):
            with self.subTest(topk_indices=topk_indices):
                events = []
                indexer_cache = SimpleNamespace(tag="indexer_kv")

                def run_indexer(*args, **kwargs):
                    events.append(("indexer_write", args[2]))
                    return topk_indices

                input_lengths = object()
                prefix_lengths = object()
                block_ids = object()
                cache_store_inputs = object()
                cache_store_writer = Mock()
                cache_store_writer.write.side_effect = (
                    lambda _inputs, cache: events.append(("cache_store", cache))
                )
                indexer_attn_inputs = SimpleNamespace(
                    is_prefill=True,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    kv_cache_block_id=block_ids,
                    cache_store_inputs=cache_store_inputs,
                    cache_store_writer=cache_store_writer,
                )

                attention.indexer = Mock(side_effect=run_indexer)
                indexer_impl = SimpleNamespace(
                    fmha_params=object(),
                    attn_inputs=indexer_attn_inputs,
                    cp_params=None,
                    is_sparse=lambda: False,
                )

                result = attention._run_sparse_indexer(
                    hidden_states=torch.empty(0),
                    q_c=None,
                    q_view=torch.empty(0),
                    kv_cache=indexer_cache,
                    fmha_impl=indexer_impl,
                )

                self.assertIs(result, topk_indices)
                self.assertEqual(
                    events,
                    [
                        ("indexer_write", indexer_cache),
                        ("cache_store", indexer_cache),
                    ],
                )
                cache_store_writer.write.assert_called_once_with(
                    cache_store_inputs,
                    indexer_cache,
                )

    def test_sparse_indexer_does_not_publish_cache_during_decode(self):
        attention = object.__new__(MlaAttention)
        nn.Module.__init__(attention)
        attention.q_lora_rank = 0
        attention.indexer = Mock(return_value=torch.tensor([3]))
        cache_store_writer = Mock()
        indexer_impl = SimpleNamespace(
            fmha_params=object(),
            attn_inputs=SimpleNamespace(
                is_prefill=False,
                cache_store_inputs=None,
                cache_store_writer=cache_store_writer,
            ),
            cp_params=None,
            is_sparse=lambda: False,
        )

        attention._run_sparse_indexer(
            hidden_states=torch.empty(0),
            q_c=None,
            q_view=torch.empty(0),
            kv_cache=SimpleNamespace(tag="indexer_kv"),
            fmha_impl=indexer_impl,
        )

        cache_store_writer.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
