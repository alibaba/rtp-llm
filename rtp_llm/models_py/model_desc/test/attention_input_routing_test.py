import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
from rtp_llm.models_py.modules.factory.attention import common as attention_common


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
        # Metadata (lengths/prefix/block table) now travels inside the
        # PyCacheStoreInputs object built by C++ prepareWriteCacheParams (covered
        # per-field by PrepareWriteCacheParamsTest.cc), so the python contract to
        # lock here is object identity: each layer's writer.write must receive
        # exactly that layer's tag-local inputs and kv_cache, not copies or
        # another tag's objects.
        layer_inputs = {}
        for tag in ("full", "linear0", "linear1"):
            cache_store_inputs = SimpleNamespace(tag=tag)
            kv_cache = SimpleNamespace(tag=tag)
            cache_store_writer = Mock()
            layer_inputs[tag] = (
                SimpleNamespace(
                    is_prefill=True,
                    cache_store_inputs=cache_store_inputs,
                    cache_store_writer=cache_store_writer,
                ),
                kv_cache,
            )

        for tag in ("full", "linear0", "linear1"):
            attention_inputs, kv_cache = layer_inputs[tag]
            _write_cp_cache_store(attention_inputs, kv_cache)
            attention_inputs.cache_store_writer.write.assert_called_once()
            passed_inputs, passed_kv_cache = (
                attention_inputs.cache_store_writer.write.call_args.args
            )
            self.assertIs(passed_inputs, attention_inputs.cache_store_inputs)
            self.assertIs(passed_kv_cache, kv_cache)

    def test_fmha_factory_path_publishes_through_writer(self):
        # FMHA publish route: create_write_cache_store_impl -> apply_write_cache_store
        # -> writer.write. Locks the forward chain by object identity.
        cache_store_inputs = SimpleNamespace(tag="full")
        kv_cache = SimpleNamespace(tag="full")
        cache_store_writer = Mock()
        attention_inputs = SimpleNamespace(
            is_prefill=True,
            cache_store_inputs=cache_store_inputs,
            cache_store_writer=cache_store_writer,
        )

        write_impl = attention_common.create_write_cache_store_impl(attention_inputs)
        self.assertIsNotNone(write_impl)

        attention_common.apply_write_cache_store(write_impl, attention_inputs, kv_cache)
        cache_store_writer.write.assert_called_once()
        passed_inputs, passed_kv_cache = cache_store_writer.write.call_args.args
        self.assertIs(passed_inputs, cache_store_inputs)
        self.assertIs(passed_kv_cache, kv_cache)

        # Null kv_cache: publishes nothing, op stays reusable.
        cache_store_writer.reset_mock()
        attention_common.apply_write_cache_store(write_impl, attention_inputs, None)
        cache_store_writer.write.assert_not_called()

    def test_cp_cache_store_skips_when_pair_incomplete(self):
        # PyWrappedModel attaches cache_store_inputs and cache_store_writer together only
        # when the C++ boundary accepts eligibility, so any half pair reaching python is a
        # contract break by an upstream that bypassed prepareWriteCacheParams. No pairing
        # below may produce a write op or a writer call; half pairs must additionally
        # leave a WARN signal (both-None is the normal non-PD path and stays silent).
        cases = [
            ("neither", None, None, False),
            ("inputs_only", SimpleNamespace(tag="linear0"), None, True),
            ("writer_only", None, Mock(), True),
        ]
        for label, cache_store_inputs, cache_store_writer, expect_warn in cases:
            with self.subTest(pairing=label):
                attention_inputs = SimpleNamespace(
                    is_prefill=True,
                    cache_store_inputs=cache_store_inputs,
                    cache_store_writer=cache_store_writer,
                )

                # Isolate the module's throttle state here rather than exposing a reset
                # hook from production code.
                with patch.dict(
                    attention_common._half_pair_warn_counts, {}, clear=True
                ), patch.object(attention_common.logger, "warning") as warning:
                    # Observable skip decision through the public factory surface.
                    self.assertIsNone(
                        attention_common.create_write_cache_store_impl(attention_inputs)
                    )
                    _write_cp_cache_store(
                        attention_inputs, SimpleNamespace(tag="linear0")
                    )

                    if cache_store_writer is not None:
                        cache_store_writer.write.assert_not_called()
                    self.assertEqual(warning.called, expect_warn)

    def test_write_cp_cache_store_helper_skips_when_not_prefill(self):
        # Helper-level guard: even with a complete pair attached, decode passes must never
        # publish KV. The equivalent forward-level guarantee comes from PyWrappedModel's
        # prepareWriteCacheParams (warmup / decode-only exit early); this test locks the
        # helper contract in isolation, matching the helper's name.
        cache_store_writer = Mock()
        attention_inputs = SimpleNamespace(
            is_prefill=False,
            cache_store_inputs=SimpleNamespace(tag="linear0"),
            cache_store_writer=cache_store_writer,
        )

        _write_cp_cache_store(attention_inputs, SimpleNamespace(tag="linear0"))

        cache_store_writer.write.assert_not_called()

    def test_non_cp_linear_attention_does_not_write_cache_store(self):
        attention_inputs = SimpleNamespace(
            is_prefill=True,
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
