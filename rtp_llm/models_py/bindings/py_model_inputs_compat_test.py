import ast
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    get_layer_cache_for_tag,
    get_layer_caches_for_tags,
    select_attention_inputs_for_layer,
    select_fmha_impl_for_tag,
)
from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.utils.kvcache import SingleGroupKVCacheAdapter
from rtp_llm.ops import HybridAttentionConfig, HybridAttentionType
from rtp_llm.ops.compute_ops import (
    CacheStoreWriter,
    KVCache,
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)


class _RoutingCache:
    def __init__(self, layer_tags: list[list[str]]) -> None:
        self._layer_tags = layer_tags

    def get_layer_cache_groups(self, layer_id: int) -> list[LayerKVCache]:
        return [
            LayerKVCache(torch.ones(1), 1, layer_id, tag)
            for tag in self._layer_tags[layer_id]
        ]


class _ConcreteRoutingCache:
    def __init__(self, caches: list[LayerKVCache]) -> None:
        self._caches = caches

    def get_layer_cache_groups(self, layer_id: int) -> list[LayerKVCache]:
        if layer_id != 0:
            raise RuntimeError(f"invalid layer {layer_id}")
        return self._caches


class PyModelInputsCompatTest(unittest.TestCase):
    def test_hybrid_attention_config_has_explicit_constructors(self) -> None:
        default_config = HybridAttentionConfig()
        self.assertFalse(default_config.enable_hybrid_attention)
        self.assertEqual(default_config.hybrid_attention_types, [])

        attention_types = [HybridAttentionType.NONE, HybridAttentionType.LINEAR]
        config = HybridAttentionConfig(True, attention_types)
        self.assertTrue(config.enable_hybrid_attention)
        self.assertEqual(config.hybrid_attention_types, attention_types)

        with self.assertRaises(TypeError):
            HybridAttentionConfig(True, True, attention_types)

    def test_sparse_routes_are_selected_by_tag_not_topology_order(self) -> None:
        default_cache = LayerKVCache(torch.ones(1), 64, 0, "default")
        indexer_cache = LayerKVCache(torch.ones(1) * 2, 64, 0, "indexer_kv")
        cache = _ConcreteRoutingCache([indexer_cache, default_cache])

        self.assertIs(get_layer_cache_for_tag(cache, 0, "default"), default_cache)
        self.assertIs(get_layer_cache_for_tag(cache, 0, "indexer_kv"), indexer_cache)
        selected = get_layer_caches_for_tags(cache, 0, ("default", "indexer_kv"))
        self.assertEqual(
            selected, {"default": default_cache, "indexer_kv": indexer_cache}
        )
        fmha = {"indexer_kv": object(), "default": object()}
        self.assertIs(select_fmha_impl_for_tag(fmha, "default"), fmha["default"])
        self.assertIs(select_fmha_impl_for_tag(fmha, "indexer_kv"), fmha["indexer_kv"])

    def test_sparse_routes_reject_missing_duplicate_and_wrong_types(self) -> None:
        duplicate = _ConcreteRoutingCache(
            [
                LayerKVCache(torch.ones(1), 64, 0, "indexer_kv"),
                LayerKVCache(torch.ones(1), 64, 0, "indexer_kv"),
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate KV cache tag"):
            get_layer_cache_for_tag(duplicate, 0, "indexer_kv")
        with self.assertRaisesRegex(RuntimeError, "missing"):
            select_fmha_impl_for_tag({"default": object()}, "indexer_kv")
        with self.assertRaisesRegex(RuntimeError, "requires a mapping"):
            select_fmha_impl_for_tag(object(), "indexer_kv")

        inputs = type("Inputs", (), {"attention_inputs": {"default": object()}})()
        with self.assertRaisesRegex(RuntimeError, "invalid value type"):
            get_attention_inputs_value(inputs)

    def test_cache_binding_stubs_match_runtime_members(self) -> None:
        stub_path = (
            Path(__file__).resolve().parents[2]
            / "ops"
            / "librtp_compute_ops"
            / "__init__.pyi"
        )
        module = ast.parse(stub_path.read_text())
        stub_classes = {
            node.name: node for node in module.body if isinstance(node, ast.ClassDef)
        }

        for class_name, runtime_class in (
            ("CacheStoreWriter", CacheStoreWriter),
            ("KVCache", KVCache),
            ("LayerKVCache", LayerKVCache),
            ("PyAttentionInputs", PyAttentionInputs),
        ):
            with self.subTest(class_name=class_name):
                stub_members = set()
                for node in stub_classes[class_name].body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        stub_members.add(node.name)
                    elif isinstance(node, ast.AnnAssign) and isinstance(
                        node.target, ast.Name
                    ):
                        stub_members.add(node.target.id)

                stub_members = {
                    name for name in stub_members if not name.startswith("_")
                }
                runtime_members = {
                    name for name in vars(runtime_class) if not name.startswith("_")
                }
                self.assertEqual(stub_members, runtime_members)

    def _attn_inputs(self, is_prefill: bool, input_length: int) -> PyAttentionInputs:
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = is_prefill
        attn_inputs.input_lengths = torch.tensor([input_length], dtype=torch.int32)
        return attn_inputs

    def test_default_constructor_then_assign_attention_inputs(self) -> None:
        attn_inputs = self._attn_inputs(is_prefill=True, input_length=3)

        inputs = PyModelInputs()
        inputs.attention_inputs = attn_inputs

        self.assertTrue(inputs.attention_inputs.is_prefill)
        self.assertEqual(3, inputs.attention_inputs.input_lengths.item())

    def test_model_outputs_only_exposes_hidden_states(self) -> None:
        hidden_states = torch.empty(0)
        outputs = PyModelOutputs(hidden_states)

        self.assertEqual(hidden_states.data_ptr(), outputs.hidden_states.data_ptr())
        self.assertFalse(hasattr(outputs, "params_ptr"))
        with self.assertRaises(TypeError):
            PyModelOutputs(hidden_states, {"full": None})

    def test_attention_inputs_field_updates_directly(self) -> None:
        inputs = PyModelInputs()
        inputs.attention_inputs = self._attn_inputs(is_prefill=False, input_length=2)

        self.assertFalse(inputs.attention_inputs.is_prefill)
        inputs.attention_inputs.is_prefill = True
        inputs.attention_inputs.input_lengths = torch.tensor([5], dtype=torch.int32)
        self.assertTrue(inputs.attention_inputs.is_prefill)
        self.assertEqual(5, inputs.attention_inputs.input_lengths.item())

    def test_kv_cache_is_runtime_constructed_and_read_only(self) -> None:
        with self.assertRaises(TypeError):
            KVCache()

        for old_name in (
            "kv_cache_base_by_layer",
            "kv_scale_base_by_layer",
            "seq_size_per_block",
            "kernel_seq_size_per_block",
            "num_kv_heads",
            "head_dim",
            "use_mla",
            "kv_lora_rank",
            "rope_head_dim",
            "layer_attn_types",
            "group_types",
            "group_seq_block_sizes",
            "group_kernel_seq_block_sizes",
            "layer_to_group_ids",
            "layer_tag_to_group_id",
            "get_layer_caches",
        ):
            self.assertFalse(hasattr(KVCache, old_name), old_name)

        for new_name in (
            "layer_count",
            "get_layer_cache",
            "get_layer_cache_groups",
            "get_seq_size_per_block",
            "get_kernel_seq_size_per_block",
        ):
            self.assertTrue(hasattr(KVCache, new_name), new_name)

    def test_layer_kv_cache_parameterized_constructor(self) -> None:
        base = torch.arange(8, dtype=torch.float16).reshape(2, 4)
        scale = torch.ones((2, 1), dtype=torch.float32)

        layer = LayerKVCache(
            base,
            16,
            layer_id=3,
            tag="full",
            kv_scale_base=scale,
        )

        self.assertEqual(base.data_ptr(), layer.kv_cache_base.data_ptr())
        self.assertEqual(scale.data_ptr(), layer.kv_scale_base.data_ptr())
        self.assertEqual(16, layer.seq_size_per_block)
        self.assertEqual(3, layer.layer_id)
        self.assertFalse(hasattr(layer, "group_id"))
        self.assertEqual("full", layer.tag)

    def test_indexer_uses_token_addressable_view_of_opaque_pool(self) -> None:
        indexer_op = object.__new__(IndexerOp)
        indexer_op.index_head_dim = 128
        indexer_op.block_size = 128
        indexer_op.blocksize = 64
        entry_elems = 132
        indexer_op.entry_elems = entry_elems
        indexer_op.page_elems = indexer_op.blocksize * entry_elems
        base = torch.empty((3, 64 * entry_elems), dtype=torch.uint8)
        layer = LayerKVCache(base, 64, layer_id=0, tag="indexer_kv")

        view = indexer_op._indexer_cache_view(layer)

        self.assertEqual(view.shape, (3, 64, entry_elems))
        self.assertEqual(view.data_ptr(), base.data_ptr())

    def test_indexer_rejects_invalid_opaque_pool_geometry(self) -> None:
        indexer_op = object.__new__(IndexerOp)
        indexer_op.index_head_dim = 128
        indexer_op.block_size = 128
        indexer_op.blocksize = 64
        row_width = 64 * 132
        indexer_op.entry_elems = 132
        indexer_op.page_elems = row_width

        with self.assertRaisesRegex(RuntimeError, "contiguous tensor"):
            indexer_op._indexer_cache_view(
                LayerKVCache(
                    torch.empty((row_width, 2), dtype=torch.uint8).transpose(0, 1),
                    64,
                    0,
                    "indexer_kv",
                )
            )
        with self.assertRaisesRegex(RuntimeError, "page geometry mismatch"):
            indexer_op._indexer_cache_view(
                LayerKVCache(
                    torch.empty((2, row_width), dtype=torch.uint8), 128, 0, "indexer_kv"
                )
            )
        with self.assertRaisesRegex(RuntimeError, "exact 2D kernel-page layout"):
            indexer_op._indexer_cache_view(
                LayerKVCache(
                    torch.empty((2, row_width - 1), dtype=torch.uint8),
                    64,
                    0,
                    "indexer_kv",
                )
            )

    def test_attention_inputs_mapping_is_selected_by_layer_tag(self) -> None:
        full = self._attn_inputs(is_prefill=False, input_length=1)
        linear = self._attn_inputs(is_prefill=False, input_length=1)
        full.kv_cache_kernel_block_id = torch.tensor([[10]], dtype=torch.int32)
        linear.kv_cache_kernel_block_id = torch.tensor([[20]], dtype=torch.int32)

        inputs = PyModelInputs()
        inputs.attention_inputs = {"full": full, "linear": linear}
        selected = select_attention_inputs_for_layer(
            inputs, _RoutingCache([["linear"]]), 0
        )

        self.assertEqual(20, selected.kv_cache_kernel_block_id.item())
        self.assertFalse(hasattr(selected, "kv_cache_kernel_block_id_by_group"))
        self.assertFalse(hasattr(selected, "kv_cache_layer_to_group"))

    def test_single_group_adapter_returns_native_layer_cache(self) -> None:
        tensors = [torch.zeros((2, 2, 1, 8, 4), dtype=torch.float16)]
        cache = SingleGroupKVCacheAdapter(tensors, 8)

        layer = cache.get_layer_cache(0, "default")
        self.assertIsInstance(layer, LayerKVCache)
        self.assertEqual(tensors[0].data_ptr(), layer.kv_cache_base.data_ptr())
        self.assertEqual(1, cache.layer_count)
        self.assertEqual(8, cache.get_seq_size_per_block("default"))
        self.assertEqual(
            ["default"], [item.tag for item in cache.get_layer_cache_groups(0)]
        )
        self.assertFalse(hasattr(cache, "get_layer_caches"))


if __name__ == "__main__":
    unittest.main()
