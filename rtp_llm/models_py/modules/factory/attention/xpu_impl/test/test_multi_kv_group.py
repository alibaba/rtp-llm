"""Unit test: XPU attention must fail-fast for hybrid / multi-KV-group models.

Hybrid KV models (GLA, sliding-window, etc.) map layers to different KV cache
groups and expose one block table per group.  The XPU attention path builds a
single block table and reuses it across every layer, so a multi-group model
would read/write the wrong KV blocks on later layers.  ``support()`` must reject
these configs explicitly instead of silently using the wrong block table.

CPU-runnable (no XPU device required).
"""

from unittest import TestCase, main, SkipTest

import torch

from rtp_llm.models_py.modules.factory.attention.xpu_impl.test._import_guard import (
    skip_or_fail_on_missing_import,
)

try:
    from rtp_llm.ops import RopeStyle
    from rtp_llm.models_py.modules.factory.attention.xpu_impl.vllm_flash_attn import (
        _reject_multi_kv_group,
        XpuVllmPrefillImpl,
        XpuVllmDecodeImpl,
    )
    _IMPORT_OK = True
    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover - import guard
    _IMPORT_OK = False
    _IMPORT_ERR = _e


class _FakeRopeConfig:
    def __init__(self, style):
        self.style = style
        self.dim = 0
        self.base = 10000.0
        self.scale = 1.0
        self.is_neox_style = True


class _FakeAttnConfigs:
    def __init__(self):
        self.rope_config = _FakeRopeConfig(RopeStyle.No)
        self.need_rope_kv_cache = False
        self.head_num = 4
        self.kv_head_num = 4
        self.size_per_head = 16
        self.kv_cache_dtype = None


class _FakeAttnInputs:
    def __init__(self, is_prefill, layer_to_group=None, block_ids_by_group=None):
        self.is_prefill = is_prefill
        self.kv_cache_layer_to_group = layer_to_group
        self.kv_cache_kernel_block_id_by_group = block_ids_by_group


class TestRejectMultiKvGroup(TestCase):
    def setUp(self):
        skip_or_fail_on_missing_import(self, _IMPORT_OK, _IMPORT_ERR)

    def test_single_group_layer_to_group_ok(self):
        inputs = _FakeAttnInputs(
            is_prefill=False,
            layer_to_group=torch.zeros(4, dtype=torch.int32),
        )
        # Must not raise for a homogeneous (single-group) model.
        _reject_multi_kv_group(inputs, "decode")

    def test_none_fields_ok(self):
        inputs = _FakeAttnInputs(is_prefill=False)
        _reject_multi_kv_group(inputs, "decode")

    def test_empty_layer_to_group_ok(self):
        inputs = _FakeAttnInputs(
            is_prefill=False,
            layer_to_group=torch.zeros(0, dtype=torch.int32),
        )
        _reject_multi_kv_group(inputs, "decode")

    def test_multi_group_layer_to_group_rejected(self):
        # kv_cache_layer_to_group=[0, 1]: two groups -> layer-specific tables.
        inputs = _FakeAttnInputs(
            is_prefill=False,
            layer_to_group=torch.tensor([0, 1], dtype=torch.int32),
        )
        with self.assertRaises(NotImplementedError):
            _reject_multi_kv_group(inputs, "decode")

    def test_multi_group_distinct_block_tables_rejected(self):
        # Two groups whose block tables differ: group 0 -> [10, 11],
        # group 1 -> [20, 21].  Reusing a single table would corrupt KV.
        group0 = torch.tensor([[10, 11]], dtype=torch.int32)
        group1 = torch.tensor([[20, 21]], dtype=torch.int32)
        inputs = _FakeAttnInputs(
            is_prefill=False,
            layer_to_group=torch.tensor([0, 1], dtype=torch.int32),
            block_ids_by_group=[group0, group1],
        )
        with self.assertRaises(NotImplementedError):
            _reject_multi_kv_group(inputs, "decode")


class TestSupportRejectsMultiGroup(TestCase):
    def setUp(self):
        skip_or_fail_on_missing_import(self, _IMPORT_OK, _IMPORT_ERR)

    def test_decode_support_rejects_multi_group(self):
        group0 = torch.tensor([[10, 11]], dtype=torch.int32)
        group1 = torch.tensor([[20, 21]], dtype=torch.int32)
        inputs = _FakeAttnInputs(
            is_prefill=False,
            layer_to_group=torch.tensor([0, 1], dtype=torch.int32),
            block_ids_by_group=[group0, group1],
        )
        with self.assertRaises(NotImplementedError):
            XpuVllmDecodeImpl.support(_FakeAttnConfigs(), inputs)

    def test_prefill_support_rejects_multi_group(self):
        group0 = torch.tensor([[10, 11]], dtype=torch.int32)
        group1 = torch.tensor([[20, 21]], dtype=torch.int32)
        inputs = _FakeAttnInputs(
            is_prefill=True,
            layer_to_group=torch.tensor([0, 1], dtype=torch.int32),
            block_ids_by_group=[group0, group1],
        )
        with self.assertRaises(NotImplementedError):
            XpuVllmPrefillImpl.support(_FakeAttnConfigs(), inputs)


if __name__ == "__main__":
    main()
