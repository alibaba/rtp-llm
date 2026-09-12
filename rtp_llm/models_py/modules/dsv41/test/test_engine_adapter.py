"""Standard adapter metadata and physical-pool contracts, not engine acceptance."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.deepseek_v41_model import (
    DeepSeekV41Model,
    _BatchedAttention,
    _pair_view,
    _read_pair,
    _request_ranges,
    _write_pair,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.dsv41.cache_layout import (
    PAIR_OWNERS,
    CacheLayout,
    CacheRegion,
    RegionSlot,
)
from rtp_llm.models_py.modules.dsv41.ced import ReplayConfig
from rtp_llm.models_py.modules.dsv41.compressor import PairCarry
from rtp_llm.models_py.modules.dsv41.inputs import V41ModelRows
from rtp_llm.ops.compute_ops import KVCache, KVCacheRegionName, PyModelOutputs


class RecordTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("embedding", torch.empty((0, 5120), dtype=torch.bfloat16))
        self.calls = 0
        self.output = None
        self.requests = None

    def forward(self, rows, requests, *, execution_mode, image_features):
        assert execution_mode == "full"
        self.calls += 1
        self.requests = requests
        for request in requests:
            request.context.completed_layers = set(range(40))
            for layer in PAIR_OWNERS:
                request.context.cache.owners[layer].pair = PairCarry.empty(
                    layer,
                    request.context.cache.request_id,
                    request.context.cache.identity,
                    request.context.end,
                )
        self.output = torch.full(
            (rows.token_ids.numel(), 5120), 0.5, dtype=torch.bfloat16
        )
        return SimpleNamespace(hidden_states=self.output)


def framework_fixture():
    layout = CacheLayout(cp_size=8, speculative_tokens=0, draft_enabled=False)
    model = DeepSeekV41Model.__new__(DeepSeekV41Model)
    nn.Module.__init__(model)
    model.layout = layout
    model.identity = ReplayConfig().cache_identity("a" * 40, layout)
    model.config = SimpleNamespace(hidden_size=5120)
    model._max_tokens = 2048
    model.max_tokens_per_rank = 2048
    model._pages, model._pair_pools, model._groups = {}, {}, {}
    model.target = RecordTarget()
    regions = [
        KVCacheRegionName.DSV41_GLOBAL_KV,
        KVCacheRegionName.DSV41_GLOBAL_KV,
        KVCacheRegionName.DSV41_INDEX_KV,
        KVCacheRegionName.DSV41_INDEX_KV,
        KVCacheRegionName.DSV41_PAIR_STATE,
        KVCacheRegionName.SWA_KV,
    ]
    width = int(KVCacheRegionName.DSV41_PAIR_STATE) + 1
    group_ids = [[-1] * width for _ in range(40)]
    raw = [[torch.empty(0) for _ in range(width)] for _ in range(40)]
    region_map = {
        CacheRegion.SWA: KVCacheRegionName.SWA_KV,
        CacheRegion.GLOBAL: KVCacheRegionName.DSV41_GLOBAL_KV,
        CacheRegion.INDEX_K: KVCacheRegionName.DSV41_INDEX_KV,
    }
    for page in layout.pages:
        layer, region = page.slot.owner_layer, region_map[page.slot.region]
        gid = (
            5
            if page.slot.region == CacheRegion.SWA
            else (0 if page.slot.region == CacheRegion.GLOBAL else 2) + int(layer == 20)
        )
        raw[layer][int(region)] = torch.zeros(
            (5, page.page_stride_bytes), dtype=torch.uint8
        )
        group_ids[layer][int(region)] = gid
    for layer in PAIR_OWNERS:
        raw[layer][int(KVCacheRegionName.DSV41_PAIR_STATE)] = torch.zeros(
            (5, 8704), dtype=torch.uint8
        )
        group_ids[layer][int(KVCacheRegionName.DSV41_PAIR_STATE)] = 4
    cache = KVCache()
    cache.seq_size_per_block = cache.kernel_seq_size_per_block = 128
    cache.group_region_names = regions
    cache.group_seq_size_per_block = [128, 128, 128, 128, 1024, 1024]
    cache.layer_region_to_group_id = group_ids
    cache.kv_cache_base_by_layer_region = raw
    model.initialize(SimpleNamespace(kv_cache=cache, is_speculative=False))
    return model, cache


def request_fixture(starts=(0,), lengths=(2,), ready=(False,), ids=(101,), fake=None):
    count = sum(lengths)
    rows = V41ModelRows(
        torch.ones(count, dtype=torch.int32),
        torch.full((count,), -1, dtype=torch.int32),
        torch.ones(count, dtype=torch.bool),
        torch.zeros((count, 3), dtype=torch.int32),
        torch.zeros((count, 3), dtype=torch.bool),
    )
    table = torch.tensor(
        [[1 + 2 * index, 2 + 2 * index] for index in range(len(starts))],
        dtype=torch.int32,
    ).reshape(len(starts), 2)
    attn = SimpleNamespace(
        is_prefill=True,
        is_target_verify=False,
        context_parallel_info=None,
        input_lengths=torch.tensor(lengths, dtype=torch.int32),
        prefix_lengths=torch.tensor(starts, dtype=torch.int32),
        sequence_lengths=torch.empty(0, dtype=torch.int32),
        kv_cache_kernel_block_id_device_by_group=[table.clone() for _ in range(6)],
    )
    inputs = SimpleNamespace(
        input_ids=rows.token_ids,
        v41_token_types=rows.token_types,
        v41_token_valid=rows.valid,
        engram_history_ids=rows.history_ids,
        engram_history_valid=rows.history_valid,
        request_id=torch.tensor(ids, dtype=torch.int64),
        v41_state_ready=torch.tensor(ready, dtype=torch.bool),
        v41_is_fake=torch.tensor(
            (False,) * len(starts) if fake is None else fake, dtype=torch.bool
        ),
        attention_inputs=attn,
        multimodal_features=None,
    )
    return inputs, rows


class EngineAdapterContractTest(unittest.TestCase):
    def test_standard_base_and_pool_views_keep_allocator_storage(self):
        model, cache = framework_fixture()
        self.assertIsInstance(model, GptModelBase)
        for slot, page in model._pages.items():
            region = {
                CacheRegion.SWA: KVCacheRegionName.SWA_KV,
                CacheRegion.GLOBAL: KVCacheRegionName.DSV41_GLOBAL_KV,
                CacheRegion.INDEX_K: KVCacheRegionName.DSV41_INDEX_KV,
            }[slot.region]
            self.assertEqual(
                page.data.data_ptr(),
                cache.get_raw_pool_tensor(slot.owner_layer, region).data_ptr(),
            )

    def test_changed_group_namespace_or_sliced_pages_are_rejected(self):
        model, cache = framework_fixture()
        cache.group_seq_size_per_block = [128] * 6
        with self.assertRaisesRegex(ValueError, "canonical CP block namespace"):
            model.initialize(SimpleNamespace(kv_cache=cache, is_speculative=False))
        model, cache = framework_fixture()
        raw = cache.kv_cache_base_by_layer_region
        raw[0][int(KVCacheRegionName.SWA_KV)] = torch.zeros(
            (5, 8448), dtype=torch.uint8
        )
        cache.kv_cache_base_by_layer_region = raw
        with self.assertRaisesRegex(ValueError, "compact pool geometry"):
            model.initialize(SimpleNamespace(kv_cache=cache, is_speculative=False))

    def test_global_hits_do_not_replace_execution_state(self):
        model, _ = framework_fixture()
        inputs, rows = request_fixture(starts=(128,))
        before = model._pages[RegionSlot(CacheRegion.SWA, 0)].data.clone()
        with self.assertRaisesRegex(ValueError, "restored state"):
            model._requests(inputs, rows)
        torch.testing.assert_close(
            model._pages[RegionSlot(CacheRegion.SWA, 0)].data, before
        )

    def test_pair_restore_uses_real_fp32_bytes_and_position(self):
        model, _ = framework_fixture()
        raw = _pair_view(model._pair_pools[2], 1, 1)
        kv = torch.arange(512, dtype=torch.float32)
        score = -kv
        # Independent literal encoding from DSV41KVCacheSpec's byte offsets.
        expected = torch.cat(
            (
                kv.view(torch.uint8),
                score.view(torch.uint8),
                torch.tensor([3], dtype=torch.int64).view(torch.uint8),
                torch.tensor([1], dtype=torch.int32).view(torch.uint8),
                torch.zeros(4, dtype=torch.uint8),
            )
        )
        _write_pair(raw, PairCarry(2, "101", model.identity, 3, kv, score))
        torch.testing.assert_close(raw, expected, rtol=0, atol=0)
        restored = _read_pair(expected, 2, "101", model.identity, 3)
        torch.testing.assert_close(restored.partial_kv, kv, rtol=0, atol=0)
        torch.testing.assert_close(restored.partial_score, score, rtol=0, atol=0)
        self.assertEqual(restored.partial_kv.data_ptr(), expected.data_ptr())
        with self.assertRaisesRegex(ValueError, "execution boundary"):
            _read_pair(raw, 2, "101", model.identity, 4)

    def test_block_boundary_copies_swa_into_the_new_private_page(self):
        model, _ = framework_fixture()
        inputs, rows = request_fixture(starts=(1024,), ready=(True,))
        for pages in model._pages.values():
            if pages.region == CacheRegion.SWA:
                pages.data[1].fill_(71)
                pages.data[2].fill_(92)
        for layer, pool in model._pair_pools.items():
            _write_pair(
                _pair_view(pool, 1, 1),
                PairCarry.empty(layer, "101", model.identity, 1024),
            )
        for gid in range(4):
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group[gid] = (
                torch.tensor([[1] * 8 + [2]], dtype=torch.int32)
            )
        requests = model._requests(inputs, rows)
        self.assertEqual(requests[0].context.start, 1024)
        self.assertEqual(requests[0].context.end, 1026)
        self.assertEqual(requests[0].context.cache.request_id, "101")
        for pages in model._pages.values():
            if pages.region == CacheRegion.SWA:
                self.assertTrue(bool((pages.data[1] == 71).all()))
                self.assertTrue(bool((pages.data[2] == 71).all()))

    def test_requests_cannot_alias_writable_owner_pages(self):
        model, _ = framework_fixture()
        inputs, rows = request_fixture(
            starts=(0, 0), lengths=(2, 2), ready=(False, False), ids=(101, 202)
        )
        inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group[0][1, 0] = 1
        with self.assertRaisesRegex(ValueError, "writable global/index"):
            model._requests(inputs, rows)

    def test_multiple_choices_share_request_id_but_keep_private_pages(self):
        model, _ = framework_fixture()
        inputs, rows = request_fixture(
            starts=(0, 0), lengths=(2, 2), ready=(False, False), ids=(101, 101)
        )
        self.assertEqual(len(model._requests(inputs, rows)), 2)

    def test_readonly_global_prefix_can_be_shared_with_private_suffixes(self):
        model, _ = framework_fixture()
        inputs, rows = request_fixture(
            starts=(128, 128), lengths=(2, 2), ready=(True, True), ids=(101, 202)
        )
        for gid in range(4):
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group[gid] = (
                torch.tensor([[1, 2], [1, 3]], dtype=torch.int32)
            )
        for layer, pool in model._pair_pools.items():
            for page, request_id in ((1, "101"), (3, "202")):
                _write_pair(
                    _pair_view(pool, page, 1),
                    PairCarry.empty(layer, request_id, model.identity, 128),
                )
        requests = model._requests(inputs, rows)
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            [request.context.cache.request_id for request in requests], ["101", "202"]
        )

    def test_forward_returns_hidden_for_the_existing_head_and_sampler(self):
        model, _ = framework_fixture()
        inputs, _ = request_fixture()
        with patch("torch.cuda.is_current_stream_capturing", return_value=False):
            output = model(inputs)
        self.assertIsInstance(output, PyModelOutputs)
        self.assertEqual(output.hidden_states.shape, (2, 5120))
        self.assertEqual(output.hidden_states.dtype, torch.bfloat16)
        self.assertEqual(
            output.hidden_states.data_ptr(), model.target.output.data_ptr()
        )
        self.assertEqual(model.target.calls, 1)
        for layer, pool in model._pair_pools.items():
            carry = _read_pair(_pair_view(pool, 1, 1), layer, "101", model.identity, 2)
            self.assertIsNone(carry.partial_kv)
            initial = _read_pair(
                _pair_view(pool, 1, 0), layer, "101", model.identity, 0
            )
            self.assertIsNone(initial.partial_kv)

    def test_empty_rank_decode_keeps_target_call_without_touching_kv(self):
        model, _ = framework_fixture()
        inputs, rows = request_fixture(starts=(1,), lengths=(1,), fake=(True,))
        rows.valid.zero_()
        attn = inputs.attention_inputs
        attn.is_prefill = False
        attn.prefix_lengths = torch.empty(0, dtype=torch.int32)
        attn.sequence_lengths = torch.tensor([1], dtype=torch.int32)
        attn.kv_cache_kernel_block_id_device_by_group = []
        pools = [page.data for page in model._pages.values()]
        pools.extend(model._pair_pools.values())
        for pool in pools:
            pool.fill_(73)
        with patch("torch.cuda.is_current_stream_capturing", return_value=False):
            output = model(inputs)
        self.assertEqual(model.target.calls, 1)
        self.assertEqual(model.target.requests, [])
        self.assertEqual(output.hidden_states.shape, (1, 5120))
        self.assertEqual(
            output.hidden_states.data_ptr(), model.target.output.data_ptr()
        )
        for pool in pools:
            self.assertTrue(bool((pool == 73).all()))

    def test_fake_rows_do_not_shift_neighboring_live_requests(self):
        model, _ = framework_fixture()
        inputs, rows = request_fixture(
            starts=(0, 1, 0),
            lengths=(2, 1, 2),
            ready=(False, False, False),
            ids=(101, 0, 202),
            fake=(False, True, False),
        )
        rows.valid[2] = False
        for table in inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group:
            table[1].zero_()
            table[2] = torch.tensor([3, 4], dtype=torch.int32)
        requests = model._requests(inputs, rows)
        self.assertEqual(
            [(request.first, request.last) for request in requests], [(0, 2), (3, 5)]
        )
        self.assertEqual(
            [request.context.cache.request_id for request in requests], ["101", "202"]
        )

    def test_fake_flag_cannot_discard_live_rows(self):
        model, _ = framework_fixture()
        inputs, rows = request_fixture(fake=(True,))
        with self.assertRaisesRegex(
            ValueError, "fake request rows must all be invalid"
        ):
            model._requests(inputs, rows)

    def test_invalid_rows_cannot_silently_discard_an_ordinary_request(self):
        model, _ = framework_fixture()
        inputs, rows = request_fixture()
        rows.valid.zero_()
        with self.assertRaisesRegex(
            ValueError, "live request rows cannot contain padding"
        ):
            model._requests(inputs, rows)

    def test_prefill_and_decode_ranges_use_framework_lengths(self):
        inputs, _ = request_fixture(starts=(9, 128), lengths=(3, 5))
        self.assertEqual(
            _request_ranges(inputs.attention_inputs, 8), ([9, 128], [3, 5])
        )
        attn = inputs.attention_inputs
        attn.is_prefill = False
        attn.prefix_lengths = torch.empty(0, dtype=torch.int32)
        attn.sequence_lengths = torch.tensor([12, 133], dtype=torch.int32)
        self.assertEqual(_request_ranges(attn, 2), ([12, 133], [1, 1]))
        attn.context_parallel_info = object()
        with self.assertRaisesRegex(NotImplementedError, "CP page communication"):
            _request_ranges(attn, 2)

    def test_attention_preserves_request_order_and_zero_padding(self):
        class Attention(nn.Module):
            def forward(self, hidden, context):
                return hidden + context

        requests = [
            SimpleNamespace(first=0, last=2, context=1),
            SimpleNamespace(first=2, last=3, context=3),
        ]
        result = _BatchedAttention(Attention())(torch.ones(4, 2), requests)
        torch.testing.assert_close(
            result, torch.tensor([[2, 2], [2, 2], [4, 4], [0, 0]]).float()
        )

    def test_uncached_warmup_and_graph_are_explicitly_pending(self):
        model, _ = framework_fixture()
        inputs, _ = request_fixture()
        model.kv_cache = None
        with self.assertRaisesRegex(RuntimeError, "cache-backed engine warmup"):
            model(inputs)
        with self.assertRaisesRegex(NotImplementedError, "CUDA Graph implementation"):
            model.prepare_fmha_impl(inputs, is_cuda_graph=True)


if __name__ == "__main__":
    unittest.main()
