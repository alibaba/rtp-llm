"""Standard adapter metadata and physical-pool contracts, not engine acceptance."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fixture import flash_config
from torch import nn

from rtp_llm.config.dsv41_config import derive_checkpoint_revision
from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import build_model_config
from rtp_llm.models.deepseek_v41 import DeepSeekV41
from rtp_llm.models.deepseek_v41_dspark import DeepSeekV41DSpark
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
    GLOBAL_OWNERS,
    PAIR_OWNERS,
    CacheLayout,
    CacheRegion,
    RegionSlot,
)
from rtp_llm.models_py.modules.dsv41.ced import ReplayConfig
from rtp_llm.models_py.modules.dsv41.compressor import PairCarry
from rtp_llm.models_py.modules.dsv41.inputs import V41ModelRows
from rtp_llm.ops import DataType, KvCacheDataType
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
    model._capture_aux = ()
    model._mtp_aux_buffer = None
    model._active_v41_graph_impl = None
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


def _write_checkpoint(folder, raw=None):
    root = Path(folder)
    (root / "config.json").write_text(
        json.dumps(raw if raw is not None else flash_config()), encoding="utf-8"
    )
    (root / "model.safetensors.index.json").write_text(
        '{"weight_map": {}}', encoding="utf-8"
    )


class EngineAdapterContractTest(unittest.TestCase):
    def test_dspark_proposal_without_vit_skips_multimodal_hooks(self):
        model = DeepSeekV41DSpark.__new__(DeepSeekV41DSpark)
        model.model_config = SimpleNamespace(is_mtp=True)
        model.vit_config = None
        self.assertIsNone(model._as_multimodal_model())

    def test_create_config_revision_defaults_to_content_derivation(self):
        with tempfile.TemporaryDirectory() as folder:
            _write_checkpoint(folder)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DSV41_HF_REVISION", None)
                config = DeepSeekV41._create_config(folder)
            self.assertEqual(
                config.dsv41_model_revision, derive_checkpoint_revision(folder)
            )

    def test_create_config_revision_env_override(self):
        with tempfile.TemporaryDirectory() as folder:
            _write_checkpoint(folder)
            with patch.dict(os.environ, {"DSV41_HF_REVISION": "b" * 40}):
                config = DeepSeekV41._create_config(folder)
            self.assertEqual(config.dsv41_model_revision, "b" * 40)

    def test_standard_config_preserves_mixed_checkpoint_and_typed_cache(self):
        raw = flash_config()
        with tempfile.TemporaryDirectory() as folder:
            _write_checkpoint(folder, raw)
            config = DeepSeekV41._create_config(folder)
            model_args = ModelArgs()
            model_args.ckpt_path = model_args.tokenizer_path = folder
            model_args.model_type = "deepseek_v41"
            model_args.act_type = "BF16"
            cache = KVCacheConfig()
            cache.seq_size_per_block = cache.kernel_seq_size_per_block = 128
            build_model_config(
                config, model_args, cache, SimpleNamespace(hack_layer_num=0)
            )
            self.assertEqual(config.data_type, DataType.TYPE_BF16)
            self.assertEqual(config.attn_config.kv_cache_dtype, KvCacheDataType.BASE)
            self.assertIsNone(config.quant_config)
            self.assertFalse(config.quant_algo.isQuant())
            self.assertEqual(
                config.dsv41_config.quantization, raw["quantization_config"]
            )
            self.assertEqual(config.num_layers, 40)
            self.assertTrue(config.enable_fp32_lm_head)

    def test_v41_precision_rejects_generic_weight_or_cache_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            _write_checkpoint(folder)
            config = DeepSeekV41._create_config(folder)
            cache = KVCacheConfig()
            with self.assertRaisesRegex(ValueError, "BF16"):
                config.init_precision_config(cache, "FP16")
            config.quantization = "FP8"
            with self.assertRaisesRegex(ValueError, "mixed quantization"):
                config.init_precision_config(cache, "BF16")
            config.quantization = ""
            for name in ("int8_kv_cache", "fp8_kv_cache"):
                with self.subTest(name=name):
                    setattr(cache, name, True)
                    with self.assertRaisesRegex(ValueError, "typed cache regions"):
                        config.init_precision_config(cache, "BF16")
                    setattr(cache, name, False)
            config.init_precision_config(cache, None)
            self.assertEqual(config.data_type, DataType.TYPE_BF16)

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

    def test_continuous_decode_readiness_derives_from_pair_bytes(self):
        model, _ = framework_fixture()
        inputs, rows = request_fixture(starts=(128,), ready=(False,))
        for layer, pool in model._pair_pools.items():
            _write_pair(
                _pair_view(pool, 1, 1),
                PairCarry.empty(layer, "101", model.identity, 128),
            )
        requests = model._requests(inputs, rows)
        self.assertEqual(requests[0].context.start, 128)
        self.assertEqual(requests[0].context.end, 130)

    def test_continuous_decode_rejects_mismatched_pair_bytes(self):
        model, _ = framework_fixture()
        inputs, rows = request_fixture(starts=(128,), ready=(False,))
        before = model._pages[RegionSlot(CacheRegion.SWA, 0)].data.clone()
        with self.assertRaisesRegex(ValueError, "execution boundary"):
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

    def test_canonical_memory_pair_requires_ready_aligned_complete_zero_region(self):
        model, _ = framework_fixture()
        region = model._pair_pools[2][1]
        raw = _pair_view(model._pair_pools[2], 1, 1)
        args = dict(checkpoint_region=region, state_ready=True, reuse_unit=1024)
        pair = _read_pair(raw, 2, "101", model.identity, 1024, **args)
        self.assertEqual(pair.next_position, 1024)
        self.assertIsNone(pair.partial_kv)
        self.assertFalse(bool(region.any()))
        for position, ready, byte in ((1024, False, None), (1026, True, None),
                                      (1025, True, None), (1024, True, 0),
                                      (1024, True, -1), (1024, True, 8208)):
            region.zero_()
            if byte is not None:
                region[byte] = 1
            with self.assertRaisesRegex(ValueError, "execution boundary"):
                _read_pair(raw, 2, "101", model.identity, position,
                           checkpoint_region=region, state_ready=ready, reuse_unit=1024)

    def test_standard_adapter_restores_canonical_memory_pair(self):
        model, _ = framework_fixture()
        inputs, rows = request_fixture(starts=(1024,), ready=(True,))
        for gid in range(4):
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group[gid] = (
                torch.tensor([[1] * 8 + [2]], dtype=torch.int32)
            )
        requests = model._requests(inputs, rows)
        for owner in PAIR_OWNERS:
            pair = requests[0].context.cache.owners[owner].pair
            self.assertEqual(pair.next_position, 1024)
            self.assertIsNone(pair.partial_kv)
            self.assertFalse(bool(model._pair_pools[owner][1].any()))

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

    def test_uncached_allocation_warmup_does_not_bind_request_pages(self):
        model, _ = framework_fixture()
        inputs, _ = request_fixture()
        model.kv_cache = None
        output = model(inputs)
        self.assertEqual(tuple(output.hidden_states.shape), (2, 5120))
        self.assertFalse(bool(output.hidden_states.any()))
        self.assertEqual(model.target.calls, 0)
        self.assertIsNone(model.prepare_fmha_impl(inputs))
        with self.assertRaisesRegex(ValueError, "complete local engine pages"):
            model.prepare_fmha_impl(inputs, is_cuda_graph=True)

    def test_cp_dispatch_chunks_the_complete_input_inside_one_forward(self):
        model, _ = framework_fixture()
        inputs, rows = request_fixture(lengths=(8,))
        inputs.attention_inputs.context_parallel_info = object()
        model.max_tokens_per_rank = 2
        result = PyModelOutputs(torch.ones((8, 5120), dtype=torch.bfloat16))
        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(model, "_requests", return_value=[]) as requests,
            patch.object(model, "_forward_cp", return_value=result) as execute,
            patch.object(model, "_write_cache_store") as write,
        ):
            self.assertIs(model(inputs), result)
        self.assertEqual(requests.call_count, 1)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(write.call_count, 1)
        self.assertEqual(model.target.calls, 0)
        torch.testing.assert_close(execute.call_args.args[1].token_ids, rows.token_ids)

    def test_decode_dispatch_refreshes_shared_aux_across_graph_buckets(self):
        model, _ = framework_fixture()
        model._capture_aux = (37, 38, 39)
        model._mtp_aux_buffer = torch.full((6, 15360), -1, dtype=torch.bfloat16)
        buffer_ptr = model._mtp_aux_buffer.data_ptr()
        calls = []

        class Target(nn.Module):
            def forward(
                self, rows, context, *, execution_mode, aux_row_indices, dense_aux
            ):
                calls.append((context, execution_mode, dense_aux))
                torch.testing.assert_close(
                    aux_row_indices, torch.arange(rows.token_ids.numel())
                )
                hidden = rows.token_ids[:, None].expand(-1, 5120).to(torch.bfloat16)
                return SimpleNamespace(
                    hidden_states=hidden, aux_hidden_states=hidden.repeat(1, 3)
                )

        model.target = Target()
        for width, token in ((6, 11), (1, 27), (6, 43)):
            inputs, rows = request_fixture(lengths=(width,))
            rows.token_ids.fill_(token)
            impl = SimpleNamespace(rows=rows, context=object())
            with (
                patch.object(impl, "begin_forward", create=True) as begin,
                patch.object(impl, "finish_forward", create=True) as finish,
                patch("torch.cuda.is_current_stream_capturing", return_value=False),
            ):
                model._active_v41_graph_impl = impl
                output = model(inputs, impl)
            self.assertEqual(begin.call_count, 1)
            self.assertEqual(finish.call_count, 1)
            self.assertEqual(tuple(output.hidden_states.shape), (width, 5120))
            aux = model.get_mtp_target_hidden_states(width)
            self.assertEqual(aux.data_ptr(), buffer_ptr)
            self.assertTrue(bool((aux == token).all()))
        self.assertTrue(all(mode == "full" and dense for _, mode, dense in calls))
        with self.assertRaisesRegex(ValueError, "fixed decode buffer"):
            model.get_mtp_target_hidden_states(7)

    def test_cp_publication_reports_materialized_boundary_facts(self):
        cache = SimpleNamespace(
            request_id="101",
            layout=SimpleNamespace(draft_enabled=True),
            swa_ends={layer: 1024 for layer in range(43)},
            owners={
                layer: SimpleNamespace(materialized_end=1024)
                for layer in GLOBAL_OWNERS
            },
            swa={
                layer: SimpleNamespace(valid_starts=[896], valid_ends=[1024])
                for layer in range(43)
            },
        )
        decoder = SimpleNamespace(cache=cache, end=1024, replay_floor=896)
        history = SimpleNamespace(token_ids=(31, 129264, 37), image_mask=(False, True, False))
        publication = DeepSeekV41Model._cp_publication(decoder, history)
        self.assertEqual(publication.request_id, 101)
        self.assertEqual(publication.materialized_end, 1024)
        self.assertEqual(len(publication.global_entries), 4)
        self.assertEqual(list(publication.swa_valid_end), [1024] * 43)
        self.assertEqual(list(publication.swa_replay_floor), [0] * 21 + [896] * 22)
        self.assertEqual((publication.aux_valid_start, publication.aux_valid_end), (896, 1024))
        self.assertEqual(list(publication.history_token_ids), [31, 129264, 37])
        self.assertEqual(list(publication.history_image_mask), [0, 1, 0])
        self.assertTrue(publication.draft_committed)

    def test_cp_publication_rejects_swa_not_at_the_boundary(self):
        cache = SimpleNamespace(
            request_id="101",
            layout=SimpleNamespace(draft_enabled=True),
            swa_ends={layer: 1024 for layer in range(43)},
            owners={layer: SimpleNamespace(materialized_end=1024) for layer in GLOBAL_OWNERS},
            swa={
                layer: SimpleNamespace(valid_starts=[896], valid_ends=[1024])
                for layer in range(43)
            },
        )
        cache.swa_ends[20] = 896
        decoder = SimpleNamespace(cache=cache, end=1024, replay_floor=896)
        history = SimpleNamespace(token_ids=(31, 129264, 37), image_mask=(False, True, False))
        with self.assertRaisesRegex(RuntimeError, "actual boundary"):
            DeepSeekV41Model._cp_publication(decoder, history)

    @staticmethod
    def _cp_publish_fixture(rank, install_result=True, publish_result=True):
        from unittest.mock import Mock

        from rtp_llm.models_py.modules.dsv41.cache_layout import CacheRegion
        from rtp_llm.ops.compute_ops import KVCacheRegionName

        cp_size = 2
        counts = [1] * 6
        native = SimpleNamespace(
            block_ids_by_group=[[gid] for gid in range(6)],
            publish=Mock(return_value=publish_result),
            install=Mock(return_value=install_result),
        )

        class Slot:
            def __init__(self, owner_layer, region):
                self.owner_layer = owner_layer
                self.region = region

        tables = {}
        groups = {}
        for gid, (owner, region) in enumerate(
            [(2, CacheRegion.GLOBAL), (8, CacheRegion.GLOBAL),
             (2, CacheRegion.INDEX_K), (8, CacheRegion.INDEX_K)]
        ):
            slot = Slot(owner, region)
            tables[slot] = torch.tensor([[10 + gid], [20 + gid]])
            groups[(owner, int(KVCacheRegionName.DSV41_GLOBAL_KV if region == CacheRegion.GLOBAL
                                  else KVCacheRegionName.DSV41_INDEX_KV))] = gid
        swa_slot = Slot(0, CacheRegion.SWA)
        tables[swa_slot] = torch.tensor([[50], [60]])
        groups[(0, int(KVCacheRegionName.SWA_KV))] = 5
        groups[(2, int(KVCacheRegionName.DSV41_PAIR_STATE))] = 4
        groups[(8, int(KVCacheRegionName.DSV41_PAIR_STATE))] = 4
        groups[(14, int(KVCacheRegionName.DSV41_PAIR_STATE))] = 4
        cache = SimpleNamespace(
            request_id="101",
            layout=SimpleNamespace(draft_enabled=True),
            swa_ends={layer: 1024 for layer in range(43)},
            owners={layer: SimpleNamespace(materialized_end=1024) for layer in GLOBAL_OWNERS},
            swa={
                layer: SimpleNamespace(valid_starts=[896], valid_ends=[1024])
                for layer in range(43)
            },
        )
        decoder = SimpleNamespace(
            cache=cache,
            end=1024,
            replay_floor=896,
            query_device="cpu",
            tables=tables,
            pair_tables={2: torch.tensor([[30], [40]])},
        )
        history = SimpleNamespace(token_ids=(31, 129264, 37), image_mask=(False, True, False))
        model = SimpleNamespace(
            _cp_rank=rank,
            _groups=groups,
            _cp_publication=DeepSeekV41Model._cp_publication,
            kv_cache=SimpleNamespace(group_region_names=[""] * 6),
            layout=SimpleNamespace(cp_size=cp_size),
        )
        remote_flat = [110, 111, 112, 113, 130, 150]
        local_flat = [10, 11, 12, 13, 30, 50]
        gathered = local_flat + remote_flat if rank == 0 else remote_flat + local_flat

        def fake_broadcast(tensor, src, group):
            tensor.copy_(torch.tensor(counts, dtype=torch.int64))

        def fake_all_gather(tensor, group):
            return torch.tensor(gathered)

        def fake_all_reduce(tensor, group):
            return tensor + (cp_size - 1)

        patches = (
            patch("rtp_llm.models_py.distributed.collective_torch.broadcast", fake_broadcast),
            patch("rtp_llm.models_py.distributed.collective_torch.all_gather", fake_all_gather),
            patch("rtp_llm.models_py.distributed.collective_torch.all_reduce", fake_all_reduce),
            patch("rtp_llm.models_py.distributed.collective_torch.barrier", lambda group: None),
            patch("torch.cuda.current_stream", return_value=SimpleNamespace(synchronize=lambda: None)),
        )
        return model, native, decoder, history, patches

    def test_cp_publish_installs_checkpoint_on_every_producer_rank(self):
        model, native, decoder, history, patches = self._cp_publish_fixture(rank=1)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            self.assertTrue(DeepSeekV41Model._cp_publish(model, native, decoder, history))
        self.assertEqual(native.publish.call_count, 0)
        self.assertEqual(native.install.call_count, 1)
        publication, actual, workers = native.install.call_args.args
        self.assertEqual(publication.request_id, 101)
        self.assertEqual(publication.materialized_end, 1024)
        self.assertEqual(actual, [[10], [11], [12], [13], [30], [50]])
        self.assertEqual(workers[0], [[110], [111], [112], [113], [130], [150]])
        self.assertEqual(workers[1], actual)

        model, native, decoder, history, patches = self._cp_publish_fixture(rank=0)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            self.assertTrue(DeepSeekV41Model._cp_publish(model, native, decoder, history))
        self.assertEqual(native.install.call_count, 0)
        self.assertEqual(native.publish.call_count, 1)
        _, actual, workers = native.publish.call_args.args
        self.assertEqual(actual, [[10], [11], [12], [13], [30], [50]])
        self.assertEqual(workers[1], [[110], [111], [112], [113], [130], [150]])

    def test_cp_publish_fails_when_a_worker_install_fails(self):
        model, native, decoder, history, patches = self._cp_publish_fixture(
            rank=1, install_result=False
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with self.assertRaisesRegex(RuntimeError, "producer rank"):
                DeepSeekV41Model._cp_publish(model, native, decoder, history)
        self.assertEqual(native.install.call_count, 1)


if __name__ == "__main__":
    unittest.main()
