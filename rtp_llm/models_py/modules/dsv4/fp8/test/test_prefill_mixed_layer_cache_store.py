"""UT: mixed SWA/CSA/HCA prefill layer loop ordering.

The FP8 prefill loop builds shared metadata once per compress-ratio bucket,
then each layer consumes its own ratio's meta and cache-store writes that
layer's KV region immediately after the layer returns. This matters for CP
overlap because async compressor/SWA gathers must be fully drained before the
Python layer call returns and before the C++ async cache-store writer records
its event.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, NamedTuple, Optional
from unittest.mock import patch

import torch

import rtp_llm.models_py.modules.dsv4.prefill.forward as prefill_forward
from rtp_llm.models_py.modules.base.common.kvcache_store import (
    create_write_cache_store_impl,
)


class _FakeMeta(NamedTuple):
    """Minimal stand-in for ``PrefillMeta``: ``forward_layers`` now threads the
    per-forward ``PrefillWorkspace`` into the broadcast meta via
    ``meta._replace(workspace=…)`` (prefill_meta.py), so the fake meta must be a
    NamedTuple exposing a ``workspace`` field for ``_replace`` to target."""

    ratio: int
    built_by_layer: int
    start_pos: int
    workspace: Optional[Any] = None


class _FakeAttn:
    def __init__(self, layer_idx: int, compress_ratio: int, events: list):
        self.layer_idx = layer_idx
        self.compress_ratio = compress_ratio
        self.events = events
        self._prefill_meta_shared = None
        self._kv_cache = "original_kv"
        self._block_tables_by_type = "original_bt"
        self._cp_ctx = None
        self.freqs_bound = False

    def _build_shared_prefill_meta(self, x, start_pos, **kwargs):
        self.events.append(
            (
                "build",
                self.layer_idx,
                self.compress_ratio,
                self._kv_cache,
                self._block_tables_by_type,
            )
        )
        return _FakeMeta(
            ratio=self.compress_ratio,
            built_by_layer=self.layer_idx,
            start_pos=start_pos,
        )

    def _ensure_freqs_cis_bound(self) -> None:
        self.freqs_bound = True
        self.events.append(("bind_freqs", self.layer_idx))

    def _set_prefill_meta_shared(self, meta) -> None:
        self._prefill_meta_shared = meta
        ratio = None if meta is None else meta.ratio
        built_by = None if meta is None else meta.built_by_layer
        self.events.append(("set_meta", self.layer_idx, ratio, built_by))


class _FakeLayer:
    def __init__(self, layer_idx: int, compress_ratio: int, events: list):
        self.layer_idx = layer_idx
        self.attn = _FakeAttn(layer_idx, compress_ratio, events)
        self.events = events

    def __call__(
        self,
        h,
        input_ids,
        positions,
        cu_seqlens,
        *,
        kv_cache=None,
        block_tables_by_type=None,
    ):
        meta = self.attn._prefill_meta_shared
        assert meta is not None
        assert meta.ratio == self.attn.compress_ratio
        self.events.append(
            (
                "layer",
                self.layer_idx,
                self.attn.compress_ratio,
                meta.built_by_layer,
            )
        )
        return h + float(self.layer_idx + 1)


class _FakeKVCache:
    def __init__(self, events: list):
        self.events = events

    def get_layer_cache_groups(self, layer_idx: int):
        self.events.append(("get_cache", layer_idx))
        return f"layer_cache_{layer_idx}"


class _FakeV4:
    def __init__(self, ratios: list[int], events: list):
        self.layers = [_FakeLayer(i, r, events) for i, r in enumerate(ratios)]
        self.events = events
        self.fp8_kv_cache = True
        self.hc_mult = 2
        self._cp_info = None
        self._cp_size = 1
        self._cp_rank = 0
        self._kv_cache_sharded = False
        self._mtp_hidden_buffer = None
        self._mtp_last_hidden_buffer = None
        self.capture_aux_hidden_layer_ids = []
        # ``forward_layers`` builds the per-forward ``PrefillWorkspace`` from
        # these bind-time dims (transformer.py resolves them on the real model).
        # Tiny values keep the CPU allocation trivial; the test patches
        # ``PrefillWorkspace`` to ``align_bytes=1`` so the 1 GiB production
        # alignment does not force a 1 GiB host alloc here.
        self._prefill_ws_q_rows = 4
        self._prefill_ws_q_dim = 4
        self._prefill_ws_full_rows = 4
        self._prefill_ws_main_w = 1
        self._prefill_ws_idx_w = 1

    def _propagate_cp_ctx(self, cp_ctx) -> None:
        self.events.append(("propagate_cp", cp_ctx))

    def set_cp_info(self, cp_info, cp_size, cp_rank, **kwargs) -> None:
        self._cp_info = cp_info
        self._cp_size = cp_size
        self._cp_rank = cp_rank

    def embed(self, input_ids):
        base = input_ids.to(torch.float32).unsqueeze(-1)
        return base.repeat(1, 4)

    def embed_full(self, input_ids):
        return self.embed(input_ids)

    def _hc_head_reduce(self, h):
        return h.mean(dim=1)

    def norm(self, h):
        return h


class MixedLayerCacheStoreOrderTest(unittest.TestCase):
    def _run_case(self, ratios: list[int]) -> list:
        events: list = []
        v4 = _FakeV4(ratios, events)
        kv_cache = _FakeKVCache(events)
        attn_inputs = SimpleNamespace(
            input_lengths=torch.tensor([4], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            is_prefill=True,
            cache_store_inputs=object(),
        )

        def fake_create_writer(attn, kv):
            events.append(("create_writer", kv))

            def write(layer_cache):
                events.append(("store", layer_cache))

            return write

        # Force the per-forward workspace to ``align_bytes=1`` so the production
        # 1 GiB rounding does not allocate 1 GiB of host RAM in this CPU test.
        _orig_ws = prefill_forward.PrefillWorkspace

        def _small_align_ws(device, **kwargs):
            kwargs.setdefault("align_bytes", 1)
            return _orig_ws(device, **kwargs)

        with patch.object(
            prefill_forward,
            "create_write_cache_store_impl",
            side_effect=fake_create_writer,
        ), patch.object(prefill_forward, "PrefillWorkspace", _small_align_ws):
            out = prefill_forward.forward_layers(
                v4,
                kv_cache,
                torch.arange(4, dtype=torch.long),
                torch.arange(4, dtype=torch.long),
                torch.tensor([0, 4], dtype=torch.int32),
                block_tables_by_type={0: torch.ones(1, 1, dtype=torch.int32)},
                attn_inputs=attn_inputs,
            )

        self.assertEqual(tuple(out.shape), (4, 4))
        for layer in v4.layers:
            self.assertTrue(layer.attn.freqs_bound)
            self.assertIsNone(layer.attn._prefill_meta_shared)
        return events

    def test_mixed_ratio_orders_build_once_and_store_after_each_layer(self) -> None:
        cases = [
            [0, 4, 128],
            [4, 0, 128],
            [128, 4, 0, 4, 0],
        ]
        for ratios in cases:
            with self.subTest(ratios=ratios):
                events = self._run_case(ratios)

                build_events = [e for e in events if e[0] == "build"]
                self.assertEqual(
                    [e[2] for e in build_events],
                    list(dict.fromkeys(ratios)),
                )

                # During meta build, the representative attention gets the
                # active framework KV handle and block tables, then they are
                # restored after propagation.
                for event in build_events:
                    self.assertIsInstance(event[3], _FakeKVCache)
                    self.assertIsInstance(event[4], dict)

                layer_store_events = [
                    e for e in events if e[0] in ("layer", "get_cache", "store")
                ]
                expected = []
                first_layer_for_ratio = {}
                for i, ratio in enumerate(ratios):
                    first_layer_for_ratio.setdefault(ratio, i)
                    expected.append(("layer", i, ratio, first_layer_for_ratio[ratio]))
                    expected.append(("get_cache", i))
                    expected.append(("store", f"layer_cache_{i}"))
                self.assertEqual(layer_store_events, expected)

                clear_events = [
                    e for e in events if e[0] == "set_meta" and e[2] is None
                ]
                self.assertEqual(len(clear_events), len(ratios))


class CacheStoreCPMetadataTest(unittest.TestCase):
    def test_create_writer_uses_cp_actual_lengths_and_pinned_prefix_lengths(
        self,
    ) -> None:
        actual_lengths = torch.tensor([7, 5], dtype=torch.int32)
        local_cp_lengths = torch.tensor([4, 4], dtype=torch.int32)
        prefix_host = torch.tensor([11, 13], dtype=torch.int32)
        prefix_device_mirror = torch.tensor([0, 0], dtype=torch.int32)
        block_ids = torch.ones(2, 3, dtype=torch.int32)
        cache_store_inputs = SimpleNamespace(
            input_lengths_host=torch.tensor([99, 99], dtype=torch.int32),
            prefix_lengths_host=prefix_host,
        )
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            cache_store_inputs=cache_store_inputs,
            input_lengths=local_cp_lengths,
            prefix_lengths=prefix_device_mirror,
            context_parallel_info=SimpleNamespace(
                prefill_actual_input_lengths_cpu=actual_lengths
            ),
            kv_cache_block_id_host=block_ids,
        )

        writer = create_write_cache_store_impl(attn_inputs)

        self.assertIs(writer.input_lengths, actual_lengths)
        self.assertIs(writer.prefix_lengths, prefix_host)
        self.assertIs(writer.kv_cache_block_id_host, block_ids)
        self.assertIs(writer.cache_store_inputs, cache_store_inputs)


class _RecordingWriter:
    def __init__(self, events):
        self.events = events

    def write(self, params, layer_cache):
        self.events.append(
            ("native_write", layer_cache.layer_id, layer_cache.tag, params)
        )


class _TaggedKVCache:
    def __init__(self, tags, events):
        self.group_tags = list(tags)
        self.events = events

    def get_layer_cache_groups(self, layer_idx):
        self.events.append(("get_cache", layer_idx))
        return [SimpleNamespace(tag=tag, layer_id=layer_idx) for tag in self.group_tags]


class TaggedCacheStoreForwardTest(unittest.TestCase):
    """Exercise the real caller, layer loop, factory and writer dispatch.

    Attention arithmetic and the native receiver are test doubles; routing
    itself is not mocked. Native CP preparation and physical-address checks
    live in pywrapped_model_cache_store_integration_test.
    """

    TAGS = (
        "swa_kv",
        "hca_state",
        "indexer_state",
        "csa_state",
        "hca_kv",
        "indexer_kv",
        "csa_kv",
    )

    def _inputs(self, events, tags=None, active=True):
        result = {}
        writer = _RecordingWriter(events)
        for i, tag in enumerate(self.TAGS if tags is None else tags):
            table = torch.tensor([[10 * i + 1, -1, 10 * i + 3]], dtype=torch.int32)
            params = SimpleNamespace(
                input_lengths_host=torch.tensor([4], dtype=torch.int32),
                prefix_lengths_host=torch.tensor([0], dtype=torch.int32),
                host_kv_cache_offset=table,
            )
            result[tag] = SimpleNamespace(
                is_prefill=True,
                input_lengths=torch.tensor([4], dtype=torch.int32),
                prefix_lengths=torch.tensor([0], dtype=torch.int32),
                cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
                combo_position_ids=torch.arange(4, dtype=torch.long),
                kv_cache_block_id=table,
                kv_cache_kernel_block_id_device=table,
                cache_store_inputs=params if active else None,
                cache_store_writer=writer if active else None,
            )
        return result

    def _run(self, attention_inputs, kv_cache, events, direct=False):
        v4 = _FakeV4([0, 4, 128], events)
        workspace_cls = prefill_forward.PrefillWorkspace

        def small_workspace(device, **kwargs):
            return workspace_cls(device, align_bytes=1, **kwargs)

        with patch.object(prefill_forward, "PrefillWorkspace", small_workspace):
            if direct:
                out = prefill_forward.forward_layers(
                    v4,
                    kv_cache,
                    torch.arange(4),
                    torch.arange(4),
                    torch.tensor([0, 4], dtype=torch.int32),
                    {},
                    attn_inputs=attention_inputs,
                )
                self.assertEqual(tuple(out.shape), (4, 4))
            else:
                model_inputs = SimpleNamespace(
                    attention_inputs=attention_inputs,
                    input_ids=torch.arange(4),
                    pp_intermediates={},
                )
                prefill_forward.forward_prefill(v4, kv_cache, None, model_inputs)

    def test_actual_forward_routes_all_groups_to_their_own_metadata(self):
        events = []
        by_tag = self._inputs(events)
        kv_cache = _TaggedKVCache(self.TAGS, events)
        self._run(by_tag, kv_cache, events)

        writes = [event for event in events if event[0] == "native_write"]
        self.assertEqual(len(writes), 3 * len(self.TAGS))
        for _, layer_id, tag, params in writes:
            self.assertIn(layer_id, (0, 1, 2))
            self.assertIs(params, by_tag[tag].cache_store_inputs)
            self.assertIs(params.host_kv_cache_offset, by_tag[tag].kv_cache_block_id)

        # Both the tag and the stage-local layer number survive dispatch, and
        # publication follows the producing layer (not the next layer).
        flow = [event[:3] for event in events if event[0] in ("layer", "native_write")]
        expected = []
        for layer_id, ratio in enumerate((0, 4, 128)):
            expected.append(("layer", layer_id, ratio))
            expected.extend(("native_write", layer_id, tag) for tag in self.TAGS)
        self.assertEqual(flow, expected)

    def test_mapping_order_does_not_change_metadata_selection(self):
        for tags in (self.TAGS, tuple(reversed(self.TAGS))):
            with self.subTest(tags=tags):
                events = []
                by_tag = self._inputs(events, tags=tags)
                self._run(by_tag, _TaggedKVCache(self.TAGS, events), events)
                for event in events:
                    if event[0] == "native_write":
                        self.assertIs(event[3], by_tag[event[2]].cache_store_inputs)

    def test_incomplete_one_entry_map_fails_before_any_publication(self):
        events = []
        by_tag = self._inputs(events, tags=("csa_kv",))
        with self.assertRaisesRegex(RuntimeError, "no writer for cache tags"):
            self._run(by_tag, _TaggedKVCache(self.TAGS, events), events)
        self.assertFalse(any(event[0] == "native_write" for event in events))

    def test_missing_writer_on_one_active_tag_is_not_silently_skipped(self):
        events = []
        by_tag = self._inputs(events)
        by_tag["hca_state"].cache_store_writer = None
        with self.assertRaisesRegex(RuntimeError, "hca_state"):
            self._run(by_tag, _TaggedKVCache(self.TAGS, events), events)
        self.assertFalse(any(event[0] == "native_write" for event in events))

    def test_pd_off_keeps_forward_without_cache_store_work(self):
        events = []
        by_tag = self._inputs(events, active=False)
        self._run(by_tag, _TaggedKVCache(self.TAGS, events), events)
        self.assertEqual(sum(event[0] == "layer" for event in events), 3)
        self.assertFalse(any(event[0] == "native_write" for event in events))

    def test_single_group_object_and_map_are_supported(self):
        for tagged in (False, True):
            with self.subTest(tagged=tagged):
                events = []
                by_tag = self._inputs(events, tags=("default",))
                inputs = by_tag if tagged else by_tag["default"]
                self._run(inputs, _TaggedKVCache(["default"], events), events)
                writes = [event for event in events if event[0] == "native_write"]
                self.assertEqual(len(writes), 3)
                self.assertTrue(
                    all(
                        event[3] is by_tag["default"].cache_store_inputs
                        for event in writes
                    )
                )

    def test_direct_multigroup_call_cannot_fall_back_to_primary_writer(self):
        events = []
        primary = self._inputs(events)["csa_kv"]
        with self.assertRaisesRegex(
            RuntimeError, "tagged attention inputs are required"
        ):
            self._run(primary, _TaggedKVCache(self.TAGS, events), events, direct=True)
        self.assertFalse(any(event[0] == "native_write" for event in events))


if __name__ == "__main__":
    unittest.main()
