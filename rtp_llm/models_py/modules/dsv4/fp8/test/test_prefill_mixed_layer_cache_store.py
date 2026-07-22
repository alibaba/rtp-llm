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
from rtp_llm.models_py.modules.factory.attention.common import (
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
    def __init__(self, events: list, cache_tags: tuple[str, ...]):
        self.events = events
        self.cache_tags = cache_tags

    def get_layer_cache_groups(self, layer_idx: int):
        self.events.append(("get_cache", layer_idx))
        return [
            SimpleNamespace(tag=tag, layer_idx=layer_idx) for tag in self.cache_tags
        ]


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
        self._prefill_ws_swa_w = 1

    def _propagate_cp_ctx(self, cp_ctx) -> None:
        self.events.append(("propagate_cp", cp_ctx))

    def embed(self, input_ids):
        base = input_ids.to(torch.float32).unsqueeze(-1)
        return base.repeat(1, 4)

    def _hc_head_reduce(self, h):
        return h.mean(dim=1)

    def norm(self, h):
        return h


class MixedLayerCacheStoreOrderTest(unittest.TestCase):
    def _run_case(
        self,
        ratios: list[int],
        *,
        tagged: bool = True,
        writer_tags: tuple[str, ...] = ("swa_kv", "hca_kv"),
        cache_tags: tuple[str, ...] = ("swa_kv", "hca_kv"),
    ) -> list:
        events: list = []
        v4 = _FakeV4(ratios, events)
        kv_cache = _FakeKVCache(events, cache_tags)
        attn_inputs = SimpleNamespace(
            input_lengths=torch.tensor([4], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            is_prefill=True,
            cache_store_inputs=object(),
        )
        attention_inputs = (
            {tag: SimpleNamespace(writer_tag=tag) for tag in writer_tags}
            if tagged
            else None
        )

        def fake_create_writer(group_inputs):
            writer_tag = getattr(group_inputs, "writer_tag", "plain")
            events.append(("create_writer", writer_tag))

            def write(layer_cache):
                events.append(
                    ("store", writer_tag, layer_cache.tag, layer_cache.layer_idx)
                )

            return write

        # Force the per-forward workspace to ``align_bytes=1`` so the production
        # 1 GiB rounding does not allocate 1 GiB of host RAM in this CPU test.
        _orig_ws = prefill_forward.PrefillWorkspace

        def _small_align_ws(device, **kwargs):
            kwargs.setdefault("align_bytes", 1)
            return _orig_ws(device, **kwargs)

        with (
            patch.object(
                prefill_forward,
                "create_write_cache_store_impl",
                side_effect=fake_create_writer,
            ),
            patch.object(prefill_forward, "PrefillWorkspace", _small_align_ws),
        ):
            out = prefill_forward.forward_layers(
                v4,
                kv_cache,
                torch.arange(4, dtype=torch.long),
                torch.arange(4, dtype=torch.long),
                torch.tensor([0, 4], dtype=torch.int32),
                block_tables_by_type={0: torch.ones(1, 1, dtype=torch.int32)},
                attn_inputs=attn_inputs,
                attention_inputs=attention_inputs,
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
                    expected.extend(
                        ("store", tag, tag, i) for tag in ("swa_kv", "hca_kv")
                    )
                self.assertEqual(layer_store_events, expected)

                clear_events = [
                    e for e in events if e[0] == "set_meta" and e[2] is None
                ]
                self.assertEqual(len(clear_events), len(ratios))

    def test_plain_input_keeps_single_group_fast_path(self) -> None:
        events = self._run_case(
            [0],
            tagged=False,
            cache_tags=("swa_kv",),
        )
        self.assertIn(("store", "plain", "swa_kv", 0), events)

    def test_plain_input_rejects_multiple_layer_cache_groups(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly one cache group"):
            self._run_case(
                [0],
                tagged=False,
                cache_tags=("swa_kv", "hca_kv"),
            )

    def test_tagged_input_rejects_missing_writer_for_layer_tag(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing cache-store writer"):
            self._run_case(
                [0],
                writer_tags=("swa_kv",),
                cache_tags=("swa_kv", "hca_kv"),
            )


class CacheStoreCPMetadataTest(unittest.TestCase):
    def test_create_writer_uses_cp_actual_lengths_and_current_prefix_lengths(
        self,
    ) -> None:
        actual_lengths = torch.tensor([7, 5], dtype=torch.int32)
        local_cp_lengths = torch.tensor([4, 4], dtype=torch.int32)
        prefix_lengths = torch.tensor([11, 13], dtype=torch.int32)
        block_ids = torch.ones(2, 3, dtype=torch.int32)
        cache_store_inputs = object()
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            cache_store_inputs=cache_store_inputs,
            input_lengths=local_cp_lengths,
            prefix_lengths=prefix_lengths,
            context_parallel_info=SimpleNamespace(
                prefill_actual_input_lengths_cpu=actual_lengths
            ),
            kv_cache_block_id=block_ids,
        )

        writer = create_write_cache_store_impl(attn_inputs)

        self.assertIs(writer.input_lengths, actual_lengths)
        self.assertIs(writer.prefix_lengths, prefix_lengths)
        self.assertIs(writer.kv_cache_block_id, block_ids)
        self.assertIs(writer.cache_store_inputs, cache_store_inputs)


if __name__ == "__main__":
    unittest.main()
