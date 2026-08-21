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
    common_token: Any
    swa_group1_token: Any
    freqs_token: Any
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
        reused = kwargs.get("reuse_common_meta")
        reused_freqs = kwargs.get("reuse_freqs_meta")
        common_token = object() if reused is None else reused.common_token
        swa_group1_token = object() if reused is None else reused.swa_group1_token
        freqs_token = object() if reused_freqs is None else reused_freqs.freqs_token
        self.events.append(
            (
                "build",
                self.layer_idx,
                self.compress_ratio,
                self._kv_cache,
                self._block_tables_by_type,
                reused,
                reused_freqs,
                common_token,
                swa_group1_token,
                freqs_token,
            )
        )
        return _FakeMeta(
            ratio=self.compress_ratio,
            built_by_layer=self.layer_idx,
            start_pos=start_pos,
            common_token=common_token,
            swa_group1_token=swa_group1_token,
            freqs_token=freqs_token,
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

    def get_layer_caches(self, layer_idx: int):
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
        self.capture_aux_hidden_layer_ids = ()
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

    def embed(self, input_ids):
        base = input_ids.to(torch.float32).unsqueeze(-1)
        return base.repeat(1, 4)

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
            [4, 128],
        ]
        for ratios in cases:
            with self.subTest(ratios=ratios):
                events = self._run_case(ratios)

                build_events = [e for e in events if e[0] == "build"]
                distinct_ratios = list(dict.fromkeys(ratios))
                if 0 in distinct_ratios:
                    distinct_ratios.remove(0)
                    distinct_ratios.insert(0, 0)
                self.assertEqual(
                    [e[2] for e in build_events],
                    distinct_ratios,
                )
                self.assertIsNone(build_events[0][5])
                for event in build_events[1:]:
                    self.assertIsNotNone(event[5])
                    self.assertIs(event[7], build_events[0][7])
                    self.assertIs(event[8], build_events[0][8])
                for compressed_rope in (False, True):
                    rope_events = [e for e in build_events if (e[2] != 0) == compressed_rope]
                    if not rope_events:
                        continue
                    self.assertIsNone(rope_events[0][6])
                    for event in rope_events[1:]:
                        self.assertIsNotNone(event[6])
                        self.assertIs(event[9], rope_events[0][9])

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


class _CommonReuseAttentionStub:
    from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

    _build_shared_prefill_meta = AttentionFP8._build_shared_prefill_meta
    _validate_reusable_prefill_common = (
        AttentionFP8._validate_reusable_prefill_common
    )

    def __init__(self, compress_ratio: int, freqs_cis: torch.Tensor):
        self.compress_ratio = compress_ratio
        self.freqs_cis = freqs_cis
        self.rope_head_dim = 4
        self.window_size = 3
        self._cp_ctx = None
        self.swa_build_calls = 0
        self.csa_meta = object()
        self.hca_meta = object()
        self.slot_compaction = object()

    def _ensure_freqs_cis_bound(self) -> None:
        pass

    def _build_swa_prefill_meta_varlen(
        self, *, topk_length_kv_full: torch.Tensor, **kwargs
    ):
        from rtp_llm.models_py.modules.dsv4.fp8.attention import SwaPrefillMeta

        self.swa_build_calls += 1
        is_swa_only = self.compress_ratio == 0
        return SwaPrefillMeta(
            slot_mapping=torch.arange(5, dtype=torch.long),
            query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
            combined_seq_lens=torch.tensor([2, 7], dtype=torch.int32),
            topk_length_kv_full=topk_length_kv_full,
            combined_gather_lens=(
                torch.tensor([2, 6], dtype=torch.int32) if is_swa_only else None
            ),
            combined_gather_len_max=6 if is_swa_only else 0,
            M=6 if is_swa_only else 0,
            cache_seq_lens=(
                torch.tensor([0, 4], dtype=torch.int32) if is_swa_only else None
            ),
            cache_gather_lens=(
                torch.tensor([0, 3], dtype=torch.int32) if is_swa_only else None
            ),
            prefix_len_max=1 if is_swa_only else 0,
            combined_indices=(
                torch.zeros((5, 3), dtype=torch.int32) if is_swa_only else None
            ),
            combined_lens=(
                torch.ones(5, dtype=torch.int32) if is_swa_only else None
            ),
            slot_in_flat=(torch.arange(5, dtype=torch.long) if is_swa_only else None),
            cache_slot_mapping=(
                torch.zeros((2, 3), dtype=torch.long) if is_swa_only else None
            ),
            slot_compaction=self.slot_compaction,
            cache_compaction=object() if is_swa_only else None,
        )

    def _build_csa_prefill_meta(self, *args, **kwargs):
        return self.csa_meta

    def _build_hca_prefill_meta(self, *args, **kwargs):
        return self.hca_meta


class CommonMetadataReuseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.zeros((5, 8), dtype=torch.bfloat16)
        self.swa_freqs = torch.complex(
            torch.arange(64, dtype=torch.float32).view(32, 2),
            torch.zeros((32, 2), dtype=torch.float32),
        )
        self.compressed_freqs = torch.complex(
            torch.arange(64, 128, dtype=torch.float32).view(32, 2),
            torch.ones((32, 2), dtype=torch.float32),
        )
        self.cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        self.input_lengths = torch.tensor([2, 3], dtype=torch.int32)
        self.prefix_lengths = torch.tensor([0, 4], dtype=torch.int32)
        self.sp_per_req = self.prefix_lengths.to(torch.int64)
        self.position_ids = torch.tensor([0, 1, 4, 5, 6], dtype=torch.long)
        self.req_id_per_token = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)

    def _build(self, stub, *, reuse_common_meta=None, reuse_freqs_meta=None):
        return stub._build_shared_prefill_meta(
            self.x,
            0,
            sp_per_req=self.sp_per_req,
            cu_seqlens=self.cu_seqlens,
            batch_size=2,
            input_lengths=self.input_lengths,
            prefix_lengths=self.prefix_lengths,
            position_ids=self.position_ids,
            req_id_per_token=self.req_id_per_token,
            max_seqlen_q=3,
            reuse_common_meta=reuse_common_meta,
            reuse_freqs_meta=reuse_freqs_meta,
        )

    def test_ratio_groups_reuse_exact_common_and_rope_identities(self) -> None:
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_ops_triton

        def fake_topk(win, cu_seqlens, positions, prefix_lengths, req_ids):
            return (
                torch.arange(15, dtype=torch.int32).view(5, 3),
                torch.tensor([1, 2, 1, 2, 3], dtype=torch.int32),
            )

        source_stub = _CommonReuseAttentionStub(0, self.swa_freqs)
        csa_stub = _CommonReuseAttentionStub(4, self.compressed_freqs)
        hca_stub = _CommonReuseAttentionStub(128, self.compressed_freqs)
        reference_stub = _CommonReuseAttentionStub(4, self.compressed_freqs)
        with patch.object(
            _swa_ops_triton,
            "compute_window_topk_and_length_varlen",
            side_effect=fake_topk,
        ) as compute_topk:
            source = self._build(source_stub)
            csa = self._build(csa_stub, reuse_common_meta=source)
            hca = self._build(
                hca_stub,
                reuse_common_meta=source,
                reuse_freqs_meta=csa,
            )

            # Full-build reference is outside the broadcast reuse path. Its
            # values must be bit-exact with the reused metadata.
            reference = self._build(reference_stub)

        self.assertEqual(compute_topk.call_count, 2)
        self.assertEqual(source_stub.swa_build_calls, 1)
        self.assertEqual(csa_stub.swa_build_calls, 0)
        self.assertEqual(hca_stub.swa_build_calls, 0)
        self.assertEqual(reference_stub.swa_build_calls, 1)
        self.assertIsNot(csa.freqs_cis, source.freqs_cis)
        self.assertIs(hca.freqs_cis, csa.freqs_cis)
        self.assertIs(csa.topk_idxs, source.topk_idxs)
        self.assertIs(hca.topk_idxs, source.topk_idxs)
        self.assertIs(csa.row_seqlens_full, source.row_seqlens_full)
        self.assertIs(hca.row_seqlens_full, source.row_seqlens_full)
        self.assertIs(csa.swa_meta.slot_mapping, source.swa_meta.slot_mapping)
        self.assertIs(hca.swa_meta.slot_mapping, source.swa_meta.slot_mapping)
        self.assertIs(
            csa.swa_meta.query_start_loc, source.swa_meta.query_start_loc
        )
        self.assertIs(
            csa.swa_meta.combined_seq_lens,
            source.swa_meta.combined_seq_lens,
        )
        self.assertIs(
            csa.swa_meta.topk_length_kv_full,
            source.swa_meta.topk_length_kv_full,
        )
        self.assertIs(
            csa.swa_meta.slot_compaction, source.swa_meta.slot_compaction
        )
        self.assertIsNone(csa.swa_meta.combined_gather_lens)
        self.assertIsNone(csa.swa_meta.combined_indices)
        self.assertIsNone(csa.swa_meta.cache_slot_mapping)
        self.assertIsNone(csa.swa_meta.cache_compaction)
        self.assertEqual(csa.swa_meta.M, 0)
        self.assertIs(csa.csa_meta, csa_stub.csa_meta)
        self.assertIs(hca.hca_meta, hca_stub.hca_meta)

        for reused, expected in (
            (csa.freqs_cis, reference.freqs_cis),
            (hca.freqs_cis, reference.freqs_cis),
            (csa.topk_idxs, reference.topk_idxs),
            (hca.topk_idxs, reference.topk_idxs),
            (csa.row_seqlens_full, reference.row_seqlens_full),
            (csa.swa_meta.slot_mapping, reference.swa_meta.slot_mapping),
            (
                csa.swa_meta.query_start_loc,
                reference.swa_meta.query_start_loc,
            ),
            (
                csa.swa_meta.combined_seq_lens,
                reference.swa_meta.combined_seq_lens,
            ),
            (
                csa.swa_meta.topk_length_kv_full,
                reference.swa_meta.topk_length_kv_full,
            ),
        ):
            self.assertTrue(torch.equal(reused, expected))
            self.assertEqual(reused.dtype, expected.dtype)
            self.assertEqual(reused.shape, expected.shape)
        self.assertEqual(csa.any_cont, reference.any_cont)

    def test_omitted_common_keeps_full_build_fallback(self) -> None:
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_ops_triton

        stub = _CommonReuseAttentionStub(128, self.compressed_freqs)
        with patch.object(
            _swa_ops_triton,
            "compute_window_topk_and_length_varlen",
            return_value=(
                torch.zeros((5, 3), dtype=torch.int32),
                torch.ones(5, dtype=torch.int32),
            ),
        ) as compute_topk:
            meta = self._build(stub)

        compute_topk.assert_called_once()
        self.assertEqual(stub.swa_build_calls, 1)
        self.assertIs(meta.hca_meta, stub.hca_meta)


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


if __name__ == "__main__":
    unittest.main()
