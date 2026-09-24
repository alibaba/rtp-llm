"""V4.1 broadcast prefill-meta per-forward cache.

Covers ``AttentionV41FP8._build_shared_prefill_meta``'s
``prefill_meta_common`` cache in the shared per-forward state:

* bucket builds with the SAME rope kind and the SAME input tensors return
  the first build's meta object without re-running the SWA planner
  (``AttentionFP8._build_shared_prefill_meta`` is invoked once);
* a different rope kind (base vs compressed) or different input-tensor
  identities rebuild;
* ``_begin_forward`` drops the cache so a later forward cannot observe a
  stale entry;
* the request-slices decoration is applied to the cached result exactly
  like the fresh build.
"""

import os
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rtp_llm.models_py.modules.dsv4.attn_type import DECODER_SWA_KV, SWA_KV
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8  # noqa: E402
from rtp_llm.models_py.modules.dsv4.fp8.attention import bind_attn_cache
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import (  # noqa: E402
    AttentionV41FP8,
)
from rtp_llm.models_py.modules.dsv4.fp8.prefill_meta import (
    build_and_propagate_prefill_meta_fp8,
    release_v41_prefill_shared,
)


class _FakeMeta(NamedTuple):
    batch_size: int = 1
    seqlen: int = 8
    cp_on: bool = False
    freqs_cis: object = None
    request_row_slices: object = None
    workspace: object = None
    swa_meta: object = None


def _make_attn(ratio: int, layer_id: int, shared: dict) -> AttentionV41FP8:
    attn = AttentionV41FP8.__new__(AttentionV41FP8)
    torch.nn.Module.__init__(attn)
    attn.layer_id = layer_id
    attn.compress_ratio = ratio
    attn.window_size = 128
    attn._shared_attention = shared
    shared.setdefault("layers", {})[layer_id] = attn
    # Rope-construction parameters (model-level constants in production).
    attn._rope_base = 160000
    attn._rope_max_seq_len = 131200
    attn._rope_o_seq_len = 131200
    attn._rope_factor = 16.0
    attn._rope_beta_fast = 32
    attn._rope_beta_slow = 1
    attn._rope_dim = 64
    attn._cp_ctx = None
    attn._kv_cache = None
    attn._block_tables_by_type = None
    return attn


class _ParentCalls:
    def __init__(self):
        self.count = 0
        self.seen_args = []

    # ``super()._build_shared_prefill_meta`` looks the attribute up on
    # ``AttentionFP8``; a plain callable class attribute is NOT bound, so the
    # call receives exactly the override's forwarded (x, start_pos, **kwargs).
    def __call__(self, *args, **kwargs):
        self.count += 1
        self.seen_args.append((args, kwargs))
        return _FakeMeta()


def _inputs():
    return dict(
        sp_per_req=torch.zeros(1, dtype=torch.int64),
        cu_seqlens=torch.tensor([0, 8], dtype=torch.int64),
        batch_size=1,
        input_lengths=torch.tensor([8], dtype=torch.int32),
        prefix_lengths=torch.tensor([0], dtype=torch.int32),
        position_ids=torch.arange(8, dtype=torch.int64),
        req_id_per_token=torch.zeros(8, dtype=torch.int32),
        max_seqlen_q=8,
    )


class V41PrefillMetaCacheTest(unittest.TestCase):
    def test_shared_tensors_live_until_their_last_consumer(self):
        layers = {
            i: SimpleNamespace(
                kv_source_layer_id=2 if i < 8 else 8,
                index_source_layer_id=2 if i < 4 else (4 if i < 8 else 8),
                is_index_source=i in (2, 4, 8),
            )
            for i in range(2, 10)
        }
        shared = {
            "layers": layers,
            "global": {2: torch.ones(2)},
            "topk": {2: torch.ones(2)},
            "candidates": torch.ones(2),
            "prefill_chunk_meta": torch.ones(2),
        }
        shared["prefill_index_plan"] = (shared["topk"][2], torch.ones(2))
        global_ref = weakref.ref(shared["global"][2])
        topk_ref = weakref.ref(shared["topk"][2])
        candidate_ref = weakref.ref(shared["candidates"])

        release_v41_prefill_shared(shared, 2)
        self.assertIsNotNone(global_ref())
        self.assertIsNotNone(topk_ref())
        release_v41_prefill_shared(shared, 3)
        self.assertIsNone(topk_ref())
        self.assertNotIn("prefill_index_plan", shared)
        self.assertIsNotNone(global_ref())
        release_v41_prefill_shared(shared, 7)
        self.assertIsNone(global_ref())
        self.assertNotIn("prefill_chunk_meta", shared)
        self.assertIsNotNone(candidate_ref())
        release_v41_prefill_shared(shared, 8)
        self.assertIsNone(candidate_ref())
        self.assertIs(shared["layers"], layers)

    def test_broadcast_releases_bucket_cache_after_success_or_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                shared = {}
                layers = [_make_attn(r, i, shared) for i, r in enumerate((0, 2, 1))]
                model = SimpleNamespace(
                    layers=[SimpleNamespace(attn=a) for a in layers]
                )
                for attn in layers:
                    attn._prefill_meta_shared = object()
                calls = []

                def parent(*args, **kwargs):
                    calls.append(1)
                    if fail and len(calls) == 2:
                        raise RuntimeError("second bucket failed")
                    return _FakeMeta()

                with patch.object(
                    AttentionFP8, "_build_shared_prefill_meta", parent
                ), patch.object(AttentionFP8, "_ensure_freqs_cis_bound"):

                    def build():
                        build_and_propagate_prefill_meta_fp8(
                            model,
                            torch.zeros(8, 5120),
                            0,
                            None,
                            None,
                            workspace=None,
                            **_inputs(),
                        )

                    if fail:
                        with self.assertRaisesRegex(
                            RuntimeError, "second bucket failed"
                        ):
                            build()
                        self.assertTrue(
                            all(a._prefill_meta_shared is None for a in layers)
                        )
                    else:
                        build()
                        self.assertTrue(
                            all(a._prefill_meta_shared is not None for a in layers)
                        )
                self.assertEqual(len(calls), 2 if fail else 3)
                self.assertNotIn("prefill_meta_common", shared)

    def test_broadcast_reuse_restores_full_swa_and_preserves_parent_rope(self):
        shared = {}
        attn = _make_attn(2, 2, shared)
        x, inputs = torch.zeros(8, 5120), _inputs()
        full_swa, current_freqs = object(), object()
        source = _FakeMeta(swa_meta=full_swa, freqs_cis=object())
        # A prior same-key entry must not bypass parent RoPE identity checks.
        self._build(attn, x, 0, inputs)
        with patch.object(
            AttentionFP8,
            "_build_shared_prefill_meta",
            return_value=_FakeMeta(swa_meta=None, freqs_cis=current_freqs),
        ) as parent:
            actual = attn._build_shared_prefill_meta(
                x, 0, **inputs, reuse_common_meta=source, reuse_freqs_meta=source
            )
        self.assertEqual(parent.call_count, 1)
        self.assertIs(parent.call_args.kwargs["reuse_common_meta"], source)
        self.assertIs(parent.call_args.kwargs["reuse_freqs_meta"], source)
        self.assertIs(actual.swa_meta, full_swa)
        self.assertIs(actual.freqs_cis, current_freqs)
        self.assertEqual(actual.request_row_slices, (slice(0, 8),))
        self.assertEqual(attn.compress_ratio, 2)

    def test_reuse_exception_restores_ratio_and_does_not_publish(self):
        attn = _make_attn(1, 20, {})
        with patch.object(
            AttentionFP8,
            "_build_shared_prefill_meta",
            side_effect=RuntimeError("reuse failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "reuse failed"):
                attn._build_shared_prefill_meta(
                    torch.zeros(8, 5120),
                    0,
                    **_inputs(),
                    reuse_common_meta=_FakeMeta(swa_meta=object()),
                )
        self.assertEqual(attn.compress_ratio, 1)
        self.assertEqual(attn._shared_attention["prefill_meta_common"], {})

    def _build(self, attn, x, start_pos, inputs):
        with patch.object(
            AttentionFP8, "_build_shared_prefill_meta", _ParentCalls()
        ) as parent:
            meta = attn._build_shared_prefill_meta(x, start_pos, **inputs)
            return meta, parent

    def test_same_rope_kind_and_inputs_reuse_first_build(self):
        shared = {}
        layer2 = _make_attn(2, 2, shared)
        layer20 = _make_attn(1, 20, shared)
        x = torch.zeros(8, 5120, dtype=torch.bfloat16)
        inputs = _inputs()

        meta2, parent = self._build(layer2, x, 0, inputs)
        self.assertEqual(parent.count, 1)
        # Second compressed-rope bucket (layer 20, ratio 1): identical inputs,
        # same rope parameters -> served from the cache, parent not re-run.
        meta20, parent20 = self._build(layer20, x, 0, inputs)
        self.assertEqual(parent20.count, 0)
        self.assertIs(meta2, meta20)
        self.assertEqual(len(shared["prefill_meta_common"]), 1)

    def test_base_rope_rebuilds_then_compressed_reuses(self):
        shared = {}
        layer0 = _make_attn(0, 0, shared)
        layer2 = _make_attn(2, 2, shared)
        layer20 = _make_attn(1, 20, shared)
        x = torch.zeros(8, 5120, dtype=torch.bfloat16)
        inputs = _inputs()

        meta0, parent0 = self._build(layer0, x, 0, inputs)
        self.assertEqual(parent0.count, 1)
        meta2, parent2 = self._build(layer2, x, 0, inputs)
        self.assertEqual(parent2.count, 1)
        self.assertIsNot(meta0, meta2)
        meta20, parent20 = self._build(layer20, x, 0, inputs)
        self.assertEqual(parent20.count, 0)
        self.assertIs(meta2, meta20)
        # Two rope kinds cached.
        self.assertEqual(len(shared["prefill_meta_common"]), 2)

    def test_different_input_identity_rebuilds(self):
        shared = {}
        layer2 = _make_attn(2, 2, shared)
        x = torch.zeros(8, 5120, dtype=torch.bfloat16)
        meta_a, parent_a = self._build(layer2, x, 0, _inputs())
        self.assertEqual(parent_a.count, 1)
        # Fresh input tensors (new forward) -> miss.
        meta_b, parent_b = self._build(layer2, x, 0, _inputs())
        self.assertEqual(parent_b.count, 1)
        self.assertIsNot(meta_a, meta_b)
        self.assertEqual(len(shared["prefill_meta_common"]), 2)

    def test_different_scalars_rebuild(self):
        shared = {}
        layer2 = _make_attn(2, 2, shared)
        x = torch.zeros(8, 5120, dtype=torch.bfloat16)
        inputs = _inputs()
        meta_a, _ = self._build(layer2, x, 0, inputs)
        inputs2 = dict(inputs, max_seqlen_q=16)
        meta_b, parent_b = self._build(layer2, x, 0, inputs2)
        self.assertEqual(parent_b.count, 1)
        self.assertIsNot(meta_a, meta_b)

    def test_begin_forward_drops_cache(self):
        shared = {}
        layer0 = _make_attn(0, 0, shared)
        layer2 = _make_attn(2, 2, shared)
        x = torch.zeros(8, 5120, dtype=torch.bfloat16)
        inputs = _inputs()
        self._build(layer2, x, 0, inputs)
        self.assertIn("prefill_meta_common", shared)
        # The min-layer-id layer clears the per-forward shared state.
        layer0._begin_forward()
        self.assertNotIn("prefill_meta_common", shared)
        # After the clear, the same inputs rebuild.
        _, parent = self._build(layer2, x, 0, inputs)
        self.assertEqual(parent.count, 1)

    def test_request_slices_applied_to_cached_result(self):
        shared = {}
        layer2 = _make_attn(2, 2, shared)
        layer20 = _make_attn(1, 20, shared)
        x = torch.zeros(8, 5120, dtype=torch.bfloat16)
        inputs = _inputs()
        # Production request slices are mandatory even with a stale A/B env.
        with patch.dict(os.environ, {"DSV41_PREFILL_REQUEST_SLICES": "0"}):
            meta2, parent = self._build(layer2, x, 0, inputs)
            self.assertEqual(parent.count, 1)
            self.assertEqual(meta2.request_row_slices, (slice(0, 8),))
            meta20, parent20 = self._build(layer20, x, 0, inputs)
            self.assertEqual(parent20.count, 0)
            self.assertEqual(meta20.request_row_slices, (slice(0, 8),))
            self.assertIs(meta2, meta20)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for actual SWA planner")
class V41CommonMetaIntegrationTest(unittest.TestCase):
    def assert_meta_equal(self, got, want):
        if isinstance(want, torch.Tensor):
            self.assertEqual(got.dtype, want.dtype)
            self.assertEqual(got.shape, want.shape)
            self.assertTrue(torch.equal(got, want))
        elif isinstance(want, tuple):
            self.assertEqual(len(got), len(want))
            for a, b in zip(got, want):
                self.assert_meta_equal(a, b)
        else:
            self.assertEqual(got, want)

    def fixture(self, rank, prefixes, *, bound=True, compressed_first=False):
        lengths = (9, 3, 5)
        chunks = tuple(2 * ((n + 7) // 8) for n in lengths)
        positions, requests = [], []
        for b, (n, p, chunk) in enumerate(zip(lengths, prefixes, chunks)):
            half = chunk // 2
            local = list(range(rank * half, (rank + 1) * half)) + list(
                range((7 - rank) * half, (8 - rank) * half)
            )
            positions.extend(p + min(i, n - 1) for i in local)
            requests.extend([b] * chunk)
        device = "cuda"
        tensor = lambda values, dtype=torch.int32: torch.tensor(
            values, device=device, dtype=dtype
        )
        cp = SimpleNamespace(
            cp_size=4,
            cp_rank=rank,
            kv_cache_sharded=True,
            seq_len_full=sum(lengths),
            chunk_lengths_per_req=chunks,
            input_lengths_global=tensor(lengths),
            cu_seqlens_global=tensor([0, 9, 12, 17]),
            global_positions=tensor(positions, torch.int64),
        )
        ratios = (2, 1, 2, 1) if compressed_first else (0, 2, 1, 2, 1)
        normal_count = len(ratios) - 2
        raw = [
            torch.zeros((8, 18048), device=device, dtype=torch.uint8) for _ in range(2)
        ]
        mapping = []
        for i in range(len(ratios)):
            row = [-1] * (DECODER_SWA_KV + 1)
            row[SWA_KV if i < normal_count else DECODER_SWA_KV] = int(i >= normal_count)
            mapping.append(row)
        cache = (
            SimpleNamespace(
                group_region_names=[SWA_KV, DECODER_SWA_KV],
                layer_region_to_group_id=mapping,
                seq_size_per_block=128,
                group_seq_size_per_block=[128, 128],
                get_layer_cache=lambda layer, region: SimpleNamespace(
                    kv_cache_base=raw[int(layer >= normal_count)]
                ),
            )
            if bound
            else None
        )
        tables = (
            {
                SWA_KV: tensor([[1, 2]] * 3),
                DECODER_SWA_KV: tensor([[3, 4]] * 3),
            }
            if bound
            else None
        )
        base = torch.arange(4096 * 32, device=device, dtype=torch.float32).reshape(
            4096, 32
        )
        compressed = base + 0.5
        layers = []
        for i, ratio in enumerate(ratios):
            a = _make_attn(ratio, i, {})
            a.rope_head_dim = 64
            a.head_dim = 512
            a._cp_ctx = cp
            a.freqs_cis = compressed if ratio else base
            a._pool_spec = {SWA_KV: (torch.uint8, 528)}
            layers.append(a)
        x = torch.zeros(sum(chunks), 5120, device=device, dtype=torch.bfloat16)
        inputs = dict(
            sp_per_req=tensor(prefixes, torch.int64),
            cu_seqlens=tensor([0, chunks[0], sum(chunks[:2]), sum(chunks)]),
            batch_size=3,
            input_lengths=tensor(chunks),
            prefix_lengths=tensor(prefixes),
            position_ids=cp.global_positions,
            req_id_per_token=tensor(requests),
            max_seqlen_q=max(chunks),
        )
        return layers, x, inputs, cache, tables, normal_count

    def test_actual_parent_cp4_broadcast_exact_region_rope_and_reentry(self):
        for rank in range(4):
            for prefixes, bound, compressed_first in (
                ((127, 0, 129), True, False),
                ((0, 0, 0), True, False),
                ((127, 0, 129), False, False),
                ((1, 0, 513), True, True),
            ):
                with self.subTest(
                    rank=rank,
                    prefixes=prefixes,
                    bound=bound,
                    compressed_first=compressed_first,
                ):
                    layers, x, inputs, cache, tables, normal_count = self.fixture(
                        rank, prefixes, bound=bound, compressed_first=compressed_first
                    )
                    expected = []
                    for a in layers:
                        with bind_attn_cache(a, cache, tables):
                            expected.append(
                                a._build_shared_prefill_meta(x, 0, **inputs)
                            )
                        a._shared_attention.pop("prefill_meta_common", None)
                    model = SimpleNamespace(
                        layers=[SimpleNamespace(attn=a) for a in layers]
                    )
                    workspace = object()
                    calls = []
                    planner = AttentionFP8._build_swa_prefill_meta_varlen

                    def counted(attn, *args, **kwargs):
                        calls.append(attn._swa_cache_region)
                        return planner(attn, *args, **kwargs)

                    with patch.object(
                        AttentionFP8, "_build_swa_prefill_meta_varlen", counted
                    ):
                        build_and_propagate_prefill_meta_fp8(
                            model, x, 0, cache, tables, workspace=workspace, **inputs
                        )
                    self.assertEqual(len(calls), 2 if bound else 1)
                    for a, want in zip(layers, expected):
                        got = a._prefill_meta_shared
                        for field in want._fields:
                            if field not in ("workspace", "cp_ctx"):
                                self.assert_meta_equal(
                                    getattr(got, field), getattr(want, field)
                                )
                        self.assertIs(got.workspace, workspace)
                        self.assertIsNone(a._kv_cache)
                        self.assertEqual(got.freqs_cis_source_id, id(a.freqs_cis))
                    source = layers[0]._prefill_meta_shared.swa_meta
                    for a in layers[:normal_count]:
                        self.assertIs(a._prefill_meta_shared.swa_meta, source)
                    if bound:
                        other = layers[-1]._prefill_meta_shared.swa_meta
                        self.assertIsNot(other, source)
                        self.assertIs(layers[-2]._prefill_meta_shared.swa_meta, other)
                    # Full real parent must honor a different table identity even
                    # with an existing same-rope-kind cache entry and reuse source.
                    a = layers[1]
                    source_meta = layers[0]._prefill_meta_shared
                    with bind_attn_cache(a, cache, tables):
                        same = a._build_shared_prefill_meta(
                            x,
                            0,
                            **inputs,
                            reuse_common_meta=source_meta,
                            reuse_freqs_meta=a._prefill_meta_shared,
                        )
                        self.assertIs(same.freqs_cis, a._prefill_meta_shared.freqs_cis)
                        a.freqs_cis = a.freqs_cis + 7
                        different = a._build_shared_prefill_meta(
                            x,
                            0,
                            **inputs,
                            reuse_common_meta=source_meta,
                            reuse_freqs_meta=same,
                        )
                        self.assertTrue(
                            torch.equal(
                                different.freqs_cis,
                                a.freqs_cis.index_select(0, inputs["position_ids"]),
                            )
                        )
                        self.assertIsNot(different.freqs_cis, same.freqs_cis)
                        self.assertIs(different.swa_meta, source_meta.swa_meta)
                    # Same identities on a new broadcast still rebuild; errors
                    # clear propagated state, and a subsequent invocation recovers.
                    calls.clear()
                    with patch.object(
                        AttentionFP8, "_build_swa_prefill_meta_varlen", counted
                    ):
                        build_and_propagate_prefill_meta_fp8(
                            model, x, 0, cache, tables, workspace=workspace, **inputs
                        )
                    self.assertEqual(len(calls), 2 if bound else 1)
                    with patch.object(
                        layers[1],
                        "_build_shared_prefill_meta",
                        side_effect=RuntimeError("injected"),
                    ):
                        with self.assertRaisesRegex(RuntimeError, "injected"):
                            build_and_propagate_prefill_meta_fp8(
                                model,
                                x,
                                0,
                                cache,
                                tables,
                                workspace=workspace,
                                **inputs,
                            )
                    self.assertTrue(all(a._prefill_meta_shared is None for a in layers))
                    build_and_propagate_prefill_meta_fp8(
                        model, x, 0, cache, tables, workspace=workspace, **inputs
                    )


if __name__ == "__main__":
    unittest.main()
