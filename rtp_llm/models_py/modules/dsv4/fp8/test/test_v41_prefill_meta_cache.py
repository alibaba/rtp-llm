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

from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8  # noqa: E402
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
                self.assertEqual(len(calls), 2)
                self.assertNotIn("prefill_meta_common", shared)

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


if __name__ == "__main__":
    unittest.main()
