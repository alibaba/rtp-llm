"""Local Prefill workspace equality and lifetime at the existing KV boundary.

Synthetic BF16 banks isolate workspace/index construction from codec arithmetic.
The explicit PPU class additionally exercises actual native ring staging; its
byte-domain decoder is deliberate, not evidence for a new codec or Attention.
"""

import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import attention_layer
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.prefill import (
    _build_reference_workspace,
    build_cp_prefill_workspace,
    build_step_context,
)
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_prefill_reuse import (
    _ProjectionBoundary,
    prefill_component,
)


ENV = attention_layer.LOCAL_PREFILL_WORKSPACE_ENV


def assert_bytes(test, actual, expected):
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.shape, expected.shape)
    test.assertTrue(torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    ))


class _WorkspaceFixture:
    def __init__(self, device, prefixes=(17,), lengths=(7,), *, ratio=2, source=0, queries=None):
        self.device, self.prefixes, self.lengths = device, prefixes, lengths
        self.totals = tuple(p + n for p, n in zip(prefixes, lengths))
        self.ratio, self.source = ratio, source
        self.prepared = []
        self.global_revision = self.swa_revision = 0
        self.geometry = NS(
            head_dim=8, window=4, ratio=lambda layer: self.ratio,
            kv_source_for=lambda layer: self.source,
        )
        self.cache = NS(
            sharded=False, lengths=self.totals, request_length=self.totals.__getitem__,
            prepare_source=self.prepared.append,
            descriptors=lambda layer: {"swa_kv": None, "global_kv_1": None, "global_kv_2": None},
            read_source=self.read_source, read_swa=self.read_swa,
        )
        self.ops = NS(decode_kv=lambda packed, *, pool_tag, descriptor, out: out.copy_(packed))
        self.state = NS(
            prefill=True, idle=False, cache=self.cache, shared={},
            lengths=lengths, query_prefixes=prefixes,
        )
        if queries is None:
            queries = [list(range(p, p + n)) for p, n in zip(prefixes, lengths)]
        self.chunks = tuple(map(len, queries))
        positions = torch.tensor([p for query in queries for p in query], device=device, dtype=torch.int64)
        requests = torch.tensor([rid for rid, query in enumerate(queries) for _ in query],
                                device=device, dtype=torch.int64)
        offsets = [0]
        for chunk in self.chunks:
            offsets.append(offsets[-1] + chunk)
        self.context = build_step_context(
            positions, requests, torch.tensor(offsets, device=device, dtype=torch.int32),
        )
        columns = torch.tensor([0, -1, 0, 2, 999], device=device, dtype=torch.int32)
        self.selected = columns.expand(positions.numel(), -1).clone() if ratio else None

    def read_source(self, source, request, *, global_entries):
        values = torch.arange(global_entries, device=self.device) + request * 37 + source * 3
        values = values + self.global_revision * 11
        return (values[:, None].expand(-1, 8).to(torch.bfloat16).contiguous(),)

    def read_swa(self, layer, request, positions):
        values = positions + request * 19 + layer * 5 + self.swa_revision * 13
        return values[:, None].expand(-1, 8).to(torch.bfloat16).contiguous()

    def args(self, layer=2, starts=None):
        return dict(
            ops=self.ops, cache=self.cache, layer_id=layer, geometry=self.geometry,
            context=self.context, selected_blocks=self.selected,
            window=self.geometry.window, swa_starts=starts,
        )

    def build(self, layer=2, *, cp=None, starts=None):
        layout = attention_layer._local_prefill_workspace_layout(
            self.state, cp, self.geometry, layer,
        )
        if layout is None:
            raise AssertionError("fixture must be admitted to the local route")
        return build_cp_prefill_workspace(
            **self.args(layer, starts), cp=layout, shared=self.state.shared, q8=False,
        )

    def compare(self, test, layer=2, *, cp=None, starts=None):
        actual = self.build(layer, cp=cp, starts=starts)
        expected = _build_reference_workspace(**self.args(layer, starts))
        for left, right in zip(actual, expected):
            assert_bytes(test, left, right)
        return actual


class _WorkspaceCases:
    @torch.inference_mode()
    def test_full_local_banks_indices_and_topk_holes(self):
        # Ratio16 tests generic index arithmetic only, not a native model mode.
        with patch.dict(os.environ, {ENV: "1"}):
            for ratio in (0, 1, 2, 16):
                with self.subTest(ratio=ratio):
                    fixture = _WorkspaceFixture(
                        self.device, prefixes=(1, 9, 33), lengths=(3, 0, 7),
                        ratio=ratio, source=None if ratio == 0 else 0,
                    )
                    fixture.compare(self, starts={2: 34})
                    self.assertEqual(fixture.prepared, [] if ratio == 0 else [0])

    @torch.inference_mode()
    def test_compact_cp1_keeps_absolute_positions_and_bounded_swa(self):
        with patch.dict(os.environ, {ENV: "1"}):
            fixture = _WorkspaceFixture(
                self.device, prefixes=(17, 64), lengths=(9, 6),
                queries=([22, 23, 24, 25], [67, 68, 69]),
            )
            compact = NS(cp_size=1, chunk_lengths_per_req=fixture.chunks)
            self.assertIs(attention_layer._local_prefill_workspace_layout(
                fixture.state, compact, fixture.geometry, 20,
            ), compact)
            fixture.compare(self, layer=20, cp=compact, starts={0: 22, 1: 67})

    @torch.inference_mode()
    def test_full_to_compact_rebuilds_even_without_explicit_shared_clear(self):
        with patch.dict(os.environ, {ENV: "1"}):
            fixture = _WorkspaceFixture(self.device, prefixes=(17,), lengths=(9,))
            full = fixture.compare(self, layer=20)
            full_layout = fixture.state.shared["local_prefill_workspace_layout"]
            compact_queries = _WorkspaceFixture(
                self.device, prefixes=fixture.prefixes, lengths=fixture.lengths,
                queries=([22, 23, 24, 25],),
            )
            fixture.context = compact_queries.context
            fixture.selected = compact_queries.selected
            compact = NS(cp_size=1, chunk_lengths_per_req=(4,))
            actual = fixture.compare(self, layer=21, cp=compact, starts={0: 22})
            self.assertIsNot(full_layout, compact)
            self.assertNotEqual(full[0].data_ptr(), actual[0].data_ptr())
            self.assertEqual(full[1].shape[0], 9)
            self.assertEqual(actual[1].shape[0], 4)
            self.assertEqual(fixture.prepared, [0, 0])

    @torch.inference_mode()
    def test_swa_changes_refresh_shared_buffer_but_not_global_bank(self):
        with patch.dict(os.environ, {ENV: "1"}):
            fixture = _WorkspaceFixture(self.device)
            first = fixture.compare(self, layer=2)
            saved = tuple(value.clone() for value in first)
            fixture.swa_revision += 1
            second = fixture.compare(self, layer=3)
            self.assertEqual(first[0].data_ptr(), second[0].data_ptr())
            self.assertIs(first[1], second[1])
            self.assertIs(first[2], second[2])
            count = fixture.totals[0] // fixture.ratio
            assert_bytes(self, second[0][:count], saved[0][:count])
            self.assertFalse(torch.equal(second[0][count:], saved[0][count:]))
            self.assertEqual(fixture.prepared, [0])

    @torch.inference_mode()
    def test_new_selection_source_and_layout_invalidate_the_right_cache(self):
        with patch.dict(os.environ, {ENV: "1"}):
            fixture = _WorkspaceFixture(self.device)
            first = fixture.compare(self)
            fixture.selected = fixture.selected.flip(1).contiguous()
            second = fixture.compare(self)
            self.assertIsNot(first[1], second[1])
            self.assertFalse(torch.equal(first[1], second[1]))
            self.assertEqual(first[0].data_ptr(), second[0].data_ptr())
            fixture.source = 1
            third = fixture.compare(self)
            self.assertNotEqual(second[0].data_ptr(), third[0].data_ptr())
            self.assertEqual(fixture.prepared, [0, 1])
            compact = NS(cp_size=1, chunk_lengths_per_req=fixture.chunks)
            fourth = fixture.compare(self, cp=compact)
            self.assertNotEqual(third[0].data_ptr(), fourth[0].data_ptr())
            self.assertEqual(fixture.prepared, [0, 1, 1])
            fixture.ratio = 1
            fixture.compare(self, cp=compact)
            self.assertEqual(fixture.prepared, [0, 1, 1, 1])

    @torch.inference_mode()
    def test_new_forward_never_reuses_old_global_values(self):
        with patch.dict(os.environ, {ENV: "1"}):
            fixture = _WorkspaceFixture(self.device)
            previous = fixture.compare(self)
            old = previous[0].clone()
            old_token = fixture.state.shared["local_prefill_workspace_layout"]
            fixture.state.shared = {}
            fixture.global_revision += 1
            current = fixture.compare(self)
            self.assertIsNot(old_token, fixture.state.shared["local_prefill_workspace_layout"])
            self.assertNotEqual(previous[0].data_ptr(), current[0].data_ptr())
            self.assertFalse(torch.equal(old, current[0]))
            self.assertEqual(fixture.prepared, [0, 0])

    @torch.inference_mode()
    def test_body_selects_only_explicit_local_prefill_route(self):
        for enabled, phase, sharded, expected in (
            ("0", True, False, "reference"),
            ("1", True, False, "local"),
            ("1", False, False, "reference"),
            ("1", True, True, "cp"),
        ):
            with self.subTest(enabled=enabled, phase=phase, sharded=sharded):
                with prefill_component(4, indexed=False, device=self.device) as (attn, _, run, _, hidden):
                    state = attn.runtime.current
                    state.prefill, state.cache.sharded = phase, sharded
                    state.lengths = (4,)
                    attn.geometry.kv_source_for = lambda layer: 0
                    if not sharded:
                        attn._cp_ctx = None
                    original_cp = attn._cp_ctx
                    calls = []

                    def vectorized(**kwargs):
                        calls.append(("cp" if sharded else "local", kwargs))
                        raise _ProjectionBoundary()

                    def reference(**kwargs):
                        calls.append(("reference", kwargs))
                        raise _ProjectionBoundary()

                    with patch.dict(os.environ, {ENV: enabled}), patch.object(
                        attention_layer, "build_cp_prefill_workspace", new=vectorized,
                    ), patch.object(
                        attention_layer, "_build_reference_workspace", new=reference,
                    ), patch.object(
                        attention_layer, "prepare_packed_attention", return_value=(None, "disabled"),
                    ):
                        run(torch.randn(4, hidden, device=self.device, dtype=torch.bfloat16))
                    self.assertEqual([name for name, _ in calls], [expected])
                    self.assertIs(attn._cp_ctx, original_cp)
                    if expected == "local":
                        self.assertFalse(calls[0][1]["q8"])
                        self.assertIs(calls[0][1]["cp"], state.shared["local_prefill_workspace_layout"])
                    elif expected == "cp":
                        self.assertIs(calls[0][1]["cp"], original_cp)


class LocalPrefillWorkspaceCPU(_WorkspaceCases, unittest.TestCase):
    device = "cpu"

    def test_admission_preserves_diagnostics_decode_cp_and_missing_source(self):
        fixture = _WorkspaceFixture("cpu")
        with patch.dict(os.environ, {ENV: "0"}):
            self.assertIsNone(attention_layer._local_prefill_workspace_layout(None, None, None, 0))
        with patch.dict(os.environ, {ENV: "1"}):
            for field, value in (("prefill", False), ("idle", True)):
                old = getattr(fixture.state, field)
                setattr(fixture.state, field, value)
                self.assertIsNone(attention_layer._local_prefill_workspace_layout(
                    fixture.state, None, fixture.geometry, 2,
                ))
                setattr(fixture.state, field, old)
            for cp, diagnostic in ((NS(cp_size=8), None), (None, {})):
                self.assertIsNone(attention_layer._local_prefill_workspace_layout(
                    fixture.state, cp, fixture.geometry, 2, diagnostic=diagnostic,
                ))
            fixture.source = None
            self.assertIsNone(attention_layer._local_prefill_workspace_layout(
                fixture.state, None, fixture.geometry, 2,
            ))
            fixture.ratio = 0
            self.assertIsNotNone(attention_layer._local_prefill_workspace_layout(
                fixture.state, None, fixture.geometry, 0,
            ))


class _LocalPools:
    group_tags = ["swa_kv"]

    def __init__(self, device):
        self.handles = {
            (layer, "swa_kv"): NS(kv_cache_base=torch.full(
                (3, 8 * 528 + 16), 173, device=device, dtype=torch.uint8,
            ), tag="swa_kv", layer_id=layer)
            for layer in (0, 1)
        }
        table = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
        self.inputs = {"swa_kv": NS(kv_cache_block_id=table, kv_cache_block_id_device=table)}

    def get_layer_cache(self, layer, tag):
        return self.handles[layer, tag]

    def get_entries_per_block(self, tag):
        return 8

    def get_seq_size_per_block(self, tag):
        return 16

    def get_kernel_seq_size_per_block(self, tag):
        return 16


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test",
)
class LocalPrefillWorkspaceDevice(_WorkspaceCases, unittest.TestCase):
    device = "cuda"

    def setUp(self):
        self.assertTrue(is_ppu(), "device qualification requires a ZW PPU, not another CUDA backend")

    @torch.inference_mode()
    def test_native_local_ring_snapshot_retains_early_fresh_queries(self):
        native = _LocalPools(self.device)
        geometry = NS(
            num_layers=2, kv_source_layer_ids=(), head_dim=4, window=4,
            ratio=lambda layer: 0, kv_source_for=lambda layer: None,
        )
        cache = SharedNativeCache(
            PrefillPolicy().freeze({}), geometry, native, native.inputs, (37,),
            device=self.device, prefill=True,
        )
        self.assertFalse(cache.sharded)
        positions = torch.arange(17, 37, device=self.device)
        requests = torch.zeros_like(positions)
        context = build_step_context(positions, requests, torch.tensor([0, 20], device=self.device))
        state = NS(prefill=True, idle=False, cache=cache, shared={}, lengths=(20,), query_prefixes=(17,))
        ops = NS(decode_kv=lambda packed, *, pool_tag, descriptor, out: out.copy_(packed[:, :4]))
        previous = None
        with patch.dict(os.environ, {ENV: "1"}):
            for layer in (0, 1):
                old_positions = torch.arange(14, 17, device=self.device)
                old_rows = ((old_positions[:, None] + layer * 11) % 151).expand(-1, 528).to(torch.uint8).contiguous()
                fresh = ((positions[:, None] + layer * 11) % 151).expand(-1, 528).to(torch.uint8).contiguous()
                cache._copy_rows((layer, "swa_kv"), 0, old_positions, old_rows)
                cache.stage_swa(layer, positions, requests, fresh, (17,))
                staged_positions = torch.arange(14, 37, device=self.device)
                assert_bytes(self, cache.read_swa(layer, 0, staged_positions), torch.cat((old_rows, fresh)))
                self.assertFalse(torch.equal(
                    cache._copy_rows((layer, "swa_kv"), 0, positions[:1]), fresh[:1],
                ), "physical ring should have overwritten an early fresh row")
                args = dict(
                    ops=ops, cache=cache, layer_id=layer, geometry=geometry, context=context,
                    selected_blocks=None, window=4, swa_starts=None,
                )
                layout = attention_layer._local_prefill_workspace_layout(state, None, geometry, layer)
                actual = build_cp_prefill_workspace(**args, cp=layout, shared=state.shared, q8=False)
                expected = _build_reference_workspace(**args)
                for left, right in zip(actual, expected):
                    assert_bytes(self, left, right)
                if previous is not None:
                    self.assertEqual(previous.data_ptr(), actual[0].data_ptr())
                previous = actual[0]


if __name__ == "__main__":
    unittest.main()
