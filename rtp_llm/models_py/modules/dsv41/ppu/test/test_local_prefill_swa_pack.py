"""Exact existing SWA pack qualification through the real PPUOps entry.

The rollout scope is P eager Prefill only. This does not qualify Decode or
Graph. Existing native writer, SourceCache and Segments are comparison users;
this module does not replace their kernels or metadata contracts.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant, attention_layer, prefill
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import default_descriptor
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    LOCAL_PREFILL_SOURCE_CACHE_ENV, LOCAL_PREFILL_SWA_SEGMENTS_ENV, _StagedSwaSegments,
)
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_rows import state_for
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_source_cache import _SourceFixture
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_segments import FLAGS, compact_plan


ENV = "RTP_V41_PPU_FUSED_KV_ROW_PACK"


def _encode(ops, kv, flag, *, descriptor=None, tag="swa_kv"):
    with patch.dict(os.environ, {ENV: str(int(flag))}):
        return ops.encode_kv(kv, pool_tag=tag, descriptor=descriptor)


def _finite_patterns(device):
    bits = torch.arange(65536, dtype=torch.int32)
    bits = bits[(bits & 0x7F80) != 0x7F80]
    assert bits.numel() == 65280
    # Every finite BF16 pattern is an entire group, hence also the scale's
    # amax. This includes both signs of zero and every finite subnormal.
    values = bits.to(torch.int16).view(torch.bfloat16)
    return values[:, None].expand(-1, 32).contiguous().reshape(4080, 512).to(device)


def _boundary_rows(device):
    anchors = torch.tensor([1e-4] + [448.0 * 2.0**e for e in range(-22, 120)], dtype=torch.bfloat16)
    bits = anchors.view(torch.int16).int()
    positive = torch.stack((bits - 1, bits, bits + 1), dim=1).reshape(-1).clamp(0, 0x7F7F)
    all_bits = torch.cat((torch.tensor([0, 0x8000]), positive, positive | 0x8000))
    groups = all_bits.to(torch.int16).view(torch.bfloat16)[:, None].expand(-1, 32).contiguous()
    # All 254 signed NaN payloads and both infinities occur once in a group
    # with ordinary neighbors, detecting group-wide poison/scale behavior.
    nonfinite = torch.cat((torch.arange(0x7F80, 0x8000), torch.arange(0xFF80, 0x10000)))
    poisoned = torch.ones(256, 32, dtype=torch.bfloat16)
    poisoned.view(torch.int16)[:, 0] = nonfinite.to(torch.int16)
    poisoned[:, 1], poisoned[:, 2], poisoned[:, 3] = -0.0, 0.0, -448.0
    neighbors = torch.tensor([0x0001, 0x007F, 0x0080, 0x8001, 0x807F, 0x8080,
                              0x7F7F, 0xFF7F], dtype=torch.int32).to(torch.int16)
    poisoned.view(torch.int16)[:, 4:12] = neighbors[None]
    groups = torch.cat((groups, poisoned))
    padding = (-groups.shape[0]) % 16
    if padding:
        groups = torch.cat((groups, torch.zeros(padding, 32, dtype=torch.bfloat16)))
    return groups.reshape(-1, 512).to(device)


class _PackCases:
    def setUp(self):
        self.ops = PPUOps(num_routed_experts=8, num_local_experts=8)
        environment = patch.dict(os.environ, {
            **FLAGS, "TP_SIZE": "1", "PREFILL_CP_SIZE": "1",
            LOCAL_PREFILL_SOURCE_CACHE_ENV: "1",
            "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE": "1",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def assert_packed(self, kv):
        original = kv.clone()
        old = _encode(self.ops, kv, False)
        got = _encode(self.ops, kv, True)
        self.assertEqual(got.dtype, torch.uint8)
        self.assertEqual(got.shape, (kv.shape[0], 528))
        self.assertTrue(got.is_contiguous())
        assert_bytes(self, got, old)
        assert_bytes(self, kv, original)
        if kv.numel():
            self.assertNotEqual(got.untyped_storage().data_ptr(), kv.untyped_storage().data_ptr())
        decoded = []
        for packed in (old, got):
            backing = torch.full((kv.shape[0], 600), -7, dtype=torch.bfloat16, device=self.device)
            out = backing[:, :512]
            self.assertIs(self.ops.decode_kv(packed, pool_tag="swa_kv", out=out), out)
            self.assertTrue((backing[:, 512:] == -7).all())
            decoded.append(out.clone())
        assert_bytes(self, decoded[1], decoded[0])
        return got

    @torch.no_grad()
    def test_real_ops_sizes_readonly_alias_offset_and_snapshot_independence(self):
        cases = [(rows, 8) for rows in (0, 1, 140, 267, 4096, 4097)] + [(1, 1), (140, 1)]
        for rows, offset in cases:
            with self.subTest(rows=rows, offset=offset):
                owner = torch.full((rows * 512 + offset + 8,), -9, dtype=torch.bfloat16, device=self.device)
                kv = owner[offset:-8].view(rows, 512)
                if rows:
                    kv.copy_(((torch.arange(rows * 512, device=self.device) % 127) - 63).reshape(rows, 512) / 64)
                alias = kv.view(rows, 512)
                before = owner.clone()
                packed = self.assert_packed(alias)
                assert_bytes(self, owner, before)
                frozen = packed.clone()
                kv.zero_()
                assert_bytes(self, packed, frozen)
                self.assertTrue((owner[:offset] == -9).all() and (owner[-8:] == -9).all())

    @torch.no_grad()
    def test_every_finite_bf16_bit_pattern_is_exact(self):
        self.assert_packed(_finite_patterns(self.device))

    @torch.no_grad()
    def test_floor_exponent_neighbors_signed_nan_inf_and_group_poison_are_exact(self):
        self.assert_packed(_boundary_rows(self.device))

    @torch.no_grad()
    def test_negative_view_falls_back_with_exact_real_ops_bytes(self):
        owner = _boundary_rows(self.device)
        before = owner.clone()
        negative = torch._neg_view(owner)
        self.assertTrue(negative.is_neg() and negative.is_contiguous())
        with patch.object(_quant, "_ppu_fast_path", side_effect=AssertionError("raw pointer admission")):
            self.assertIsNone(_quant.try_pack_swa_rows(negative))
        old = _encode(self.ops, negative, False)
        got = _encode(self.ops, negative, True)
        assert_bytes(self, got, old)
        assert_bytes(self, owner, before)

    @torch.no_grad()
    def test_old_stage_sourcecache_segments_native_pools_and_workspace_match(self):
        cases = (
            (1, (129,), (4096,), False),
            (2, (129,), (4097,), False),
            (1, (17, 99, 64), (4096, 5, 0), True),
            (2, (129, 257, 31), (9, 0, 7), False),
        )
        for ratio, prefixes, lengths, compact in cases:
            for segments in (False, True):
                arms = []
                for flag in (False, True):
                    with self.subTest(ratio=ratio, lengths=lengths, compact=compact, segments=segments, pack=flag):
                        with patch.dict(os.environ, {LOCAL_PREFILL_SWA_SEGMENTS_ENV: str(int(segments))}):
                            fixture = _SourceFixture(self.device, ratio, prefixes, lengths)
                            cache = fixture.cache
                            state = state_for(self.device, prefixes, lengths)
                            state.cache = cache
                            state.runtime.policy = SharedPrefillPolicy(mode="compact")
                            state.runtime.model._is_speculative = False
                            cp = compact_plan(state) if compact else None
                            pos, req, real = state.full_rows(cp, selected_only=compact)
                            positions, requests = pos[real], req[real]
                            values = fixture.values(positions, 7)
                            packed = _encode(self.ops, values, flag)
                            bounded = tuple(p + max(n - 140, 0) for p, n in zip(prefixes, lengths)) if compact else None
                            options = dict(retain_rows=True, retain_prefix_tail=not compact, bounded_starts=bounded)
                            ticket = attention_layer._local_swa_stage_ranges(
                                state, cp, 0, pos, req, real, positions, requests, packed,
                                packed_plan=None, **options,
                            )
                            if segments:
                                self.assertIsNotNone(ticket)
                            # Warm actual authoritative Main+Indexer reads before
                            # SWA mutation, including zero-fresh nonzero-prefix.
                            banks = {(source, rid): fixture.packed(source, rid)
                                     for source in (0, 20) for rid in range(len(lengths))}
                            cache.stage_swa(0, positions, requests, packed, prefixes, **options,
                                            **({"local_ranges": ticket} if ticket is not None else {}))
                            self.assertEqual(all(isinstance(saved, _StagedSwaSegments) for saved in cache._fresh_swa.values()), segments)
                            snapshots = []
                            for rid, (prefix, size) in enumerate(zip(prefixes, lengths)):
                                first = prefix + max(size - 140, 0) if compact else max(0, prefix - 127)
                                domain = torch.arange(first, prefix + size, device=self.device)
                                snapshots.append(cache.read_swa(0, rid, domain).clone())
                            qpos, qreq, cu, valid = state.rows(cp)
                            context = prefill.build_step_context(qpos, qreq, cu, valid_mask=valid)
                            selected = torch.tensor([0, -1, 1, 0], device=self.device, dtype=torch.int32)[None].repeat(qpos.numel(), 1)
                            layout = cp if compact else NS(chunk_lengths_per_req=lengths, prefix_lengths_host=prefixes)
                            shared = {"local_prefill_workspace_layout": layout} if not compact else {}
                            workspaces = []
                            for source in (0, 20):
                                fixture.geometry.kv_source_for = lambda layer, source=source: source
                                args = dict(ops=self.ops, cache=cache, layer_id=0, geometry=fixture.geometry,
                                            context=context, selected_blocks=selected, window=128,
                                            swa_starts=dict(enumerate(bounded)) if compact else None)
                                expected = prefill._build_reference_workspace(**args)
                                actual = prefill.build_cp_prefill_workspace(**args, cp=layout, shared=shared, q8=False)
                                for left, right in zip(actual, expected):
                                    assert_bytes(self, left, right)
                                workspaces.append(tuple(t.clone() for t in actual))
                            pool_bytes = tuple(handle.kv_cache_base.clone() for handle in fixture.native.handles.values())
                            for handle in fixture.native.handles.values():
                                self.assertTrue((handle.kv_cache_base[:, -16:] == 173).all())
                            # The SWA stage and release retain source banks;
                            # snapshots are owned, independent of encode input.
                            packed.zero_()
                            for rid, (prefix, size) in enumerate(zip(prefixes, lengths)):
                                first = prefix + max(size - 140, 0) if compact else max(0, prefix - 127)
                                domain = torch.arange(first, prefix + size, device=self.device)
                                assert_bytes(self, cache.read_swa(0, rid, domain), snapshots[rid])
                            cache.release_swa(0)
                            self.assertFalse(cache._fresh_swa)
                            for key, saved in banks.items():
                                self.assertIs(cache._source_reads[key][0], saved[0])
                                for left, right in zip(fixture.packed(*key), saved):
                                    assert_bytes(self, left, right)
                            arms.append((pool_bytes, snapshots, workspaces, list(cache.writes)))
                self.assertEqual(arms[0][3], arms[1][3])
                for baseline, candidate in zip(arms[0][0], arms[1][0]):
                    assert_bytes(self, candidate, baseline)
                for baseline, candidate in zip(arms[0][1], arms[1][1]):
                    assert_bytes(self, candidate, baseline)
                for old, new in zip(arms[0][2], arms[1][2]):
                    for baseline, candidate in zip(old, new):
                        assert_bytes(self, candidate, baseline)


class LocalPrefillSwaPackCPU(_PackCases, unittest.TestCase):
    device = "cpu"

    def test_raw_pointer_unsupported_layout_and_viewbits_return_before_backend(self):
        normal = torch.ones(2, 512, dtype=torch.bfloat16)
        complex_view = torch.ones(2, 512, dtype=torch.complex64).conj()
        for value in (None, normal.to_sparse(), torch._neg_view(normal), complex_view):
            with self.subTest(value_type=type(value)), patch.object(
                _quant, "_ppu_fast_path", side_effect=AssertionError("unsupported backend admission"),
            ):
                self.assertIsNone(_quant.try_pack_swa_rows(value))

    def test_descriptor_non_swa_flag_and_original_fallback_errors(self):
        normal = torch.ones(2, 512, dtype=torch.bfloat16)
        descriptor = default_descriptor("swa_kv")
        with patch.dict(os.environ, {ENV: "1"}), patch.object(
            _quant, "try_pack_swa_rows", side_effect=AssertionError("explicit descriptor must bypass pack"),
        ):
            got = self.ops.encode_kv(normal, pool_tag="swa_kv", descriptor=descriptor)
        assert_bytes(self, got, _encode(self.ops, normal, False))
        for tag, value in (("global_kv_1", normal), ("global_kv_2", normal),
                           ("indexer_kv", normal[:, :128]), ("decoder_swa_kv", normal)):
            with self.subTest(tag=tag), patch.dict(os.environ, {ENV: "1"}), patch.object(
                _quant, "try_pack_swa_rows", side_effect=AssertionError("wrong codec pack"),
            ):
                self.ops.encode_kv(value, pool_tag=tag)
        for value in (normal.float(), normal[:, ::2].repeat_interleave(2, 1),
                      torch.ones(2, 1024, dtype=torch.bfloat16)[:, ::2], normal.reshape(-1),
                      torch.ones(2, 511, dtype=torch.bfloat16), normal.to_sparse()):
            outcomes = []
            for flag in (False, True):
                try:
                    outcomes.append(_encode(self.ops, value, flag))
                except Exception as error:
                    outcomes.append((type(error), str(error)))
            if isinstance(outcomes[0], torch.Tensor):
                assert_bytes(self, outcomes[0], outcomes[1])
            else:
                self.assertEqual(outcomes[0], outcomes[1])
        with patch.dict(os.environ, {ENV: "0"}), patch.object(
            _quant, "try_pack_swa_rows", side_effect=AssertionError("disabled pack"),
        ):
            self.ops.encode_kv(normal, pool_tag="swa_kv")


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillSwaPackDevice(_PackCases, unittest.TestCase):
    device = "cuda"

    def setUp(self):
        super().setUp()
        self.assertTrue(_quant.is_ppu(), "requires an actual ZW PPU")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))

    @torch.no_grad()
    def test_actual_profile_one_pack_replaces_the_old_23_kernel_codec(self):
        for rows in (4096, 140):
            kv = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
            _encode(self.ops, kv, True)  # JIT outside the measured scope.
            _encode(self.ops, kv, False)
            results = []
            for flag in (False, True):
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                    results.append(_encode(self.ops, kv, flag))
                    torch.cuda.synchronize()
                root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
                with tempfile.TemporaryDirectory(prefix="swa-pack-") as temporary:
                    path = Path(root or temporary) / f"local-prefill-swa-pack-N{rows}-flag{int(flag)}.trace.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    profile.export_chrome_trace(str(path))
                    events = json.loads(path.read_text())["traceEvents"]
                kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
                self.assertEqual(sum("pack_v41_swa_row" in name for name in kernels), int(flag))
                self.assertEqual(len(kernels), 1 if flag else 23, kernels)
                if flag:
                    self.assertFalse(any(event.get("name") in ("aten::amax", "aten::cat", "aten::exp2") for event in events))
            assert_bytes(self, results[1], results[0])


if __name__ == "__main__":
    unittest.main()
