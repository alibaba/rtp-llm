"""Creation-proved ordinary producer packing, with unchanged native writes.

CPU tests exercise real provenance and producer/control/cache contracts, with
a reference pack spy. Explicit PPU tests use the unchanged rotary FP4 kernel.
No process group or checkpoint/model weights are required.
"""

import json
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant, attention_layer, prefill
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUResidentBF16Linear
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    LOCAL_PREFILL_SOURCE_CACHE_ENV, LOCAL_PREFILL_SWA_SEGMENTS_ENV,
)
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_rows import state_for
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_source_cache import _SourceFixture
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_pack import _boundary_rows, _finite_patterns
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_segments import FLAGS, arguments, compact_plan, stage
from rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_prefill_out import make_linear


ENV = prefill.LOCAL_PREFILL_ROTARY_PACK_ENV
PACK_ENV = "RTP_V41_PPU_FUSED_KV_ROW_PACK"


def _observation_bytes(value):
    # A one-element stride2 view is logically contiguous: contiguous() may
    # return it unchanged, whereas dtype view(Byte) requires last stride1.
    # Copy only this test observation; do not normalize the primitive input.
    dense = torch.empty(tuple(value.shape), dtype=value.dtype, device=value.device)
    dense.copy_(value)
    return dense.view(torch.uint8)


def _frequency(device, rows=13, half=64, *, identity=False):
    angle = torch.zeros(rows, half, device=device) if identity else (
        torch.arange(rows, device=device)[:, None] * 0.17
        + torch.arange(half, device=device)[None] * 0.03 - 1.5
    )
    return torch.polar(torch.ones_like(angle), angle)


def _fixture(device, ratio=2, prefixes=(129,), lengths=(9,), *, op=None, carry=True):
    source = _SourceFixture(device, ratio, prefixes, lengths)
    source.geometry.norm_eps = 1e-6
    state = state_for(device, prefixes, lengths)
    state.cache = source.cache
    state.runtime.policy = SharedPrefillPolicy(mode="compact")
    state.runtime.model._is_speculative = False
    args = arguments(state)
    proof = prefill._make_local_producer_rotary_proof(
        args[-1], cache=source.cache, layer_id=0, prefixes=prefixes, lengths=lengths,
    )
    # The preceding real SWA write changes the payload version. Its original
    # table/row birth proof must survive without recertifying a pool key.
    stage(state, args)
    positions, requests = args[:2]
    values = source.values(positions, 3).float()
    scores = source.values(positions, 7).float() * 0.25 if ratio == 2 else None
    if ratio == 2:
        for rid, prefix in enumerate(prefixes):
            if prefix % 2:
                old = torch.tensor([prefix - 1], device=device)
                previous = torch.cat((source.values(old, 3).float(), source.values(old, 7).float() * 0.25), -1)
                source.cache._copy_rows((0, "csa_state"), rid, old,
                                        previous if carry else torch.zeros_like(previous))
    if op is None:
        weight = (torch.arange(128 * 512, device=device).reshape(128, 512) % 41 - 20).float() / 200
        op = PPUResidentBF16Linear(weight.bfloat16())
    layer = NS(index_wk=op, compressor_norm=torch.ones(512, device=device, dtype=torch.bfloat16),
               index_k_norm=torch.ones(128, device=device, dtype=torch.bfloat16))
    return NS(source=source, state=state, args=args, proof=proof, layer=layer,
              values=values, scores=scores, freqs=_frequency(device),
              positions=positions, requests=requests)


def _produce(fixture, ops, *, proof=True, append_starts=None):
    return prefill._produce_global(
        ops=ops, layer=fixture.layer, geometry=fixture.source.geometry, freqs=fixture.freqs,
        values=fixture.values, scores=fixture.scores, positions=fixture.positions,
        request_ids=fixture.requests, cache=fixture.source.cache, layer_id=0,
        append_starts=append_starts,
        prefill_layout=(fixture.source.prefixes, fixture.source.fresh),
        **({"local_rotary_proof": fixture.proof} if proof else {}),
    )


class _ProducerCases:
    def setUp(self):
        environment = patch.dict(os.environ, {**FLAGS, ENV: "1", PACK_ENV: "1",
                                               "TP_SIZE": "1", "PREFILL_CP_SIZE": "1",
                                               LOCAL_PREFILL_SOURCE_CACHE_ENV: "1"})
        environment.start()
        self.addCleanup(environment.stop)
        self.ops = PPUOps(num_routed_experts=8, num_local_experts=8)

    @torch.no_grad()
    def test_ratio1_ratio2_odd_carry_append_ragged_zero_native_bytes_and_receipts(self):
        for ratio, prefixes, lengths in (
            (1, (0,), (1,)), (1, (129,), (4096,)), (1, (17,), (4097,)),
            (2, (128,), (4096,)), (2, (129,), (9,)), (2, (0,), (1,)),
            (2, (129, 257, 31), (9, 0, 7)), (1, (17, 99, 64), (140, 5, 0)),
            (1, (17, 9), (0, 0)),
        ):
            arms = []
            for enabled in (False, True):
                f = _fixture(self.device, ratio, prefixes, lengths)
                self.assertIsNotNone(f.proof)
                self.assertTrue(prefill._valid_local_producer_rotary_proof(
                    f.proof, cache=f.source.cache, layer_id=0,
                    prefill_layout=(prefixes, lengths), positions=f.positions, request_ids=f.requests,
                ))
                with patch.dict(os.environ, {ENV: str(int(enabled))}):
                    _produce(f, self.ops, append_starts={rid: p + min(n, 3)
                                                       for rid, (p, n) in enumerate(zip(prefixes, lengths))})
                arms.append((tuple(pool.tensor.clone() for pool in f.source.cache.pools.values()),
                             list(f.source.cache.writes), f))
            self.assertEqual(arms[0][1], arms[1][1])
            for old, got in zip(arms[0][0], arms[1][0]):
                assert_bytes(self, got, old)
            for handle in arms[1][2].source.native.handles.values():
                assert_bytes(self, handle.kv_cache_base[:, -16:], torch.full_like(handle.kv_cache_base[:, -16:], 173))
            for rid, count in enumerate(lengths):
                expected = arms[0][2].source.packed(0, rid)
                got = arms[1][2].source.packed(0, rid)
                for left, right in zip(got, expected):
                    assert_bytes(self, left, right)

    @torch.no_grad()
    def test_missing_odd_carry_and_old_error_order_remain_before_pack(self):
        errors = []
        for enabled in (False, True):
            f = _fixture(self.device, 2, (129,), (9,), carry=False)
            with patch.dict(os.environ, {ENV: str(int(enabled))}), patch.object(
                self.ops, "try_encode_rotary_kv", side_effect=AssertionError("carry error precedes pack"),
            ):
                with self.assertRaises(RuntimeError) as error:
                    _produce(f, self.ops)
                errors.append(str(error.exception))
        self.assertEqual(errors[0], errors[1])
        self.assertIn("starts at odd position 129", errors[1])

    @torch.no_grad()
    def test_full_writes_still_invalidate_all_source_banks(self):
        f = _fixture(self.device, 2, (129, 257), (9, 7))
        for source in (0, 20):
            for rid in (0, 1):
                f.source.packed(source, rid)
        self.assertEqual(len(f.source.cache._local_source_reads), 4)
        _produce(f, self.ops)
        self.assertFalse(f.source.cache._local_source_reads or f.source.cache._source_reads)


class LocalPrefillRotaryPackCPU(_ProducerCases, unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        super().setUp()
        # Only CPU protocol tests substitute the primitive. The real codec,
        # norms, linear modules and native insert/read paths remain in use.
        primitive = patch.object(_quant, "try_pack_rotary_fp4_rows", new=lambda kv, freqs, positions, global_pool:
                                 encode_kv(prefill._rope_tail(kv, freqs[positions.clamp_max(freqs.shape[0] - 1)]),
                                           pool_tag="global_kv_1" if global_pool else "indexer_kv"))
        primitive.start()
        self.addCleanup(primitive.stop)

    @torch.no_grad()
    def test_creation_owners_versions_payload_independence_and_host_ranges(self):
        f = _fixture("cpu")
        valid = lambda proof=f.proof, **changes: prefill._valid_local_producer_rotary_proof(
            proof, cache=changes.get("cache", f.source.cache), layer_id=changes.get("layer_id", 0),
            prefill_layout=changes.get("layout", (f.source.prefixes, f.source.fresh)),
            positions=changes.get("positions", f.positions), request_ids=changes.get("requests", f.requests),
        )
        self.assertTrue(valid())
        self.assertFalse(valid(positions=f.positions.view_as(f.positions)))
        self.assertFalse(valid(positions=f.positions.clone()))
        self.assertFalse(valid(layer_id=20))
        self.assertFalse(valid(layout=((130,), (8,))))
        self.assertFalse(valid(cache=_fixture("cpu").source.cache))
        f.source.cache.pools[0, "swa_kv"].tensor.zero_()
        self.assertTrue(valid(), "own SWA payload is not metadata provenance")
        f.positions.add_(1).sub_(1)
        self.assertFalse(valid())
        self.assertIsNone(prefill._make_local_producer_rotary_proof(
            f.args[-1], cache=f.source.cache, layer_id=0, prefixes=(129,), lengths=(9,),
        ), "changed-then-restored rows cannot be recertified")
        fresh = _fixture("cpu")
        fresh.state.attn.cu_seqlens.add_(1).sub_(1)
        self.assertFalse(prefill._valid_local_producer_rotary_proof(
            fresh.proof, cache=fresh.source.cache, layer_id=0, prefill_layout=((129,), (9,)),
        ))
        for mutate in (
            lambda f: f.requests.add_(1).sub_(1),
            lambda f: setattr(f.state.attn, "cu_seqlens", f.state.attn.cu_seqlens.clone()),
            lambda f: setattr(f.state, "lengths", (8,)),
        ):
            f = _fixture("cpu")
            mutate(f)
            self.assertFalse(prefill._valid_local_producer_rotary_proof(
                f.proof, cache=f.source.cache, layer_id=0, prefill_layout=((129,), (9,)),
            ))
        f = _fixture("cpu")
        self.assertIsNone(prefill._make_local_producer_rotary_proof(
            f.args[-1]._replace(ranges=((0, 130, 8),)), cache=f.source.cache,
            layer_id=0, prefixes=(129,), lengths=(9,),
        ))

    @torch.no_grad()
    def test_phase_missing_birth_selected_only_and_selector_changes_fall_back(self):
        for target in ("idle", "prefill", "is_graph", "verify", "proposal", "spec", "diagnostic_graph"):
            f = _fixture("cpu")
            if target == "spec":
                f.state.runtime.model._is_speculative = True
            elif target == "diagnostic_graph":
                f.state.attn.is_cuda_graph = True
            else:
                setattr(f.state, target, target != "prefill")
            with patch.object(self.ops, "try_encode_rotary_kv", side_effect=AssertionError("phase fallback")):
                _produce(f, self.ops)
        f = _fixture("cpu", prefixes=(129,), lengths=(4096,))
        compact_plan(f.state)
        selected = arguments(f.state, compact=True)[-1]
        self.assertIsNone(prefill._make_local_producer_rotary_proof(
            selected, cache=f.source.cache, layer_id=0, prefixes=(129,), lengths=(4096,),
        ))
        with patch.dict(os.environ, {LOCAL_PREFILL_SWA_SEGMENTS_ENV: "0"}):
            unborn = _fixture("cpu")
        self.assertIsNone(unborn.proof)
        self.assertIsNone(prefill._make_local_producer_rotary_proof(
            unborn.args[-1], cache=unborn.source.cache, layer_id=0, prefixes=(129,), lengths=(9,),
        ))
        with patch.dict(os.environ, {ENV: "0"}):
            later = _fixture("cpu")
        self.assertIsNone(later.proof)
        self.assertIsNotNone(prefill._make_local_producer_rotary_proof(
            later.args[-1], cache=later.source.cache, layer_id=0, prefixes=(129,), lengths=(9,),
        ), "a local selector needs no new latch when the original A08 birth proof exists")
        f = _fixture("cpu")
        self.assertIsNone(attention_layer._local_swa_stage_ranges(
            f.state, NS(cp_size=8, kv_cache_sharded=True), 0,
            f.positions, f.requests, slice(None), f.positions, f.requests, f.args[2],
            packed_plan=None, **f.args[3],
        ))
        with patch.dict(os.environ, {ENV: "0"}), patch.object(
            self.ops, "try_encode_rotary_kv", side_effect=AssertionError("disabled selector"),
        ):
            _produce(f, self.ops)
        self.assertTrue(prefill._valid_local_producer_rotary_proof(
            f.proof, cache=f.source.cache, layer_id=0, prefill_layout=((129,), (9,)),
        ))

    @torch.no_grad()
    def test_stale_negative_positions_preserve_old_negative_index_and_error(self):
        for out_of_range in (False, True):
            arms, errors = [], []
            for proof in (False, True):
                f = _fixture("cpu", 1, (0,), (5,))
                f.positions.copy_(-torch.arange(5))
                if out_of_range:
                    f.positions[0] = -f.freqs.shape[0] - 1
                with patch.object(self.ops, "try_encode_rotary_kv", side_effect=AssertionError("stale positions")):
                    if out_of_range:
                        with self.assertRaises(IndexError) as error:
                            _produce(f, self.ops, proof=proof)
                        errors.append(str(error.exception))
                    else:
                        _produce(f, self.ops, proof=proof)
                        arms.append(tuple(pool.tensor.clone() for pool in f.source.cache.pools.values()))
            self.assertEqual(errors[:1], errors[1:])
            if arms:
                for left, right in zip(*arms):
                    assert_bytes(self, left, right)
        for proof in (False, True):
            f = _fixture("cpu", 1, (0,), (5,))
            f.values = f.values[:-1]
            with patch.object(self.ops, "try_encode_rotary_kv", side_effect=AssertionError("count error precedes pack")):
                with self.assertRaisesRegex(RuntimeError, "produced 4 entries.*expected 5"):
                    _produce(f, self.ops, proof=proof)

    @torch.no_grad()
    def test_no_proof_and_independent_primitive_fallbacks_keep_old_chain(self):
        for proof in (False, True):
            for missing in (True, False):
                f = _fixture("cpu", 1, (0,), (5,))
                if not proof:
                    with patch.object(self.ops, "try_encode_rotary_kv", side_effect=AssertionError("no proof")):
                        _produce(f, self.ops, proof=False)
                    continue
                baseline = _fixture("cpu", 1, (0,), (5,))
                _produce(baseline, self.ops, proof=False)
                original = self.ops.try_encode_rotary_kv
                with patch.object(self.ops, "try_encode_rotary_kv", side_effect=lambda *a, **kw:
                                  None if kw["global_pool"] == missing else original(*a, **kw)):
                    _produce(f, self.ops)
                for key in f.source.cache.pools:
                    assert_bytes(self, f.source.cache.pools[key].tensor, baseline.source.cache.pools[key].tensor)
        f = _fixture("cpu")
        with patch.dict(os.environ, {PACK_ENV: "0"}), patch.object(
            _quant, "try_pack_rotary_fp4_rows", side_effect=AssertionError("primitive flag bypass"),
        ):
            _produce(f, self.ops)

    @torch.no_grad()
    def test_actual_attention_source_call_binds_same_stage_owners_only_with_valid_proof(self):
        class Captured(RuntimeError):
            pass

        for enabled, invalidate in ((False, False), (True, False), (True, True)):
            f = _fixture("cpu", 1, (0,), (5,))
            attn = attention_layer.SharedPPUAttention.__new__(attention_layer.SharedPPUAttention)
            torch.nn.Module.__init__(attn)
            g = f.source.geometry
            g.hidden_size, g.n_heads = 8, 1
            attn.geometry, attn.layer_id, attn.compress_ratio = g, 0, 1
            attn.is_kv_source, attn.is_index_source, attn._cp_ctx = True, False, None
            attn.ops = NS(encode_kv=encode_kv)
            q = f.source.values(f.positions, 4).unsqueeze(1)
            kv = f.source.values(f.positions, 5)
            attn.project_query_kv = lambda x, **kw: (x, q.clone(), kv.clone(), None)
            attn.global_wkv, attn.global_wgate = lambda x: f.values, None
            attn.runtime = NS(current=f.state, policy=f.state.runtime.policy,
                              freqs_for=lambda layer: f.freqs, qkv_streams=None)
            seen, produced = {}, {}
            original_stage = f.source.cache.stage_swa

            def stage_probe(layer, positions, requests, *args, **kwargs):
                seen.update(positions=positions, request_ids=requests)
                original_stage(layer, positions, requests, *args, **kwargs)
                if invalidate:
                    positions.add_(1).sub_(1)

            def producer_probe(**kwargs):
                produced.update(kwargs)
                raise Captured()

            with patch.dict(os.environ, {ENV: str(int(enabled)), attention_layer.FUSION_ENV: "0",
                                          attention_layer.STREAMS_ENV: "0", "RTP_V41_PPU_FUSED_BF16_ROPE": "0"}), \
                 patch.object(attention_layer, "prepare_packed_attention", return_value=(None, "disabled")), \
                 patch.object(attention_layer, "optimized_diagnostics_enabled", return_value=False), \
                 patch.object(attention_layer, "_local_prefill_grouped_qo_rope_enabled", return_value=False), \
                 patch.object(f.source.cache, "stage_swa", new=stage_probe), \
                 patch.object(attention_layer, "_produce_global", new=producer_probe):
                with self.assertRaises(Captured):
                    attn._body(torch.zeros(5, 8, dtype=torch.bfloat16))
            if enabled and not invalidate:
                self.assertIs(produced["positions"], seen["positions"])
                self.assertIs(produced["request_ids"], seen["request_ids"])
                self.assertIsInstance(produced["local_rotary_proof"], prefill._LocalProducerRotaryProof)
            else:
                self.assertNotIn("local_rotary_proof", produced)
                self.assertIsNot(produced["positions"], seen["positions"])


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillRotaryPackDevice(_ProducerCases, unittest.TestCase):
    def setUp(self):
        self.assertTrue(_quant.is_ppu())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        self.device = "cuda:0"
        self._rotary_diagnostic_count = 0
        super().setUp()

    def _diagnose_primitive(self, kv, freqs, positions, packed, expected, rotated, *, case, rope_mode, global_pool):
        if self._rotary_diagnostic_count >= 8:
            return
        self._rotary_diagnostic_count += 1
        got, want = packed.detach().cpu(), expected.detach().cpu()
        mismatch = (got != want).nonzero()[:8].tolist()
        dim, group = (512, 16) if global_pool else (128, 32)
        details = []
        for row, byte in mismatch:
            group_id = 2 * byte // group if byte < dim // 2 else byte - dim // 2
            begin, end = group_id * group, (group_id + 1) * group
            position = int(positions[row].cpu())
            frequency = min(position, freqs.shape[0] - 1)
            source_bits = kv[row, begin:end].detach().cpu().view(torch.int16).tolist()
            rotated_bits = rotated[row, begin:end].detach().cpu().view(torch.int16).tolist()
            details.append(dict(
                row=row, byte=byte, kind="payload" if byte < dim // 2 else "scale", group=group_id,
                actual=int(got[row, byte]), expected=int(want[row, byte]), position=position,
                clamped_frequency_row=frequency,
                input_bf16_bits=[f"{value & 0xffff:04x}" for value in source_bits],
                rope_bf16_bits=[f"{value & 0xffff:04x}" for value in rotated_bits],
                frequency_fp32_bits=[f"{value & 0xffffffff:08x}" for value in
                                     torch.view_as_real(freqs[frequency]).detach().cpu().view(torch.int32).flatten().tolist()],
            ))
        print("ROTARY_PACK_BYTE_MISMATCH " + json.dumps(dict(
            case=case, global_pool=global_pool, rope_mode=rope_mode,
            kv_shape=list(kv.shape), kv_stride=list(kv.stride()), freqs_shape=list(freqs.shape),
            position_shape=list(positions.shape), position_stride=list(positions.stride()),
            first_mismatches=details,
        )), flush=True)

    def assert_primitive(self, kv, freqs, positions, *, global_pool, case="primitive"):
        original = tuple(t.clone() for t in (kv, freqs, positions))
        packed = self.ops.try_encode_rotary_kv(kv, freqs, positions, global_pool=global_pool)
        self.assertIsNotNone(packed)
        tag = "global_kv_1" if global_pool else "indexer_kv"
        for rope_mode in ("0", "1"):
            with self.subTest(case=case, global_pool=global_pool, half=freqs.shape[1], rope_mode=rope_mode):
                with patch.dict(os.environ, {"RTP_V41_PPU_FUSED_BF16_ROPE": rope_mode}):
                    rotated = prefill._rope_tail(kv, freqs[positions.clamp_max(freqs.shape[0] - 1)])
                    expected = self.ops.encode_kv(rotated, pool_tag=tag)
                if not torch.equal(packed, expected):
                    self._diagnose_primitive(kv, freqs, positions, packed, expected, rotated,
                                             case=case, rope_mode=rope_mode, global_pool=global_pool)
                assert_bytes(self, packed, expected)
        self.assertEqual(packed.shape, (kv.shape[0], 288 if global_pool else 68))
        self.assertEqual(packed.dtype, torch.uint8)
        for value, old in zip((kv, freqs, positions), original):
            self.assertEqual(value.dtype, old.dtype)
            self.assertEqual(value.shape, old.shape)
            assert_bytes(self, _observation_bytes(value), _observation_bytes(old))
        if packed.numel():
            self.assertNotEqual(packed.untyped_storage().data_ptr(), kv.untyped_storage().data_ptr())

    @torch.no_grad()
    def test_primitive_sizes_complex_rounding_clamp_strides_offset_and_canaries(self):
        for global_pool in (True, False):
            dim = 512 if global_pool else 128
            for rows in (0, 1, 140, 267, 4096, 4097):
                owner = torch.full((rows * dim + 16,), -9, dtype=torch.bfloat16, device=self.device)
                kv = owner[8:-8].view(rows, dim)
                kv.copy_(((torch.arange(rows * dim, device=self.device) % 127) - 63).view(rows, dim) / 64)
                pos_owner = torch.arange(2 * rows, device=self.device, dtype=torch.int64)
                positions = pos_owner[::2]
                if rows:
                    positions[-1] = 2**63 - 1
                for half in (32, 64):
                    for freq in (_frequency(self.device, half=half), _frequency(self.device, rows=1, half=half)):
                        self.assert_primitive(kv, freq, positions, global_pool=global_pool,
                                              case=f"sizes-N{rows}-F{freq.shape[0]}-half{half}-stride2")
                self.assertTrue(bool((owner[:8] == -9).all() and (owner[-8:] == -9).all()))

    @torch.no_grad()
    def test_primitive_all_finite_bf16_nan_signedzero_floor_ties_and_reciprocal_bytes(self):
        anchors = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
                                6 * 2**-126, 6 * 2**-9, 6 * 448], dtype=torch.bfloat16)
        bits = anchors.view(torch.int16).int()
        adjacent = torch.stack((bits - 1, bits, bits + 1), -1).flatten().to(torch.int16).view(torch.bfloat16)
        groups = torch.ones(adjacent.numel() * 2, 32, dtype=torch.bfloat16) * 6
        groups[:, 0] = torch.cat((adjacent, -adjacent))
        for global_pool in (True, False):
            dim = 512 if global_pool else 128
            for label, source in (("finite-bf16", _finite_patterns(self.device)),
                                  ("boundary-nan-inf", _boundary_rows(self.device)),
                                  ("ties-floor", groups.flatten().repeat(16).reshape(-1, dim).to(self.device))):
                kv = source.reshape(-1, dim)
                for identity in (True, False):
                    for half in (32, 64):
                        freq = _frequency(self.device, half=half, identity=identity)
                        positions = torch.arange(kv.shape[0], device=self.device)
                        self.assert_primitive(kv, freq, positions, global_pool=global_pool,
                                              case=f"{label}-identity{int(identity)}-half{half}")
            real = torch.tensor([1.0, -1.0, 0.0, -0.0, float("nan"), -float("nan"),
                                 float("inf"), -float("inf")], device=self.device)
            kv = torch.ones(8, dim, device=self.device, dtype=torch.bfloat16)
            kv[:, 0::4], kv[:, 1::4] = 0.0, -0.0
            for half in (32, 64):
                index = (torch.arange(8, device=self.device)[:, None]
                         + torch.arange(half, device=self.device)[None]) % 8
                freq = torch.complex(real[index], real[(index + 1) % 8])
                self.assert_primitive(kv, freq, torch.arange(8, device=self.device), global_pool=global_pool,
                                      case=f"nonfinite-complex-frequencies-half{half}")

    @torch.no_grad()
    def test_raw_unsupported_views_decline_before_kernel(self):
        kv = torch.ones(3, 512, device=self.device, dtype=torch.bfloat16)
        freqs = _frequency(self.device)
        positions = torch.arange(3, device=self.device)
        for values in ((torch._neg_view(kv), freqs, positions), (kv, freqs.conj(), positions),
                       (kv, torch._neg_view(freqs), positions), (kv, freqs, torch._neg_view(positions)),
                       (kv, freqs, positions[:1].expand(3)), (kv[:, ::2], freqs, positions),
                       (kv.float(), freqs, positions), (kv.to_sparse(), freqs, positions)):
            with patch.object(_quant, "_ppu_fast_path", side_effect=AssertionError("raw pointer guard")):
                self.assertIsNone(_quant.try_pack_rotary_fp4_rows(*values, global_pool=True))

    @torch.no_grad()
    def test_actual_ptpc_producer_profile_uses_eight_rotary_packs_without_new_position_reduction(self):
        torch.manual_seed(4110)
        op = make_linear(device=self.device)
        arms = []
        for enabled in (False, True):
            fixtures = [_fixture(self.device, ratio, (128,), (4096,), op=op) for ratio in (2, 2, 2, 1)]
            shapes, original_pack = [], self.ops.try_encode_rotary_kv

            def pack_probe(kv, freqs, positions, *, global_pool):
                shapes.append((tuple(kv.shape), tuple(positions.shape), global_pool))
                return original_pack(kv, freqs, positions, global_pool=global_pool)

            with patch.dict(os.environ, {ENV: str(int(enabled))}), \
                 patch.object(self.ops, "try_encode_rotary_kv", new=pack_probe), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            ) as profile:
                for f in fixtures:
                    _produce(f, self.ops)
                torch.cuda.synchronize()
            events = _trace_events(profile, f"local-rotary-pack-{int(enabled)}")
            kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
            self.assertEqual(sum("pack_v41_rotary_fp4_row" in name for name in kernels), 8 * int(enabled))
            expected_shapes = [((count, dim), (count,), global_pool)
                               for count in (2048, 2048, 2048, 4096)
                               for dim, global_pool in ((512, True), (128, False))]
            self.assertEqual(shapes, expected_shapes if enabled else [])
            # The primitive and proof must not add a value-range download.
            reductions = [e for e in events if e.get("cat") == "cpu_op"
                          and e.get("name") in ("aten::min", "aten::max")
                          and e.get("args", {}).get("Input Dims") == [[4096]]]
            arms.append((fixtures, len(reductions)))
        self.assertLessEqual(arms[1][1], arms[0][1])
        for old, got in zip(arms[0][0], arms[1][0]):
            self.assertEqual(old.source.cache.writes, got.source.cache.writes)
            for key in old.source.cache.pools:
                assert_bytes(self, old.source.cache.pools[key].tensor, got.source.cache.pools[key].tensor)


if __name__ == "__main__":
    unittest.main()
