"""Original-CP SWA packet routing and same-native row-permutation equality.

CPU cases retain the actual full-row birth and attention body while doubling
native arithmetic/transport explicitly. PPU cases compare the same native
kernel before and after restore. WORLD8 is separately enabled and exercises
real transport/scratch reuse plus ordinary native-cache SWA snapshots; it is
not a model benchmark or qualification of other collective protocols.
"""

import os
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41 import cp as cp_module
from rtp_llm.models_py.modules.dsv41.ppu import attention_layer as attention
from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_local_swa_pack as local
from rtp_llm.models_py.modules.dsv41.ppu import norm_store
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import (
    DENSE_PROJECTIONS,
    MEGA_ATTENTION,
    NORM_STORE,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_cross_layer_mhc import (
    CP_ENV,
    _state,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_l20_query_projection import (
    _context,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_source_cache import (
    _SourceFixture,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_split_l20 import (
    _component,
)

FLAGS = dict(
    CP_ENV,
    **{
        NORM_STORE: "1",
        DENSE_PROJECTIONS: "0",
        MEGA_ATTENTION: "0",
        local.PACK_LOCAL_SWA: "1",
    },
)


def _birth(rank=0, lengths=(256, 272), prefixes=(17, 49), device="cpu"):
    cp, ids = _context(rank, tuple(lengths), tuple(prefixes), device)
    state = _state(cp, ids)
    if state._cp_full_rows_birth is None:
        raise AssertionError("test requires a construction-bound original CP birth")
    return state, cp


def _frequencies(state, *, identity=False):
    device = state.inputs.input_ids.device
    rows = max(state.cache.lengths)
    if identity:
        return torch.ones(rows, 32, dtype=torch.complex64, device=device)
    angles = (
        torch.arange(rows, device=device, dtype=torch.float32)[:, None]
        * (torch.arange(32, device=device, dtype=torch.float32)[None] + 1)
        / 1024
    )
    return torch.complex(angles.cos(), angles.sin()).contiguous()


def _raw_rows(cp, *, strided=False, special=False):
    width = 1792 if strided else 512
    owner = torch.full(
        (cp.chunk_length, width),
        0.25,
        dtype=torch.bfloat16,
        device=cp.global_positions.device,
    )
    raw = owner[:, -512:]
    values = (
        (
            torch.arange(raw.numel(), device=raw.device).reshape_as(raw) * 13
            + cp.cp_rank * 31
        )
        % 127
        - 63
    ) / 64
    raw.copy_(values)
    if special and raw.shape[0]:
        patterns = torch.tensor(
            [0, 0x8000, 1, 0x8001, 0x7F80, 0xFF80, 0x7FC1, 0xFFC1],
            dtype=torch.int32,
            device=raw.device,
        ).to(torch.int16)
        raw[0, :8].view(torch.int16).copy_(patterns)
        raw[-1].zero_()
    return owner, raw


def _bytes_equal(actual, expected):
    return (
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
    )


class CP8LocalSWAPackContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ, FLAGS))
        self.stack.enter_context(
            patch(
                "rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", return_value=True
            )
        )
        self.stack.enter_context(
            patch("torch.cuda.is_current_stream_capturing", return_value=False)
        )
        self.stack.enter_context(
            patch.object(attention, "optimized_diagnostics_enabled", return_value=False)
        )
        self.stack.enter_context(
            patch.object(local, "_device_supported", return_value=True)
        )

    def admitted(self, state, cp, **kwargs):
        return local.host_admitted(
            state, cp, layer_id=20, deferred_kv_norm=True, yoco_small_ops=True, **kwargs
        )

    def test_original_birth_is_required_without_minting_or_derived_request_owners(self):
        for rank in range(8):
            state, cp = _birth(rank, (257, 145, 33), (17, 256, 8192))
            birth = state._cp_full_rows_birth
            self.assertTrue(self.admitted(state, cp))
            self.assertIs(state._cp_full_rows_birth, birth)
            # state.rows is allowed to cast request IDs for consumers. That
            # derived tensor must not replace the native birth owner.
            rows = state.rows(cp)
            self.assertEqual(rows[1].dtype, torch.int64)
            self.assertEqual(cp.req_id_per_token.dtype, torch.int32)
            self.assertTrue(self.admitted(state, cp))
        state, cp = _birth()
        cp.global_positions = cp.global_positions.clone()
        with self.assertRaisesRegex(RuntimeError, "original full-row birth"):
            self.admitted(state, cp)
        self.assertTrue(state._cp_full_rows_invalid)

    def test_other_phases_flags_and_compact_contexts_keep_the_existing_protocol(self):
        for field, value in (
            ("prefill", False),
            ("idle", True),
            ("compacted", True),
            ("verify", True),
            ("is_graph", True),
        ):
            state, cp = _birth()
            setattr(state, field, value)
            self.assertFalse(self.admitted(state, cp), field)
        state, cp = _birth()
        for flag in (NORM_STORE, local.PACK_LOCAL_SWA):
            with patch.dict(os.environ, {flag: "0"}):
                self.assertFalse(self.admitted(state, cp))
        self.assertFalse(self.admitted(state, cp, diagnostic={}))
        self.assertFalse(self.admitted(state, cp, packed_plan=object()))
        for extra in (
            {"deferred_kv_norm": False},
            {"yoco_small_ops": False},
            {"layer_id": 21},
        ):
            args = dict(layer_id=20, deferred_kv_norm=True, yoco_small_ops=True)
            args.update(extra)
            self.assertFalse(local.host_admitted(state, cp, **args))
        state._cp_full_rows_birth = None
        with self.assertRaisesRegex(RuntimeError, "original full-row birth"):
            self.admitted(state, cp)

    def test_pack_is_private_uses_original_positions_and_preserves_joined_raw_qr(self):
        state, cp = _birth(lengths=(256, 272), prefixes=(17, 49))
        owner, raw = _raw_rows(cp, strided=True)
        before = owner.clone()
        weight, frequencies = torch.ones(512), _frequencies(state)
        seen = []

        def native(value, norm_weight, eps, indexed):
            seen.append((value, norm_weight, eps, indexed.clone()))
            return torch.full((value.shape[0], 528), 37, dtype=torch.uint8)

        with patch.object(norm_store, "try_norm_rope_pack_swa", side_effect=native):
            packet = local.pack(state, cp, raw, weight, 1e-6, frequencies, layer_id=20)
        self.assertIs(seen[0][0], raw)
        self.assertIs(seen[0][1], weight)
        self.assertEqual(seen[0][2], 1e-6)
        self.assertTrue(torch.equal(seen[0][3], frequencies[cp.global_positions]))
        self.assertTrue(_bytes_equal(owner, before))
        self.assertFalse(torch._C._is_alias_of(packet, owner))
        self.assertEqual(
            state.shared["cp_local_swa_pack"],
            [
                dict(
                    layer=20,
                    rows=cp.chunk_length,
                    rawBytes=cp.chunk_length * 1024,
                    packedBytes=cp.chunk_length * 528,
                )
            ],
        )
        raw.zero_()
        self.assertTrue((packet == 37).all())

    def test_admitted_local_decline_unsupported_input_and_epoch_change_never_fall_back(
        self,
    ):
        state, cp = _birth()
        owner, raw = _raw_rows(cp, strided=True)
        before = owner.clone()
        weight, frequencies = torch.ones(512), _frequencies(state)
        with patch.object(
            norm_store, "try_norm_rope_pack_swa", return_value=None
        ) as native:
            with self.assertRaisesRegex(
                RuntimeError, "declined after protocol admission"
            ):
                local.pack(state, cp, raw, weight, 1e-6, frequencies, layer_id=20)
            self.assertEqual(native.call_count, 1)
            with self.assertRaisesRegex(RuntimeError, "admitted native input"):
                local.pack(
                    state, cp, raw.float(), weight, 1e-6, frequencies, layer_id=20
                )
            self.assertEqual(native.call_count, 1)
        self.assertNotIn("cp_local_swa_pack", state.shared)
        self.assertTrue(_bytes_equal(owner, before))

        def changed(*args):
            cp.local_is_real.logical_not_()
            return torch.zeros(cp.chunk_length, 528, dtype=torch.uint8)

        with patch.object(norm_store, "try_norm_rope_pack_swa", side_effect=changed):
            with self.assertRaisesRegex(
                RuntimeError, "source owners or full-row epoch"
            ):
                local.pack(state, cp, raw, weight, 1e-6, frequencies, layer_id=20)

    def _body(self, *, producer_callback):
        state, cp = _birth(lengths=(16,) * 8, prefixes=(256,) * 8)
        # The inherited construction fixture creates a140-row CP1 plan.
        # Its unused state needs at least140 rows before we replace it below.
        block, _, _, _, _ = _component("cpu", max(512, cp.chunk_length))
        unit = block.attn
        unit.runtime.current, unit.runtime.policy, unit.runtime.model = (
            state,
            state.runtime.policy,
            state.runtime.model,
        )
        unit._cp_ctx = cp
        unit._get_mega_attention = Mock(return_value=None)
        frequencies = _frequencies(state)
        unit.runtime.freqs_for = lambda layer: frequencies
        unit.kv_norm = NS(weight=torch.ones(512), variance_epsilon=1e-6)
        owner, raw = _raw_rows(cp, strided=True)
        before = owner.clone()
        x = torch.zeros(
            cp.chunk_length, unit.geometry.hidden_size, dtype=torch.bfloat16
        )
        unit.project_query_kv = Mock(
            return_value=(owner[:, :8], owner[:, :8].clone(), raw, None)
        )
        unit.ops.encode_kv = Mock(side_effect=AssertionError("second SWA encode"))
        unit._project_indexer = Mock(
            return_value=(
                torch.zeros(cp.chunk_length, 1, 8),
                torch.ones(cp.chunk_length, 1),
            )
        )
        state.plan = NS(
            exchange=NS(compact=lambda value: value[:2]), context=NS(chunk_length=2)
        )
        events, packets = [], []

        class Complete(RuntimeError):
            pass

        def stage(layer, positions, requests, packed, prefixes, **kwargs):
            events.append(("swa", packed.clone(), positions.clone(), requests.clone()))
            self.assertEqual(packed.shape, (sum(state.lengths), 528))
            self.assertEqual(packed.dtype, torch.uint8)
            if not producer_callback:
                raise Complete("SWA stage reached")

        state.cache.stage_swa = stage

        def gather(value, context, **kwargs):
            packets.append((value.dtype, value.shape, value.clone()))
            gathered = torch.cat([value] * 8)
            return cp_module._cp_restore_gathered_full_2d(gathered, context)

        def native(value, weight, eps, indexed):
            events.append(("pack", value.shape[0]))
            self.assertIs(value, raw)
            return torch.full((cp.chunk_length, 528), 71, dtype=torch.uint8)

        def produce(**kwargs):
            events.append(("global_indexer_csa", kwargs["values"].shape[0]))
            self.assertEqual(kwargs["values"].shape[0], sum(state.lengths))

        def callback():
            events.append(("compact_callback",))
            self.assertTrue(_bytes_equal(owner, before))
            raise Complete("full producers completed before callback")

        with patch.object(
            norm_store, "try_norm_rope_pack_swa", side_effect=native
        ) as native_call, patch.object(
            attention, "cp_all_gather_full", side_effect=gather
        ), patch.object(
            attention, "_rope_tail", side_effect=lambda value, *args, **kwargs: value
        ), patch.object(
            attention, "_produce_global", side_effect=produce
        ), patch.object(
            attention, "source_identity", return_value=object()
        ):
            with self.assertRaisesRegex(
                Complete, "callback" if producer_callback else "SWA stage"
            ):
                unit._body(
                    x, compact_after_producer=callback if producer_callback else None
                )
        self.assertEqual(native_call.call_count, 1)
        unit.ops.encode_kv.assert_not_called()
        self.assertEqual(packets[0][0], torch.uint8)
        self.assertEqual(packets[0][1], (cp.chunk_length, 528))
        self.assertTrue(_bytes_equal(owner, before))
        if producer_callback:
            self.assertEqual(
                [event[0] for event in events],
                ["pack", "swa", "global_indexer_csa", "compact_callback"],
            )
            self.assertEqual(
                [tuple(packet[1]) for packet in packets[1:]],
                [(cp.chunk_length, 8), (cp.chunk_length, 8)],
            )
        return state

    def test_actual_body_gathers_one_native_packet_and_never_packs_twice(self):
        state = self._body(producer_callback=False)
        self.assertEqual(state.shared["current_row_gathers"][0]["width"], 528)

    def test_actual_l20_body_keeps_full_swa_global_indexer_csa_before_callback(self):
        self._body(producer_callback=True)


def _native_commutation(
    lengths,
    prefixes,
    *,
    strided=False,
    special=False,
    weight_dtype=torch.bfloat16,
    eps=1e-6,
):
    contexts, raw_rows, packets, owners = [], [], [], []
    state0 = None
    for rank in range(8):
        state, cp = _birth(rank, lengths, prefixes, "cuda")
        if state0 is None:
            state0 = state
            frequencies = _frequencies(state, identity=special)
            weight = torch.linspace(0.5, 1.5, 512, device="cuda").to(weight_dtype)
        owner, raw = _raw_rows(cp, strided=strided, special=special)
        before = owner.clone()
        packet = local.pack(state, cp, raw, weight, eps, frequencies, layer_id=20)
        if not _bytes_equal(owner, before):
            raise AssertionError("local native pack modified raw KV/joined QR")
        contexts.append(cp)
        raw_rows.append(raw.contiguous())
        packets.append(packet)
        owners.append(owner)
    cp = contexts[0]
    restored_raw = cp_module._cp_restore_gathered_full_2d(torch.cat(raw_rows), cp)
    positions, _, real = state0.full_rows(cp)
    expected = norm_store.try_norm_rope_pack_swa(
        restored_raw[real], weight, eps, frequencies[positions[real]]
    )
    actual = cp_module._cp_restore_gathered_full_2d(torch.cat(packets), cp)[real]
    if expected is None or not _bytes_equal(actual, expected):
        raise AssertionError(
            "same native SWA kernel changed bytes across original CP restore"
        )
    return actual


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1",
    "explicit PPU device test",
)
class CP8LocalSWAPackDevice(unittest.TestCase):
    @torch.no_grad()
    def test_same_native_full_and_local_paths_are_byte_equal_for_full_and_ragged_cp(
        self,
    ):
        self.assertTrue(is_ppu())
        with patch.dict(os.environ, FLAGS):
            _native_commutation((4096,) * 8, (126976,) * 8, strided=True)
            _native_commutation(
                (257, 145, 33, 512, 17, 1, 140, 271),
                (17, 256, 99, 512, 0, 31, 0, 129),
                strided=True,
                special=True,
                weight_dtype=torch.float32,
                eps=0,
            )


@patch.dict(os.environ, dict(FLAGS, TP_SIZE="1", PREFILL_CP_SIZE="1"))
def _world8_worker(rank, store):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + store,
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=300),
    )
    try:
        with torch.no_grad(), patch.object(
            cp_module.collective_torch, "_get_group", return_value=dist.group.WORLD
        ), patch.object(
            cp_module.collective_torch,
            "_get_symm_mem",
            return_value=NS(get_symm_mem_communicator=lambda: None),
        ):
            for lengths, prefixes in (
                ((4096,) * 8, (129,) * 8),
                (
                    (257, 145, 33, 512, 17, 1, 140, 271),
                    (17, 256, 99, 512, 0, 31, 0, 129),
                ),
            ):
                state, cp = _birth(rank, lengths, prefixes, f"cuda:{rank}")
                frequency_table = _frequencies(state)
                weight = torch.ones(
                    512, dtype=torch.bfloat16, device=cp.global_positions.device
                )
                owner, raw = _raw_rows(cp, strided=True, special=True)
                before = owner.clone()
                receive = cp_module.CPGatherReceive(8, rank)
                restored_raw = cp_module.cp_all_gather_full(
                    raw.contiguous(), cp, receive=receive
                )
                positions, requests, real = state.full_rows(cp)
                expected = norm_store.try_norm_rope_pack_swa(
                    restored_raw[real], weight, 1e-6, frequency_table[positions[real]]
                )
                packet = local.pack(
                    state, cp, raw, weight, 1e-6, frequency_table, layer_id=20
                )
                actual = cp_module.cp_all_gather_full(packet, cp, receive=receive)[real]
                if (
                    expected is None
                    or not _bytes_equal(actual, expected)
                    or not _bytes_equal(owner, before)
                ):
                    raise AssertionError(
                        (rank, "WORLD8 native packet/restore byte mismatch")
                    )
                snapshot = actual.clone()
                # Actual scratch is overwritten by the next gather. Restored
                # rows and the staged SWA snapshot must have independent life.
                receive.gather(torch.full_like(packet, 231))
                if not _bytes_equal(actual, snapshot):
                    raise AssertionError((rank, "restored bytes alias receive scratch"))
                # Ordinary native cache receivers test the unchanged stage/
                # retention API independently from sharded-writer qualification.
                # The ordinary-cache environment is fixed before group init.
                with nullcontext():
                    fixture = _SourceFixture(
                        cp.global_positions.device,
                        ratio=1,
                        prefixes=prefixes,
                        fresh=lengths,
                        sources=(),
                    )
                    fixture.cache.stage_swa(
                        0, positions[real], requests[real], actual, prefixes
                    )
                    offset = 0
                    for request, length in enumerate(lengths):
                        query_positions = torch.arange(
                            prefixes[request],
                            prefixes[request] + length,
                            device=actual.device,
                        )
                        frozen = snapshot[offset : offset + length]
                        if not _bytes_equal(
                            fixture.cache.read_swa(0, request, query_positions), frozen
                        ):
                            raise AssertionError(
                                (
                                    rank,
                                    request,
                                    "staged snapshot differs from native packed rows",
                                )
                            )
                        offset += length
                    packet.zero_()
                    receive.gather(torch.full_like(packet, 117))
                    offset = 0
                    for request, length in enumerate(lengths):
                        query_positions = torch.arange(
                            prefixes[request],
                            prefixes[request] + length,
                            device=actual.device,
                        )
                        if not _bytes_equal(
                            fixture.cache.read_swa(0, request, query_positions),
                            snapshot[offset : offset + length],
                        ):
                            raise AssertionError(
                                (
                                    rank,
                                    request,
                                    "staged snapshot changed after local/scratch reuse",
                                )
                            )
                        offset += length
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_CP8_WORLD8") == "1",
    "explicit eight-PPU lease required",
)
class CP8LocalSWAPackWorld8(unittest.TestCase):
    def test_real_cp_transport_and_swa_snapshots_survive_packet_and_scratch_reuse(self):
        self.assertTrue(is_ppu())
        with tempfile.TemporaryDirectory(prefix="cp8-local-swa-pack-") as directory:
            torch.multiprocessing.spawn(
                _world8_worker,
                args=(str(Path(directory) / "store"),),
                nprocs=8,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
