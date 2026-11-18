"""CP fake-route ownership and mask contracts against same-snapshot control.

CPU cases run actual PPUMoE preparation/pattern/forward lifecycle methods,
with explicit native-operation doubles. The gated PPU case runs the real
existing native FakeBalance pattern builder and mask kernel. Its standalone
buffer handle is not an EP8 transport or model performance qualification.
"""

from contextlib import ExitStack
from dataclasses import replace
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_fake_routes as cp_fake
from rtp_llm.models_py.modules.dsv41.ppu import moe
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu


class _Buffer:
    def __init__(self):
        self.destroy = Mock()


def _engine(device="cpu", capacity=4096):
    engine = moe.PPUMoE.__new__(moe.PPUMoE)
    engine.backend, engine.precision = "megamoe", "w4a4"
    engine.ep_size, engine.ep_rank = 8, 0
    engine.num_routed_experts, engine.num_physical_experts, engine.num_local_experts = 384, 384, 48
    engine.hidden_size, engine.inter_dim, engine.top_k = 5120, 2304, 6
    engine._fake_balance_config, engine._fake_balance_policy = (True, 6, 0, 1), "rtp"
    engine._mega_buffer, engine._mega_capacity = _Buffer(), capacity
    engine.group = object()  # Standalone ownership token; no communication is run.
    engine._mega_device, engine._mega_generation = torch.device(device), 7
    engine._cp_fake_epoch, engine._cp_fake_active = 0, None
    engine._prefill_fake_pattern = engine._prefill_fake_pattern_generation = None
    engine._prefill_fake_pattern_builds = engine._prefill_fake_pattern_calls = 0
    engine._fake_balance_op = None
    engine._eplb_layers, engine._eplb_routes = {}, {}
    engine._weights, engine._current_layer = {0: ()}, 0
    engine._dp_prepare_epoch, engine._dp_active_preparation = 9, None
    return engine


def _inputs(rows, slots, device="cpu"):
    ids = (torch.arange(rows * slots, device=device, dtype=torch.int64).reshape(rows, slots) % 384)
    weights = torch.arange(rows * slots, device=device, dtype=torch.float32).reshape(rows, slots) * 0.03125
    ids[0, :6] = torch.tensor((-1, -7, -3, -9, -12, -2), device=device)
    if rows > 3:
        ids[1, 1], ids[2, 5], ids[3, 2] = -17, -3, -99
        weights[1, 0], weights[2, 0] = torch.nan, torch.inf
    if rows == 144:
        # CP8 B8 rank0 has14 real/18 padded rows per request. The gate has
        # already represented those invalid rows with negative route slots.
        ids[torch.arange(rows, device=device) % 18 >= 14, :6] = -1
    return ids.contiguous(), weights.contiguous()


def _control(engine, ids, weights, slots=None):
    """Retain this source snapshot's RTP mapping/weight constant and policy.

    Some snapshots return six real slots, others retain padded eight slots.
    Normalize only that communication shape for this adapter comparison.
    Never hardcode a uniform route-weight value across snapshot versions.
    """
    selected, scaled = engine._apply_fake_balance(ids, weights)
    slots = ids.shape[1] if slots is None else slots
    return moe.pad_topk_for_dispatch(selected[:, :6], scaled[:, :6], slots=slots)


class _NativePatternDouble:
    def __init__(self, engine):
        self.engine, self.calls = engine, []

    def __call__(self, ids, weights):
        self.calls.append(ids.shape[0])
        # The real native builder overwrites every uninitialized pattern row.
        source = torch.zeros_like(ids)
        # Call the class implementation directly: a forward-dispatch spy on
        # the instance must not mistake this test-only native authority for a
        # production fallback invocation.
        selected, scaled = moe.PPUMoE._apply_fake_balance(self.engine, source, torch.zeros_like(weights))
        ids.copy_(selected[:, :6])
        weights.copy_(scaled[:, :6])


class _MaskDouble:
    def __init__(self, engine):
        source = torch.zeros(1, 6, dtype=torch.int64)
        _, weights = _control(engine, source, torch.zeros(1, 6))
        self.weight = float(weights[0, 0])
        self.calls, self.error = [], None

    def __getitem__(self, grid):
        def launch(original, pattern, out_ids, out_weights, rows, *, BLOCK, SLOTS=8):
            self.calls.append((rows, SLOTS, grid))
            if self.error is not None:
                raise self.error
            live = original[:, :6] >= 0
            selected = torch.where(live, pattern[:rows], original[:, :6])
            scaled = torch.where(live, self.weight, 0).float()
            selected, scaled = moe.pad_topk_for_dispatch(selected, scaled, slots=SLOTS)
            out_ids.copy_(selected)
            out_weights.copy_(scaled)
        return launch


class CP8FakeRoutePatternContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stream = NS(handle=19, device=None)
        self.stack.enter_context(patch.object(cp_fake, "_device_supported", return_value=True))
        self.stack.enter_context(patch.object(cp_fake, "_stream_key", side_effect=lambda device:
            (device if self.stream.device is None else self.stream.device, self.stream.handle)))
        self.stack.enter_context(patch.object(torch.distributed, "all_reduce", side_effect=AssertionError("no new agreement")))
        self.stack.enter_context(patch.object(torch.distributed, "all_gather_into_tensor", side_effect=AssertionError("no CP transport here")))

    def fixture(self):
        engine = _engine()
        native, mask = _NativePatternDouble(engine), _MaskDouble(engine)
        engine._fake_balance_op = native
        kernels = NS(mask_fake_routes=mask, triton=NS(cdiv=lambda n, d: (n + d - 1) // d))
        guard = patch.object(moe, "_get_triton_kernels", return_value=kernels)
        self.stack.enter_context(guard)
        receipt = cp_fake.prepare(engine, engine._mega_buffer, engine._mega_device, 4096)
        self.assertIs(type(receipt), cp_fake.CPForwardMegaBuffer)
        return engine, native, mask, receipt

    def test_actual_full_cp8_preparation_wraps_uniform_buffer_without_another_allocation(self):
        engine = _engine("cuda:0")
        raw, world = engine._mega_buffer, object()
        engine.group = world
        engine._ensure_mega_capacity = Mock(return_value=raw)
        with patch.object(torch.distributed, "is_initialized", return_value=True), patch.object(
            torch.distributed, "group", NS(WORLD=world)
        ), patch.object(torch, "empty", side_effect=AssertionError("receipt must not allocate tensors")):
            actual = engine.prepare_full_cp8(torch.device("cuda:0"), 4096)
        self.assertIs(type(actual), cp_fake.CPForwardMegaBuffer)
        self.assertIs(actual.engine, engine)
        self.assertIs(actual.buffer, raw)
        self.assertIs(engine._cp_fake_active, actual)
        engine._ensure_mega_capacity.assert_called_once_with(torch.device("cuda:0"), 4096, uniform_rows=True)
        self.assertEqual(engine._cp_fake_evidence["calls"], 0)

    def test_full4096_compact144_slots6_and8_keep_per_slot_masks_and_private_outputs(self):
        engine, native, kernel, receipt = self.fixture()
        for rows in (4096, 144):
            for slots in (6, 8):
                ids, weights = _inputs(rows, slots)
                original = ids.clone(), weights.view(torch.uint8).clone()
                expected = _control(engine, ids, weights)
                actual = cp_fake.try_balance(engine, ids, weights, receipt)
                self.assertEqual(actual[0].shape, ids.shape)
                for value, reference in zip(actual, expected):
                    self.assertTrue(torch.equal(value.view(torch.uint8), reference.contiguous().view(torch.uint8)))
                    self.assertFalse(any(torch._C._is_alias_of(value, source) for source in (ids, weights)))
                self.assertTrue(torch.equal(actual[0][:, :6][ids[:, :6] < 0], ids[:, :6][ids[:, :6] < 0]))
                self.assertTrue((actual[1][:, :6][ids[:, :6] < 0] == 0).all())
                if slots == 8:
                    self.assertTrue((actual[0][:, 6:] == -1).all())
                    self.assertTrue((actual[1][:, 6:] == 0).all())
                again = cp_fake.try_balance(engine, ids, weights, receipt)
                self.assertFalse(torch._C._is_alias_of(again[0], actual[0]))
                actual[0].fill_(-81)
                self.assertTrue(torch.equal(again[0], expected[0]))
                self.assertTrue(torch.equal(ids, original[0]))
                self.assertTrue(torch.equal(weights.view(torch.uint8), original[1]))
        self.assertEqual(native.calls, [4096])
        self.assertEqual(len(kernel.calls), 8)
        self.assertEqual(engine._prefill_fake_pattern_builds, 1)
        self.assertFalse(engine._prefill_fake_pattern.ids.is_inference())
        self.assertFalse(engine._prefill_fake_pattern.weights.is_inference())
        self.assertEqual(engine._cp_fake_evidence["calls"], 8)
        self.assertEqual(engine._cp_fake_evidence["pattern_builds"], 1)

    def test_new_forward_epoch_reuses_pattern_but_stales_old_receipt(self):
        engine, native, kernel, receipt = self.fixture()
        ids, weights = _inputs(144, 8)
        cp_fake.try_balance(engine, ids, weights, receipt)
        pattern = engine._prefill_fake_pattern
        current = cp_fake.prepare(engine, receipt.buffer, engine._mega_device, 4096)
        self.assertEqual(current.epoch, receipt.epoch + 1)
        with self.assertRaisesRegex(RuntimeError, "stale CP"):
            cp_fake.try_balance(engine, ids, weights, receipt)
        self.assertIsNotNone(cp_fake.try_balance(engine, ids, weights, current))
        self.assertIs(engine._prefill_fake_pattern, pattern)
        self.assertEqual(native.calls, [4096])
        self.assertEqual(engine._cp_fake_evidence["pattern_builds"], 0)
        self.assertEqual(engine._cp_fake_evidence["calls"], 1)

    def test_receipt_engine_binding_generation_capacity_bound_stream_and_configuration(self):
        for field in ("engine", "buffer", "capacity", "bound", "generation", "epoch", "device", "stream", "group", "binding"):
            engine, _, _, receipt = self.fixture()
            replacement = {"engine": object(), "buffer": object(), "capacity": 8192, "bound": 1,
                           "generation": 8, "epoch": 2, "device": torch.device("cuda:1"),
                           "stream": (torch.device("cpu"), 77), "group": object(), "binding": ()}[field]
            bad = replace(receipt, **{field: replacement})
            engine._cp_fake_active = bad
            with self.assertRaisesRegex(RuntimeError, "stale CP"):
                cp_fake.validate(engine, bad, 144, torch.device("cpu"))
        for change in ("active", "generation", "capacity", "buffer", "policy", "backend", "top_k", "geometry", "stream", "group"):
            engine, _, kernel, receipt = self.fixture()
            if change == "active":
                engine._cp_fake_active = replace(receipt)
            elif change == "generation":
                engine._mega_generation += 1
            elif change == "capacity":
                engine._mega_capacity += 1
            elif change == "buffer":
                engine._mega_buffer = _Buffer()
            elif change == "policy":
                engine._fake_balance_policy = "sail_round_robin"
            elif change == "backend":
                engine.backend = "deepep"
            elif change == "top_k":
                engine.top_k = 3
            elif change == "geometry":
                engine.inter_dim = 2048
            elif change == "group":
                engine.group = object()
            else:
                self.stream.handle += 1
            with self.assertRaisesRegex(RuntimeError, "stale CP"):
                cp_fake.try_balance(engine, *_inputs(144, 6), receipt)
            self.assertFalse(kernel.calls)
            self.stream.handle = 19
        engine, _, _, receipt = self.fixture()
        bad = replace(receipt, bound=8192)
        engine._cp_fake_active = bad
        with self.assertRaisesRegex(RuntimeError, "stale CP"):
            cp_fake.validate(engine, bad, 4097, torch.device("cpu"))

    def test_unsupported_repreparation_expires_the_previous_forward_permission(self):
        engine, _, _, receipt = self.fixture()
        engine._fake_balance_config = (False, 6, 0, 1)
        raw = cp_fake.prepare(engine, engine._mega_buffer, engine._mega_device, 4096)
        self.assertIs(raw, receipt.buffer)
        self.assertIsNone(engine._cp_fake_active)
        engine._fake_balance_config = (True, 6, 0, 1)
        with self.assertRaisesRegex(RuntimeError, "stale CP"):
            cp_fake.validate(engine, receipt, 144, torch.device("cpu"))

    def test_unsupported_admission_and_route_metadata_decline_before_pattern_launch(self):
        for change in ("disabled", "dp", "sail", "precision", "hidden", "rows", "buffer", "grad", "device"):
            engine = _engine()
            buffer, rows = engine._mega_buffer, 4096
            if change == "disabled":
                engine._fake_balance_config = (False, 6, 0, 1)
            elif change == "dp":
                engine._fake_balance_config = (True, 6, 0, 8)
            elif change == "sail":
                engine._fake_balance_policy = "sail_round_robin"
            elif change == "precision":
                engine.precision = "w4a8"
            elif change == "hidden":
                engine.hidden_size = 4096
            elif change == "rows":
                rows = 0
            elif change == "buffer":
                buffer = _Buffer()
            with torch.set_grad_enabled(change == "grad"), patch.object(
                cp_fake, "_device_supported", return_value=change != "device"
            ):
                self.assertIs(cp_fake.prepare(engine, buffer, engine._mega_device, rows), buffer)
            self.assertIsNone(engine._cp_fake_active)
        engine, native, kernel, receipt = self.fixture()
        ids, weights = _inputs(144, 8)
        strided_ids = torch.empty(144, 16, dtype=ids.dtype)[:, ::2]
        strided_weights = torch.empty(144, 16, dtype=weights.dtype)[:, ::2]
        strided_ids.copy_(ids)
        strided_weights.copy_(weights)
        for values in ((ids.int(), weights), (ids, weights.to(torch.bfloat16)),
                       (ids[:, ::2], weights[:, ::2]), (ids, weights[:, :6]),
                       (strided_ids, strided_weights), (torch._neg_view(ids), weights)):
            self.assertIsNone(cp_fake.try_balance(engine, *values, receipt))
        with torch.enable_grad():
            self.assertIsNone(cp_fake.try_balance(engine, ids, weights, receipt))
        with patch.object(cp_fake, "_device_supported", return_value=False):
            self.assertIsNone(cp_fake.try_balance(engine, ids, weights, receipt))
        self.assertFalse(native.calls)
        self.assertFalse(kernel.calls)

    def test_pattern_mutation_rebinding_missing_cache_and_partial_failure_do_not_remint(self):
        for change in ("ids_write", "weights_write", "clone", "missing", "generation"):
            engine, native, kernel, receipt = self.fixture()
            ids, weights = _inputs(144, 8)
            cp_fake.try_balance(engine, ids, weights, receipt)
            pattern = engine._prefill_fake_pattern
            if change == "ids_write":
                pattern.ids.add_(0)
            elif change == "weights_write":
                pattern.weights.add_(0)
            elif change == "clone":
                engine._prefill_fake_pattern = replace(pattern, ids=pattern.ids.clone())
            elif change == "missing":
                engine._prefill_fake_pattern = None
            else:
                engine._prefill_fake_pattern_generation += 1
            self.assertIsNone(cp_fake.try_balance(engine, ids, weights, receipt))
            self.assertEqual(native.calls, [4096])
            self.assertEqual(len(kernel.calls), 1)
        engine, _, kernel, receipt = self.fixture()
        broken = Mock(side_effect=RuntimeError("native pattern started"))
        engine._fake_balance_op = broken
        with self.assertRaisesRegex(RuntimeError, "pattern started"):
            cp_fake.try_balance(engine, *_inputs(144, 6), receipt)
        self.assertIsNone(cp_fake.try_balance(engine, *_inputs(144, 6), receipt))
        broken.assert_called_once()
        self.assertFalse(kernel.calls)

    def test_actual_forward_unwraps_before_dispatch_and_never_retries_native_failure(self):
        engine, _, kernel, receipt = self.fixture()
        ids, weights = _inputs(144, 6)
        x = torch.empty(144, 5120, dtype=torch.bfloat16)
        expected = _control(engine, ids, weights)
        with patch.object(engine, "_apply_fake_balance", wraps=engine._apply_fake_balance) as legacy, patch.object(
            engine, "_map_expert_ids", wraps=engine._map_expert_ids
        ) as mapping, patch.object(engine, "_forward_megamoe", return_value=x) as native:
            self.assertIs(engine.forward(x, ids, weights, layer_id=0, prepared_buffer=receipt), x)
        legacy.assert_not_called()
        self.assertIs(mapping.call_args.args[0], native.call_args.args[1])
        self.assertTrue(torch.equal(native.call_args.args[1], expected[0]))
        self.assertTrue(torch.equal(native.call_args.args[2], expected[1]))
        self.assertIs(native.call_args.kwargs["prepared_buffer"], receipt.buffer)
        kernel.error = RuntimeError("native mask started")
        with patch.object(engine, "_apply_fake_balance", side_effect=AssertionError("no serial retry")) as legacy, patch.object(
            engine, "_forward_megamoe", side_effect=AssertionError("mask failure precedes dispatch")
        ):
            with self.assertRaisesRegex(RuntimeError, "mask started"):
                engine.forward(x, ids, weights, layer_id=0, prepared_buffer=receipt)
        legacy.assert_not_called()

    def test_close_and_growth_stale_old_receipts_but_allow_one_new_generation_pattern(self):
        engine, native, _, receipt = self.fixture()
        ids, weights = _inputs(144, 8)
        cp_fake.try_balance(engine, ids, weights, receipt)
        raw = receipt.buffer
        with patch.object(torch.cuda, "synchronize") as synchronize:
            engine.close()
        synchronize.assert_called_once_with(device=torch.device("cpu"))
        raw.destroy.assert_called_once()
        self.assertIsNone(engine._cp_fake_active)
        with self.assertRaisesRegex(RuntimeError, "stale CP"):
            cp_fake.validate(engine, receipt, 144, torch.device("cpu"))
        engine._mega_buffer, engine._mega_capacity, engine._mega_generation = _Buffer(), 8192, 8
        current = cp_fake.prepare(engine, engine._mega_buffer, engine._mega_device, 4096)
        self.assertIsNotNone(cp_fake.try_balance(engine, ids, weights, current))
        self.assertEqual(native.calls, [4096, 8192])
        self.assertEqual(engine._cp_fake_evidence["pattern_builds"], 1)

    def test_actual_raw_dp_and_sail_forward_keep_same_snapshot_legacy_dispatch(self):
        for mode in ("raw", "none", "dp", "sail", "real"):
            engine = _engine()
            if mode == "dp":
                engine._fake_balance_config = (True, 6, 0, 8)
            elif mode == "sail":
                engine._fake_balance_policy = "sail_round_robin"
            elif mode == "real":
                engine._fake_balance_config = (False, 6, 0, 1)
            ids, weights = _inputs(144, 8)
            expected = engine._apply_fake_balance(ids, weights)
            x = torch.empty(144, 5120, dtype=torch.bfloat16)
            prepared = None if mode == "none" else engine._mega_buffer
            if mode == "dp":
                prepared = moe._DPForwardMegaBuffer(engine, engine._mega_buffer, 4096, 4096, 7, 9)
                engine._dp_active_preparation = prepared
            with patch.object(cp_fake, "try_balance", side_effect=AssertionError("legacy path entered CP adapter")), patch.object(
                engine, "_try_prefill_fake_balance", return_value=None
            ), patch.object(engine, "_apply_fake_balance", wraps=engine._apply_fake_balance) as legacy, patch.object(
                engine, "_forward_megamoe", return_value=x
            ) as native:
                self.assertIs(engine.forward(x, ids, weights, layer_id=0, prepared_buffer=prepared), x)
            legacy.assert_called_once()
            self.assertTrue(torch.equal(native.call_args.args[1], expected[0]))
            self.assertTrue(torch.equal(native.call_args.args[2].view(torch.uint8), expected[1].view(torch.uint8)))
            self.assertIs(native.call_args.kwargs["prepared_buffer"],
                          prepared.buffer if mode == "dp" else prepared)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU native fake-route test")
class CP8FakeRoutePatternDevice(unittest.TestCase):
    @torch.no_grad()
    def test_native_pattern_mask_matches_same_snapshot_existing_control_and_private_outputs(self):
        self.assertTrue(is_ppu())
        engine = _engine("cuda:0")
        receipt = cp_fake.prepare(engine, engine._mega_buffer, engine._mega_device, 4096)
        self.assertIs(type(receipt), cp_fake.CPForwardMegaBuffer)
        for rows in (4096, 144):
            for slots in (6, 8):
                ids, weights = _inputs(rows, slots, "cuda:0")
                original_ids, original_weights = ids.clone(), weights.view(torch.uint8).clone()
                expected = _control(engine, ids, weights)
                actual = cp_fake.try_balance(engine, ids, weights, receipt)
                self.assertIsNotNone(actual)
                for value, reference in zip(actual, expected):
                    self.assertEqual(value.shape, reference.shape)
                    self.assertTrue(torch.equal(value.view(torch.uint8), reference.contiguous().view(torch.uint8)),
                                    (rows, slots, "same-snapshot native/control mismatch"))
                    self.assertFalse(any(torch._C._is_alias_of(value, source) for source in (ids, weights)))
                again = cp_fake.try_balance(engine, ids, weights, receipt)
                self.assertFalse(torch._C._is_alias_of(actual[0], again[0]))
                self.assertTrue(torch.equal(ids, original_ids))
                self.assertTrue(torch.equal(weights.view(torch.uint8), original_weights))
        self.assertEqual(engine._prefill_fake_pattern_builds, 1)
        self.assertEqual(engine._cp_fake_evidence["calls"], 8)
        pattern = engine._prefill_fake_pattern
        self.assertFalse(pattern.ids.is_inference())
        side = torch.cuda.Stream(device="cuda:0")
        with torch.cuda.stream(side):
            with self.assertRaisesRegex(RuntimeError, "stale CP"):
                cp_fake.validate(engine, receipt, 144, engine._mega_device)
        engine._mega_generation += 1
        with self.assertRaisesRegex(RuntimeError, "stale CP"):
            cp_fake.validate(engine, receipt, 144, engine._mega_device)
        engine._mega_generation -= 1
        engine._fake_balance_policy = "sail_round_robin"
        with self.assertRaisesRegex(RuntimeError, "stale CP"):
            cp_fake.validate(engine, receipt, 144, engine._mega_device)
        engine._fake_balance_policy = "rtp"
        with patch.object(cp_fake, "_device_supported", return_value=False):
            self.assertIs(cp_fake.prepare(engine, receipt.buffer, engine._mega_device, 4096), receipt.buffer)
        with self.assertRaisesRegex(RuntimeError, "stale CP"):
            cp_fake.validate(engine, receipt, 144, engine._mega_device)
        receipt = cp_fake.prepare(engine, engine._mega_buffer, engine._mega_device, 4096)
        self.assertIsNotNone(cp_fake.try_balance(engine, *_inputs(144, 8, "cuda:0"), receipt))
        self.assertIs(engine._prefill_fake_pattern, pattern)
        pattern.ids.add_(0)
        self.assertIsNone(cp_fake.try_balance(engine, *_inputs(144, 8, "cuda:0"), receipt))
        self.assertEqual(engine._prefill_fake_pattern_builds, 1)
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
