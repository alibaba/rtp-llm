"""Cold start, publish and restart of resident weights on one GPU.

The point of the main test is that ``_initialize`` below -- the stand-in for the python model's
``initialize()`` -- is byte-for-byte the same code on a cold start and on a restart, and that the
weight dicts and transform outputs it leaves behind are identical either way.
"""

import gc
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

import torch

from rtp_llm.resident_weights import (
    GLOBAL_SCOPE,
    ResidentWeights,
    WeightDict,
    resident_init_scalar,
    resident_init_tensors,
    should_consume_loaded,
    store,
    writable_view,
)
from rtp_llm.resident_weights.vmm import VmmAllocation

FINGERPRINT = {"model_type": "fake", "checkpoint_path": "/fake", "layers": "1"}


class _ModelWeights:
    """The two attributes of ModelWeights that resident weights use."""

    def __init__(self, layers, global_weights):
        self.weights = layers
        self.global_weights = global_weights


def _build_weights(plain: bool = False) -> _ModelWeights:
    torch.manual_seed(0)
    dev = "cuda"
    pair = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)  # 2 MiB, two views
    twin = torch.randn(
        256, 256, device=dev, dtype=torch.bfloat16
    )  # one view, two names
    layer = {
        "big_fp8": torch.randn(2048, 1024, device=dev).to(torch.float8_e4m3fn),
        "transposed": torch.randn(2048, 1024, device=dev, dtype=torch.bfloat16).t(),
        "small": torch.randn(16, device=dev, dtype=torch.bfloat16),
        "pair_a": pair[:512],
        "pair_b": pair[512:],
        "twin_a": twin,
        "twin_b": twin,
        "e8m0": torch.randint(0, 255, (64,), device=dev, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        ),
        "empty": torch.empty(0, 8, device=dev),
    }
    globals_ = {"embed": torch.randn(4096, 512, device=dev, dtype=torch.bfloat16)}
    if plain:
        return _ModelWeights([layer], globals_)
    return _ModelWeights([WeightDict(0, layer)], WeightDict(GLOBAL_SCOPE, globals_))


def _fill(model_weights: _ModelWeights, restored) -> None:
    """Put the attached tensors back where the loader would have, as ``T1`` does."""
    for item in restored:
        target = (
            model_weights.global_weights
            if item.scope == GLOBAL_SCOPE
            else model_weights.weights[item.scope]
        )
        target[item.name] = item.tensor


def _initialize(weights: _ModelWeights) -> dict:
    """The model's initialization-time weight transforms, unchanged between the two paths."""
    layer = weights.weights[0]
    outputs = {}

    # Preprocessing that consumes loaded weights: skipped entirely on a restart.
    if should_consume_loaded(layer, ("small",)):
        layer.pop("small")

    # A transform that consumes a tensor the model renamed first (checkpoint name -> canonical).
    if "big_fp8" in layer:
        layer["canonical"] = layer.pop("big_fp8")
    (outputs["fp8"],) = resident_init_tensors(
        layer,
        transform_key="test.fp8",
        output_names=("test.fp8.out",),
        consumed_names=("canonical",),
        builder=lambda: (layer.pop("canonical").float(),),
    )

    # A scalar read off a tensor the next transform consumes: unreadable on a restart.
    outputs["twin_rows"] = resident_init_scalar(
        layer, "test.twin_rows", lambda: int(layer["twin_a"].shape[0])
    )
    # One view under two names: consuming either settles both.
    (outputs["twin"],) = resident_init_tensors(
        layer,
        transform_key="test.twin",
        output_names=("test.twin.out",),
        consumed_names=("twin_a",),
        builder=lambda: (torch.cat([layer.pop("twin_a"), layer.pop("twin_b")]),),
    )

    # A merge that re-points the loader names at slices of its own output, as the dsv41 QKV
    # merge does; the shape it reads back must survive a restart exactly.
    q_rows = layer["pair_a"].shape[0]
    (merged,) = resident_init_tensors(
        layer,
        transform_key="test.qkv",
        output_names=("test.qkv.w",),
        consumed_names=("pair_a", "pair_b"),
        builder=lambda: (torch.cat([layer["pair_a"], layer["pair_b"]]).float(),),
    )
    layer["pair_a"], layer["pair_b"] = merged.split((q_rows, merged.shape[0] - q_rows))
    outputs["merged"] = merged
    outputs["q_rows"] = q_rows

    # A transform whose branch produces nothing: its inputs stay loaded weights.
    assert (
        resident_init_tensors(
            layer,
            transform_key="test.degenerate",
            output_names=("test.degenerate.out",),
            consumed_names=("e8m0",),
            builder=lambda: None,
        )
        is None
    )
    layer["renamed"] = layer.pop("e8m0")
    return outputs


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def _snapshot(weights: _ModelWeights, outputs: dict) -> dict:
    out = {}
    for scope, weight_dict in [
        (0, weights.weights[0]),
        (GLOBAL_SCOPE, weights.global_weights),
    ]:
        for name, tensor in weight_dict.items():
            out[f"{scope}/{name}"] = [
                _digest(tensor),
                list(tensor.shape),
                list(tensor.stride()),
                tensor.storage_offset(),
                str(tensor.dtype),
            ]
    for name, value in outputs.items():
        out[f"output/{name}"] = (
            [_digest(value), list(value.shape), str(value.dtype)]
            if isinstance(value, torch.Tensor)
            else value
        )
    return out


def _slot_key(role: str = "target") -> str:
    return f"{torch.cuda.get_device_properties(0).uuid}/{role}"


def _manager(
    verify: bool = False, fingerprint=FINGERPRINT, role: str = "target"
) -> ResidentWeights:
    manager = ResidentWeights(role, 0, dict(fingerprint), verify)
    assert manager.ensure_store_running()
    return manager


class ResidentWeightsTest(unittest.TestCase):
    def test_vmm_capability_gate_accepts_the_test_device(self):
        from rtp_llm.resident_weights.vmm import is_available

        self.assertTrue(is_available(0))

    def setUp(self):
        os.environ["RTP_LLM_RESIDENT_STORE"] = (
            f"rtp_llm_resident_test.{uuid.uuid4().hex}"
        )
        self.name = store.default_store_name()
        store.ensure_running(self.name, owner_pid=os.getpid())

    def tearDown(self):
        try:
            store.stop_store(self.name)
        except ConnectionRefusedError:
            pass

    def _listed(self):
        return store.list_store(self.name).slots.get(_slot_key())

    def _published_record(self, fingerprint=FINGERPRINT):
        text = "\n".join(f"{k}: {v}" for k, v in sorted(fingerprint.items()))
        result = store.attach_slot(
            self.name, _slot_key(), hashlib.sha256(text.encode()).hexdigest()
        )
        for fd in result.allocation_fds:
            os.close(fd)
        return result.record

    def _cold_start(self, verify=False, plain=False, before_move=lambda w: None):
        weights = _build_weights(plain)
        before_move(weights)
        manager = _manager(verify)
        if verify:
            manager.begin_verify()  # what load_weights does before the native load
        manager.move_to_vmm(weights)
        return manager, weights, _initialize(weights)

    # ---- relocation ---------------------------------------------------------------------

    def test_relocation_keeps_bytes_layout_and_storage_sharing(self):
        weights = _build_weights()
        before = _snapshot(weights, {})
        _manager().move_to_vmm(weights)
        self.assertEqual(_snapshot(weights, {}), before)
        layer = weights.weights[0]
        self.assertEqual(
            layer["pair_a"].untyped_storage().data_ptr(),
            layer["pair_b"].untyped_storage().data_ptr(),
        )
        self.assertFalse(layer["transposed"].is_contiguous())
        self.assertEqual(layer["empty"].numel(), 0)

    def test_relocation_frees_each_original_once_moved(self):
        # Holding every original until the end would need twice the weights' memory.
        weights = _build_weights()
        allocated = []
        create = VmmAllocation.create

        def recording(size_bytes, device):
            allocated.append(torch.cuda.memory_allocated(device))
            return create(size_bytes, device)

        with patch.object(VmmAllocation, "create", recording):
            _manager().move_to_vmm(weights)
        self.assertGreater(len(allocated), 2)
        self.assertLess(allocated[-1], allocated[0])

    def test_plain_weight_dicts_are_wrapped_by_bind(self):
        # _load_from_ft_style replaces the dicts with plain ones before the manager sees them.
        weights = _build_weights(plain=True)
        manager = _manager()
        manager.move_to_vmm(weights)
        self.assertIsInstance(weights.weights[0], WeightDict)
        self.assertIsInstance(weights.global_weights, WeightDict)
        self.assertEqual(weights.weights[0].scope, 0)
        self.assertEqual(weights.global_weights.scope, GLOBAL_SCOPE)
        self.assertIs(weights.weights[0].manager, manager)

    # ---- cold start, publish and restart ------------------------------------------------

    def test_restart_rebuilds_exactly_what_the_cold_start_published(self):
        manager, weights, outputs = self._cold_start()
        self.assertEqual(outputs["twin_rows"], 256)
        self.assertEqual(outputs["q_rows"], 512)
        expected = _snapshot(weights, outputs)
        manager.finalize()
        self.assertIsNotNone(self._listed())
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(expected, f)
        self.addCleanup(os.unlink, f.name)
        child = subprocess.run(
            [sys.executable, __file__, "restart", f.name],
            capture_output=True,
            text=True,
        )
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
        # The weights are mapped read-only: a buggy kernel cannot corrupt the shared copy.
        fault = subprocess.run(
            [sys.executable, __file__, "write_read_only"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(fault.returncode, 0)
        self.assertIn("illegal memory access", fault.stdout + fault.stderr)
        child = subprocess.run(
            [sys.executable, __file__, "restart", f.name],
            capture_output=True,
            text=True,
        )
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)

    def test_a_degenerate_transform_keeps_its_inputs_as_loaded_weights(self):
        manager, weights, outputs = self._cold_start()
        manager.finalize()
        record = self._published_record()
        groups = record.manifest.transform_returned_tensors[0]
        self.assertFalse(groups["test.degenerate"])
        self.assertTrue(groups["test.qkv"])
        self.assertEqual(record.manifest.init_scalars[0]["test.twin_rows"], 256)
        loaded = {
            spec.name: spec
            for spec in record.manifest.tensors
            if spec.origin is store.WeightOrigin.LOADED and spec.scope == 0
        }
        self.assertEqual(
            sorted(loaded), ["e8m0", "empty", "pair_a", "pair_b", "transposed"]
        )
        # The re-pointed names are published with the slice they ended up viewing.
        self.assertEqual(loaded["pair_a"].shape, [512, 1024])
        self.assertEqual(loaded["pair_a"].dtype, "float32")
        self.assertEqual(loaded["empty"].storage_index, -1)
        del outputs

    # ---- the publish-time account must balance ------------------------------------------

    def test_consumption_outside_the_entry_functions_is_not_published(self):
        weights = _build_weights()
        manager = _manager()
        manager.move_to_vmm(weights)
        del weights.weights[0]["big_fp8"]
        with self.assertLogs(level=logging.ERROR) as logs:
            manager.finalize()
        self.assertIn("0/big_fp8", "\n".join(logs.output))
        self.assertIsNone(self._listed())

    def test_a_sibling_view_consumed_on_the_side_is_reported(self):
        # pair_a and pair_b share one storage but are different views: settling pair_a says
        # nothing about pair_b, and a transform that quietly takes both is a bug.
        weights = _build_weights()
        manager = _manager()
        manager.move_to_vmm(weights)
        layer = weights.weights[0]
        kept = resident_init_tensors(
            layer,
            transform_key="test.pair",
            output_names=("test.pair.out",),
            consumed_names=("pair_a",),
            builder=lambda: (
                torch.cat([layer.pop("pair_a"), layer.pop("pair_b")]).float(),
            ),
        )
        with self.assertLogs(level=logging.ERROR) as logs:
            manager.finalize()
        output = "\n".join(logs.output)
        self.assertIn("0/pair_b", output)
        self.assertNotIn("0/pair_a", output)
        self.assertIsNone(self._listed())
        del kept

    def test_one_view_under_two_names_is_settled_together(self):
        weights = _build_weights()
        manager = _manager()
        manager.move_to_vmm(weights)
        layer = weights.weights[0]
        kept = resident_init_tensors(
            layer,
            transform_key="test.twin",
            output_names=("test.twin.out",),
            consumed_names=("twin_a",),
            builder=lambda: (torch.cat([layer.pop("twin_a"), layer.pop("twin_b")]),),
        )
        with self.assertNoLogs(level=logging.ERROR):
            manager.finalize()
        self.assertIsNotNone(self._listed())
        del kept

    def test_a_replaced_weight_dict_is_refused(self):
        weights = _build_weights()
        manager = _manager()
        manager.move_to_vmm(weights)
        weights.weights[0] = {}
        with self.assertLogs(level=logging.ERROR) as logs:
            manager.finalize()
        self.assertIn("the weight dict was replaced", "\n".join(logs.output))
        self.assertIsNone(self._listed())

    # ---- a replay must line up with the record exactly ----------------------------------

    def _attach_into_fresh(self):
        restarted = _manager()
        loaded = restarted.attach()
        self.assertIsNotNone(loaded)
        fresh = _ModelWeights([{}], {})
        _fill(fresh, loaded)
        restarted.bind(fresh)
        return restarted, fresh

    def test_a_replay_that_never_asks_for_a_transform_is_refused(self):
        manager, weights, outputs = self._cold_start()
        manager.finalize()
        del weights, outputs
        restarted, fresh = self._attach_into_fresh()
        with self.assertRaisesRegex(RuntimeError, "never asked for"):
            restarted.finalize()
        self.assertIsNone(self._listed())

    def test_a_replay_of_a_transform_that_was_not_published_is_refused(self):
        manager, weights, outputs = self._cold_start()
        manager.finalize()
        del weights, outputs
        restarted, fresh = self._attach_into_fresh()
        with self.assertRaisesRegex(RuntimeError, "has no transform group"):
            resident_init_tensors(
                fresh.weights[0],
                transform_key="test.unpublished",
                output_names=("x",),
                consumed_names=(),
                builder=lambda: None,
            )
        self.assertIsNone(self._listed())

    def test_a_replay_of_a_scalar_that_was_not_published_is_refused(self):
        manager, weights, outputs = self._cold_start()
        manager.finalize()
        del weights, outputs
        restarted, fresh = self._attach_into_fresh()
        with self.assertRaisesRegex(RuntimeError, "has no init scalar"):
            resident_init_scalar(fresh.weights[0], "test.unpublished", lambda: 1)
        self.assertIsNone(self._listed())

    def test_fingerprint_change_drops_the_record(self):
        manager, weights, outputs = self._cold_start()
        manager.finalize()
        with self.assertLogs(level=logging.WARNING) as logs:
            self.assertIsNone(
                _manager(fingerprint=dict(FINGERPRINT, layers="2")).attach()
            )
        self.assertIn("layers", "\n".join(logs.output))
        self.assertIsNone(self._listed())

    # ---- fault injection ----------------------------------------------------------------

    def test_a_failed_import_leaks_no_descriptors(self):
        manager, weights, outputs = self._cold_start()
        manager.finalize()
        del weights, outputs
        before = len(os.listdir("/proc/self/fd"))
        real = VmmAllocation.from_fd
        calls = []

        def failing(fd, size_bytes, device, writable):
            calls.append(fd)
            if len(calls) == 3:
                raise RuntimeError("injected import failure")
            return real(fd, size_bytes, device, writable)

        with patch.object(VmmAllocation, "from_fd", failing):
            with self.assertRaisesRegex(RuntimeError, "injected import failure"):
                _manager().attach()
        gc.collect()
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)

    def test_a_stopped_store_degrades_without_breaking_the_engine(self):
        weights = _build_weights()
        manager = _manager()
        manager.move_to_vmm(weights)
        store.stop_store(self.name)
        with self.assertLogs(level=logging.WARNING) as logs:
            manager.finalize()
        self.assertIn("not published", "\n".join(logs.output))

    # ---- verify mode and EPLB -----------------------------------------------------------

    def test_verify_mode_reports_changed_weights(self):
        manager, *held = self._cold_start()
        manager.finalize()
        manager, weights, outputs = self._cold_start(verify=True)
        with self.assertLogs(level=logging.INFO) as logs:
            manager.finalize()
        self.assertIn("match the published ones", "\n".join(logs.output))
        manager, weights, outputs = self._cold_start(verify=True)
        writable_view(weights.global_weights["embed"])[0, 0] += 1
        with self.assertLogs(level=logging.ERROR) as logs:
            manager.finalize()
        self.assertIn("global/embed", "\n".join(logs.output))
        # Same bytes, another layout: a checksum alone would match.
        manager, *held = self._cold_start()
        manager.finalize()

        def reinterpret(weights):
            embed = weights.global_weights["embed"]
            weights.global_weights["embed"] = embed.view(torch.int16)

        manager, weights, outputs = self._cold_start(
            verify=True, before_move=reinterpret
        )
        with self.assertLogs(level=logging.ERROR) as logs:
            manager.finalize()
        self.assertIn("global/embed", "\n".join(logs.output))

    def test_eplb_placement_survives_restart_unless_interrupted(self):
        manager, weights, outputs = self._cold_start()
        manager.finalize()
        manager.eplb_begin()
        self.assertEqual(
            store.attach_slot(self.name, manager.slot_key, manager.fingerprint).status,
            store.AttachStatus.EPLB_DIRTY,
        )
        manager.eplb_commit({0: [1, 0]})
        del weights, outputs
        restarted = _manager()
        self.assertIsNotNone(restarted.attach())
        self.assertEqual(restarted.eplb_placement, {0: [1, 0]})


def _restart(expected_path: str) -> None:
    expected = json.load(open(expected_path))
    manager = _manager()
    loaded = manager.attach()
    assert loaded is not None, "no published weights"
    weights = _ModelWeights([{}], {})
    _fill(weights, loaded)
    manager.bind(weights)
    outputs = _initialize(weights)
    got = _snapshot(weights, outputs)
    assert got == expected, (
        "restored weights differ: "
        f"{sorted(set(got) ^ set(expected))} "
        f"{sorted(k for k in set(got) & set(expected) if got[k] != expected[k])}"
    )
    manager.finalize()  # every published result was asked for exactly once
    # EPLB writes through a second, writable mapping that the read-only one sees.
    embed = weights.global_weights["embed"]
    alias = writable_view(embed)
    saved = alias[0, 0].item()
    alias[0, 0] = 7.0
    torch.cuda.synchronize()
    assert embed[0, 0].item() == 7.0
    alias[0, 0] = saved
    torch.cuda.synchronize()


def _write_read_only() -> None:
    loaded = _manager().attach()
    tensor = {(item.scope, item.name): item.tensor for item in loaded}[
        (GLOBAL_SCOPE, "embed")
    ]
    tensor.add_(1)
    torch.cuda.synchronize()


class _FakeLoader:
    """The two ModelLoader methods ``ResidentWeights.load_weights`` calls."""

    def __init__(self, weights: _ModelWeights, restored_into=None):
        self.weights = weights
        self.restored_into = restored_into
        self.loaded_from_checkpoint = 0
        self.assembled = 0
        self.assemble_args = None

    def load_weights(self, device, global_weight_aliases):
        self.loaded_from_checkpoint += 1
        return self.weights

    def assemble_weights(self, device, aliases, restored):
        self.assembled += 1
        self.assemble_args = (device, dict(aliases), list(restored))
        _fill(self.restored_into, restored)
        return self.restored_into


class ResidentLoadWeightsTest(unittest.TestCase):
    """``load_weights`` picks the record or the checkpoint, and nothing in between."""

    def setUp(self):
        os.environ["RTP_LLM_RESIDENT_STORE"] = (
            f"rtp_llm_resident_test.{uuid.uuid4().hex}"
        )
        self.name = store.default_store_name()
        store.ensure_running(self.name, owner_pid=os.getpid())

    def tearDown(self):
        try:
            store.stop_store(self.name)
        except ConnectionRefusedError:
            pass

    def _publish_a_record(self, role: str = "target"):
        weights = _build_weights()
        manager = _manager(role=role)
        manager.move_to_vmm(weights)
        # The transform outputs are weights too: a model holds them past initialize(), and
        # dropping them here makes publish refuse the record.
        outputs = _initialize(weights)
        manager.finalize()
        return weights, outputs

    def _published_slot_keys(self):
        return set(store.list_store(self.name).slots)

    def test_a_miss_loads_the_checkpoint_and_moves_it_to_vmm(self):
        weights = _build_weights()
        loader = _FakeLoader(weights)
        manager = _manager()

        returned = manager.attach_or_load_weights(loader, "cuda:0", {})

        self.assertIs(returned, weights)
        self.assertEqual(loader.loaded_from_checkpoint, 1)
        self.assertEqual(loader.assembled, 0)
        # move_to_vmm() ran: the weight dicts are bound and the storages are resident.
        self.assertIsNotNone(manager.allocation_of(weights.weights[0]["big_fp8"]))

    def test_a_target_cold_load_releases_the_stale_draft_record(self):
        # A target that misses is about to read the checkpoint, and fastsafetensors refuses to
        # start unless the card has room; the draft's record would still be holding its share.
        draft_weights, draft_outputs = self._publish_a_record(role="draft")
        self.assertIn(_slot_key("draft"), self._published_slot_keys())
        del draft_weights, draft_outputs
        gc.collect()

        manager = _manager()
        manager.attach_or_load_weights(_FakeLoader(_build_weights()), "cuda:0", {})

        self.assertEqual(self._published_slot_keys(), set())

    def test_peer_miss_discards_a_local_hit_before_loading(self):
        from unittest.mock import patch

        published, outputs = self._publish_a_record()
        del published, outputs
        gc.collect()
        manager = _manager()
        loader = _FakeLoader(_build_weights())
        with patch(
            "rtp_llm.resident_weights.weights.all_ranks_true", return_value=False
        ):
            manager.attach_or_load_weights(loader, "cuda:0", {}, synchronize_ranks=True)
        self.assertFalse(manager._attached)
        self.assertIsNone(manager._replay)
        self.assertEqual(loader.loaded_from_checkpoint, 1)
        self.assertEqual(loader.assembled, 0)
        self.assertNotIn(_slot_key(), self._published_slot_keys())

    def test_host_tensor_fallback_does_not_activate_the_readonly_guard(self):
        import rtp_llm.resident_weights.weights as resident_module

        manager = _manager()
        manager.move_to_vmm(_ModelWeights([{"host": torch.ones(1)}], {}))
        self.assertNotIn(manager, resident_module._managers)
        self.assertFalse(resident_module.is_resident_weights_active())

    def test_a_draft_cold_load_leaves_the_target_record_alone(self):
        target_weights, target_outputs = self._publish_a_record()
        self.assertIn(_slot_key(), self._published_slot_keys())

        manager = _manager(role="draft")
        manager.attach_or_load_weights(_FakeLoader(_build_weights()), "cuda:0", {})

        # The target is the expensive record; the draft never evicts it to make room.
        self.assertIn(_slot_key(), self._published_slot_keys())
        self.assertTrue(target_outputs)
        del target_weights

    def test_aliases_and_excluded_names_stay_out_of_the_record(self):
        weights = _build_weights()
        alias = weights.global_weights["embed"]
        loader = _FakeLoader(weights)
        manager = _manager()

        manager.attach_or_load_weights(
            loader, "cuda:0", {"embed": alias}, excluded_names=("transposed",)
        )

        self.assertIsNone(manager.allocation_of(weights.global_weights["embed"]))
        self.assertIsNone(manager.allocation_of(weights.weights[0]["transposed"]))
        self.assertIsNotNone(manager.allocation_of(weights.weights[0]["big_fp8"]))

    def test_a_hit_assembles_from_the_record_without_reading_the_checkpoint(self):
        published, outputs = self._publish_a_record()
        before = _snapshot(published, outputs)
        del published, outputs
        gc.collect()

        # assemble_weights builds ModelWeights from scratch and fills only what the record
        # restored, so the attach side starts empty -- never from a cold dict.
        target = _ModelWeights([{}], {})
        loader = _FakeLoader(_build_weights(), restored_into=target)
        manager = _manager()

        returned = manager.attach_or_load_weights(loader, "cuda:0", {})

        self.assertIs(returned, target)
        self.assertEqual(loader.loaded_from_checkpoint, 0)
        self.assertEqual(loader.assembled, 1)
        self.assertEqual(loader.assemble_args[0], "cuda:0")
        replayed = _initialize(target)
        self.assertEqual(_snapshot(target, replayed), before)

    def test_changed_weight_processing_requires_explicit_invalidation(self):
        def initialize(weights, increment):
            (packed,) = resident_init_tensors(
                weights.weights[0],
                "manual.layout",
                ("packed",),
                ("raw",),
                lambda: (weights.weights[0].pop("raw") + increment,),
            )
            return packed

        cold = _ModelWeights([{"raw": torch.ones(4, device="cuda")}], {})
        cold_manager = _manager()
        cold_manager.attach_or_load_weights(_FakeLoader(cold), "cuda:0", {})
        cold_output = initialize(cold, 1)
        cold_manager.finalize()

        warm = _ModelWeights([{}], {})
        warm_loader = _FakeLoader(None, restored_into=warm)
        warm_manager = _manager()
        warm_manager.attach_or_load_weights(warm_loader, "cuda:0", {})
        # The same registered configuration reuses published bytes, not a new builder.
        warm_output = initialize(warm, 2)
        warm_manager.finalize()
        self.assertEqual(warm_loader.loaded_from_checkpoint, 0)
        self.assertTrue(torch.equal(warm_output, cold_output))

        store.drop_slot(self.name, _slot_key())
        reloaded = _ModelWeights([{"raw": torch.ones(4, device="cuda")}], {})
        reload_manager = _manager()
        reload_manager.attach_or_load_weights(_FakeLoader(reloaded), "cuda:0", {})
        new_output = initialize(reloaded, 2)
        reload_manager.finalize()
        self.assertTrue(torch.equal(new_output, torch.full_like(new_output, 3)))
        self.assertTrue(
            torch.equal(warm_output, cold_output)
        )  # drop is not live-model update

    def test_verify_mode_reloads_the_checkpoint_even_with_a_record(self):
        self._publish_a_record()
        weights = _build_weights()
        loader = _FakeLoader(weights)
        manager = _manager(verify=True)

        manager.attach_or_load_weights(loader, "cuda:0", {})

        self.assertEqual(loader.loaded_from_checkpoint, 1)
        self.assertEqual(loader.assembled, 0)
        # begin_verify took the record out of the store, so nothing is published yet.
        self.assertIsNone(store.list_store(self.name).slots.get(_slot_key()))

    def test_finalize_after_initialize_without_a_python_model_publishes_immediately(
        self,
    ):
        weights = _build_weights()
        manager = _manager()
        manager.attach_or_load_weights(_FakeLoader(weights), "cuda:0", {})
        outputs = _initialize(weights)

        manager.finalize_after_initialize(None)
        self.assertTrue(outputs)  # kept alive across finish(), as a model would

        self.assertIsNotNone(store.list_store(self.name).slots.get(_slot_key()))

    def test_finalize_after_initialize_with_a_python_model_waits_for_initialize(self):
        weights = _build_weights()
        manager = _manager()
        manager.attach_or_load_weights(_FakeLoader(weights), "cuda:0", {})

        class _PyModel:
            def __init__(self):
                self.calls = 0
                self.outputs = None

            def initialize(self):
                self.calls += 1
                self.outputs = _initialize(weights)
                return True

        py_model = _PyModel()
        manager.finalize_after_initialize(py_model)
        self.assertIsNone(store.list_store(self.name).slots.get(_slot_key()))

        self.assertTrue(py_model.initialize())
        self.assertEqual(py_model.calls, 1)
        self.assertIsNotNone(store.list_store(self.name).slots.get(_slot_key()))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "restart":
        _restart(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "write_read_only":
        _write_read_only()
    else:
        unittest.main()
