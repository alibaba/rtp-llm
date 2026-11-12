"""CPU tests for R27/C DSV41_GRAPH_PROBE_ROUTER (_v41_graph_probe.py).

Flag is cached at import — each case loads a fresh module instance under a
patched environment. CUDA capture/replay itself is GPU-only (probed on the
node); here we pin the flag matrix, the one/all claim semantics, and the
eager-fallback path (no CUDA calls when disabled / when ptrs keep changing
with capture impossible).
"""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch  # noqa: F401

ROOT = Path(__file__).resolve().parents[5]
for suffix in ["", ".models_py", ".models_py.modules", ".models_py.modules.dsv4"]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)

HELPER = ROOT / "rtp_llm/models_py/modules/dsv4/_v41_graph_probe.py"


def _load(flag=None, scope=None):
    env = {}
    if flag is not None:
        env["DSV41_GRAPH_PROBE_ROUTER"] = flag
    if scope is not None:
        env["DSV41_GRAPH_PROBE_ROUTER_SCOPE"] = scope
    with mock.patch.dict(os.environ, env, clear=True):
        spec = importlib.util.spec_from_file_location("dsv4_graph_probe_test", HELPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class _FakeTensor:
    def __init__(self, ptr, shape=(4, 4), dtype="bf16"):
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr

    @property
    def shape(self):
        return (4, 4)

    @property
    def dtype(self):
        return "bf16"


class FlagTest(unittest.TestCase):
    def test_default_on_all(self):
        # R28: adopted as default (escape = explicit "0" / scope "one").
        module = _load()
        self.assertTrue(module.graph_probe_enabled())
        self.assertEqual(module.probe_scope(), "all")

    def test_value_based(self):
        for value, expected in (("1", True), ("0", False), ("true", False)):
            self.assertEqual(_load(flag=value).graph_probe_enabled(), expected, value)

    def test_scope(self):
        self.assertEqual(_load(flag="1", scope="all").probe_scope(), "all")
        self.assertEqual(_load(flag="1", scope="one").probe_scope(), "one")

    def test_attn_flag_default_on(self):
        # R29: adopted as default after the pairs passed (escape = "0").
        self.assertTrue(_load().graph_attn_enabled())
        self.assertTrue(_load(flag="1").graph_attn_enabled())

        env = {"DSV41_GRAPH_ATTN": "1"}
        with mock.patch.dict(os.environ, env, clear=True):
            spec = importlib.util.spec_from_file_location(
                "dsv4_graph_probe_attn_test", HELPER
            )
            m2 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m2)
        self.assertTrue(m2.graph_attn_enabled())
        self.assertTrue(m2.graph_probe_enabled())  # R28 adopted default


class ClaimTest(unittest.TestCase):
    def test_one_scope_claims_once_per_device(self):
        module = _load(flag="1", scope="one")
        self.assertTrue(module.claim_one_layer(("boundary", "dev0")))
        self.assertFalse(module.claim_one_layer(("boundary", "dev0")))
        self.assertTrue(module.claim_one_layer(("boundary", "dev1")))
        self.assertTrue(module.claim_one_layer(("gate", "dev0")))

    def test_all_scope_never_blocks(self):
        module = _load(flag="1", scope="all")
        for _ in range(5):
            self.assertTrue(module.claim_one_layer(("boundary", "dev0")))


class FallbackTest(unittest.TestCase):
    def test_second_sighting_attempts_capture_and_fails_safely_on_cpu(self):
        module = _load(flag="1")
        probe = module.GraphReplay("test")
        calls = []
        t = _FakeTensor(0x1000)
        # first sighting: eager
        out, kind = probe.run([t], lambda _s: calls.append(1) or "eager1")
        self.assertEqual((out, kind), ("eager1", "fallback"))
        # second matching sighting: capture attempt -> fails without CUDA
        # (torch.cuda.synchronize on a CPU-only box raises) -> key blacklisted
        out, kind = probe.run([t], lambda _s: calls.append(1) or "eager2")
        self.assertEqual((out, kind), ("eager2", "fallback"))
        self.assertFalse(probe.permanently_off)
        self.assertEqual(len(probe.failed_keys), 1)
        out, kind = probe.run([t], lambda _s: calls.append(1) or "eager3")
        self.assertEqual((out, kind), ("eager3", "fallback"))
        self.assertEqual(len(calls), 3)

    def test_multi_key_tracking(self):
        module = _load(flag="1")
        probe = module.GraphReplay("test")
        a, b = _FakeTensor(0x1000), _FakeTensor(0x2000)
        # alternating addresses: each key captures on its 2nd sighting
        # (fails on CPU), then eager; both keys tracked independently
        for i in range(3):
            probe.run([a], lambda _s: None)
            probe.run([b], lambda _s: None)
        self.assertEqual(len(probe.failed_keys), 2)
        self.assertEqual(len(probe.graphs), 0)

    def test_replay_suspension_forces_eager_without_capture(self):
        # R58: the decoder-trim/CED compacted region suspends replay (stale
        # baked pre_mix_out addresses). Suspended runs must stay eager WITHOUT
        # recording sightings (the compact shape must never capture).
        module = _load(flag="1")
        probe = module.GraphReplay("test")
        calls = []
        t = _FakeTensor(0x1000)
        module.suspend_graph_replay()
        try:
            self.assertTrue(module.graph_replay_suspended())
            for _ in range(3):
                out, kind = probe.run([t], lambda _s: calls.append(1) or "eager")
                self.assertEqual((out, kind), ("eager", "fallback"))
        finally:
            module.resume_graph_replay()
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(probe.pending), 0)
        self.assertEqual(len(probe.failed_keys), 0)
        self.assertFalse(module.graph_replay_suspended())
        # after resume the normal path is restored (2nd sighting attempts
        # capture and blacklists the key on CPU)
        probe.run([t], lambda _s: "eager1")
        probe.run([t], lambda _s: "eager2")
        self.assertEqual(len(probe.failed_keys), 1)

    def test_key_extra_separates_side_channel_addresses(self):
        # R59: two sightings with the SAME input but DIFFERENT key_extra
        # (e.g. a re-pointed predecessor premix) must occupy DISTINCT keys —
        # a stale graph can never replay into the wrong side-channel buffer.
        # (Note: a key lives in pending AND failed_keys after a failed capture
        # attempt, so count the union.)
        module = _load(flag="1")
        probe = module.GraphReplay("test")
        x = _FakeTensor(0x1000)
        mix_a = _FakeTensor(0xA000, shape=(1024, 4))
        mix_b = _FakeTensor(0xB000, shape=(36, 4))

        def n_keys():
            return len(set(probe.pending) | set(probe.failed_keys))

        for _ in range(3):
            probe.run([x], lambda _s: None, key_extra=(mix_a,))
        # one distinct key (failed on CPU after the 2nd-sighting capture try)
        self.assertEqual(n_keys(), 1)
        probe.run([x], lambda _s: None, key_extra=(mix_b,))
        # new side-channel address/shape -> a second, independent key
        self.assertEqual(n_keys(), 2)
        # and the first key is untouched (its failed state persists)
        probe.run([x], lambda _s: None, key_extra=(mix_a,))
        self.assertEqual(n_keys(), 2)

    def test_suspend_compact_flag_default_off(self):
        self.assertFalse(_load().suspend_compact_enabled())
        env = {"DSV41_GRAPH_REPLAY_SUSPEND_COMPACT": "1"}
        with mock.patch.dict(os.environ, env, clear=True):
            import importlib.util as _ilu

            spec = _ilu.spec_from_file_location("dsv4_graph_probe_test2", HELPER)
            module = _ilu.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.assertTrue(module.suspend_compact_enabled())

    def test_boundary_key_extra_reads_predecessor_premix(self):
        module = _load()

        class _Unit:
            def __init__(self, mix):
                self.pre_mix_out = mix

        import weakref

        mix = _FakeTensor(0xA000)
        prev = _Unit(mix)
        unit = _Unit(None)
        unit._previous_ref = weakref.ref(prev)
        self.assertEqual(module.boundary_key_extra(unit), (mix,))
        # no predecessor -> empty (nothing side-channelled into the boundary)
        solo = _Unit(None)
        solo._previous_ref = None
        self.assertEqual(module.boundary_key_extra(solo), ())
        # predecessor without premix yet -> empty
        prev2 = _Unit(None)
        unit2 = _Unit(None)
        unit2._previous_ref = weakref.ref(prev2)
        self.assertEqual(module.boundary_key_extra(unit2), ())

    def test_not_implemented_error_is_permanent(self):
        module = _load(flag="1")
        probe = module.GraphReplay("test")
        a = _FakeTensor(0x1000)
        calls = []

        class _FakeGraph:
            def replay(self):
                raise AssertionError("replay must not run after failed capture")

        class _FakeGraphCtx:
            def __init__(self, g):
                self.g = g

            def __enter__(self):
                return self.g

            def __exit__(self, *args):
                return False

        def fn_ok(_staged):
            return calls.append("eager") or "out"

        boom_once = [True]

        def fn_boom_once(_staged):
            if boom_once[0]:
                boom_once[0] = False
                raise NotImplementedError("contract refusal")
            return calls.append("eager-after-failed-capture") or "out"

        with mock.patch.object(
            torch.cuda, "synchronize", lambda: None, create=True
        ), mock.patch.object(
            torch.cuda, "CUDAGraph", _FakeGraph, create=True
        ), mock.patch.object(
            torch.cuda, "graph", _FakeGraphCtx, create=True
        ):
            # first sighting: eager fallback
            out, kind = probe.run([a], fn_ok)
            self.assertEqual((out, kind), ("out", "fallback"))
            # second sighting: capture attempt -> fn raises NotImplementedError
            # inside the (fake) capture context -> probe falls back (fn now
            # succeeds) and marks the probe permanently off
            out, kind = probe.run([a], fn_boom_once)
            self.assertEqual((out, kind), ("out", "fallback"))
            self.assertTrue(probe.permanently_off)
            # third sighting: straight to eager, no further capture attempts
            out, kind = probe.run([a], fn_ok)
            self.assertEqual((out, kind), ("out", "fallback"))
        self.assertEqual(calls, ["eager", "eager-after-failed-capture", "eager"])

    def test_side_effect_callbacks_stored_per_key(self):
        module = _load(flag="1")
        probe = module.GraphReplay("test")
        t = _FakeTensor(0x1000)
        probe.run([t], lambda _s: "eager", side_get=lambda: "side-v1")
        # capture attempt fails on CPU; no side stored
        probe.run(
            [t],
            lambda _s: "eager2",
            side_get=lambda: "side-v1",
            side_set=lambda v: None,
        )
        self.assertEqual(len(probe.graphs), 0)

    def test_premix_pin_default(self):
        # boundary + attn probes are adopted default-on (R28/R29) => pin on
        self.assertTrue(_load(flag=None).premix_pin_enabled())
        # fully-off needs BOTH flags escaped (either one on => pin on)
        with mock.patch.dict(
            os.environ,
            {"DSV41_GRAPH_PROBE_ROUTER": "0", "DSV41_GRAPH_ATTN": "0"},
            clear=True,
        ):
            spec = importlib.util.spec_from_file_location(
                "dsv4_graph_probe_pin_off", HELPER
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        self.assertFalse(m.premix_pin_enabled())
        attn_only = _load(flag="0")
        with mock.patch.dict(os.environ, {"DSV41_GRAPH_ATTN": "1"}, clear=True):
            spec = importlib.util.spec_from_file_location(
                "dsv4_graph_probe_pin", HELPER
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        self.assertTrue(m.premix_pin_enabled())


if __name__ == "__main__":
    unittest.main(verbosity=0)
    print("OK")
