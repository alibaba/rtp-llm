"""Sleep replay uses the live factory, preserving storages and ownership."""

import gc
import sys
import unittest
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.model_loader.weight_memory_saver import model_build_scope
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.fp8_fp4_base import (
    Fp8Fp4ExecutorBase,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import weight_reload
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.weight_adapter import (
    adapt_split_moe_weights,
)
from rtp_llm.utils.model_weight import W


def raw_weights(shared=0):
    weights = {
        "routed_gate": torch.full((2, 4, 4), 1, dtype=torch.int8),
        "routed_up": torch.full((2, 4, 4), 3, dtype=torch.int8),
        "routed_gate_scale": torch.full((2, 4, 1), 2, dtype=torch.uint8),
        "routed_up_scale": torch.full((2, 4, 1), 4, dtype=torch.uint8),
        "routed_down": torch.full((2, 8, 2), 5, dtype=torch.int8),
        "routed_down_scale": torch.full((2, 8, 1), 6, dtype=torch.uint8),
        "router": torch.arange(16, dtype=torch.float32).reshape(2, 8),
        "router_bias": torch.ones(2),
        "router_tid2eid": torch.ones(4, 1, dtype=torch.int64),
    }
    if shared:
        weights.update(
            shared_gate_up=torch.ones(8 * shared, 8),
            shared_gate_up_scale=torch.ones(8 * shared, 1),
            shared_down=torch.ones(8, 4 * shared),
            shared_down_scale=torch.ones(8, 1),
        )
    return weights


class WeightReloadTest(unittest.TestCase):
    def test_canonical_packing_allocates_inside_weight_region(self):
        from rtp_llm.model_loader import weight_memory_saver

        active = False
        allocations = []
        allocate = torch.empty

        @contextmanager
        def region():
            nonlocal active
            active = True
            try:
                yield
            finally:
                active = False

        def tracked_empty(*args, **kwargs):
            allocations.append(active)
            return allocate(*args, **kwargs)

        raw = raw_weights()
        with mock.patch.object(
            weight_memory_saver, "weights_region", region
        ), mock.patch.object(torch, "empty", side_effect=tracked_empty):
            adapt_split_moe_weights(dict(raw), 4, 0, {k: k for k in raw})
        self.assertEqual(allocations, [True, True])
        self.assertFalse(active)

    def test_resident_weight_setup_is_pausable_but_runtime_setup_is_not(self):
        from rtp_llm.model_loader import weight_memory_saver
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
            MegaMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
            MegaMoeSEExecutor,
        )

        active = False

        @contextmanager
        def region():
            nonlocal active
            self.assertFalse(active)
            active = True
            try:
                yield
            finally:
                active = False

        for cls in (Fp8Fp4ExecutorBase, MegaMoeExecutor, MegaMoeSEExecutor):
            with self.subTest(executor=cls.__name__):
                obj = cls.__new__(cls)
                torch.nn.Module.__init__(obj)
                events = []
                with mock.patch.object(
                    weight_memory_saver, "weights_region", region
                ), mock.patch.object(
                    obj,
                    "_setup_kernel_weights",
                    side_effect=lambda _: events.append(("weights", active)),
                ), mock.patch.object(
                    obj,
                    "_setup_runtime",
                    create=True,
                    side_effect=lambda: events.append(("runtime", active)),
                ):
                    obj.setup_weights({})
                self.assertEqual(events, [("weights", True), ("runtime", False)])
                self.assertFalse(active)

    def test_copy_validates_all_tensors_before_writing(self):
        for invalid in (torch.zeros(3), torch.zeros(2, dtype=torch.int64)):
            live = {"a": torch.ones(2), "b": torch.ones(2)}
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                weight_reload.copy_tensors_in_place(
                    live, {"a": torch.zeros(2), "b": invalid}
                )
            self.assertTrue(torch.equal(live["a"], torch.ones(2)))
        with self.assertRaisesRegex(RuntimeError, "keys differ"):
            weight_reload.copy_tensors_in_place({"a": torch.ones(2)}, {})

    def test_split_adapter_restores_canonical_aliases_and_popped_inputs(self):
        for shared in (0, 1, 3):
            with self.subTest(shared=shared):
                raw = raw_weights(shared)
                names = {key: key for key in raw}
                canonical = adapt_split_moe_weights(dict(raw), 4, shared, names)
                # The real executor pops these; gate/shared weights may remain
                # in ModelWeights under canonical (not checkpoint) names.
                live = {
                    key: value.clone()
                    for key, value in canonical.items()
                    if key not in (W.moe_w1, W.moe_w2, W.moe_s1, W.moe_s2)
                }
                layer = torch.nn.Module()
                layer.layer_id = 7
                layer.reload_weights = mock.Mock()
                reload = weight_reload.SplitMoeWeightReload(
                    layer, live, names, raw.keys(), 4, shared
                )
                pointers = {k: v.data_ptr() for k, v in live.items()}
                for _ in range(2):
                    for tensor in live.values():
                        tensor.zero_()
                    restored = reload.reload_weights(raw)
                    self.assertEqual(restored, live.keys())
                    passed = layer.reload_weights.call_args.args[0]
                    self.assertEqual(passed.keys(), canonical.keys())
                    for key in live:
                        torch.testing.assert_close(live[key], canonical[key])
                        self.assertEqual(live[key].data_ptr(), pointers[key])
                    torch.testing.assert_close(passed[W.moe_w1], canonical[W.moe_w1])
                missing = dict(raw)
                missing.pop("router")
                with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
                    reload.reload_weights(missing)
                self.assertEqual(layer.reload_weights.call_count, 2)

    def test_registry_is_scoped_weak_and_disabled_without_sleep(self):
        with mock.patch.object(weight_reload, "_RELOADERS", weakref.WeakSet()):
            for enabled in (False, True):
                with mock.patch.object(
                    weight_reload, "sleep_enabled", return_value=enabled
                ):
                    target, draft = torch.nn.Module(), torch.nn.Module()
                    target.layer_id = draft.layer_id = 0
                    for scope, layer in (("target", target), ("draft", draft)):
                        with model_build_scope(scope):
                            weight_reload.register_split_moe_reload(
                                layer, {}, {}, (), 4, 0
                            )
                    del layer
                    if not enabled:
                        self.assertEqual(weight_reload.iter_moe_reloaders(), [])
                        continue
                    self.assertEqual(
                        {
                            x._sleep_model_scope
                            for x in weight_reload.iter_moe_reloaders()
                        },
                        {"target", "draft"},
                    )
                    target_ref = weakref.ref(target)
                    del target
                    gc.collect()
                    self.assertIsNone(target_ref())
                    self.assertEqual(len(weight_reload.iter_moe_reloaders()), 1)
                    del draft
                    gc.collect()
                    self.assertEqual(weight_reload.iter_moe_reloaders(), [])

    def test_registry_failure_is_not_swallowed(self):
        layer = torch.nn.Module()
        layer.layer_id = 0
        with mock.patch.object(
            weight_reload, "sleep_enabled", return_value=True
        ), mock.patch.object(weight_reload, "_RELOADERS") as registry:
            registry.add.side_effect = RuntimeError("registration failed")
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                weight_reload.register_split_moe_reload(layer, {}, {}, (), 4, 0)

    def test_executor_reload_does_not_run_constructor_or_runtime_setup(self):
        class Executor(Fp8Fp4ExecutorBase):
            def __init__(self, *args, **kwargs):
                raise AssertionError("constructor must not run during reload")

            def setup_weights(self, weights):
                raise AssertionError("runtime setup must not run during reload")

            def _setup_kernel_weights(self, weights):
                self.kernel = weights["raw"].clone() * self.cfg.factor

            def sleep_weight_tensors(self):
                return {"kernel": self.kernel}

        executor = Executor.__new__(Executor)
        torch.nn.Module.__init__(executor)
        executor.cfg = SimpleNamespace(factor=2)
        executor.kernel = torch.zeros(2, 3)
        alias = executor.kernel[:, 1:]
        ptr = executor.kernel.data_ptr()
        for value in (3, 5):
            executor.reload_weights({"raw": torch.full((2, 3), float(value))})
            self.assertEqual(ptr, executor.kernel.data_ptr())
            torch.testing.assert_close(alias, torch.full((2, 2), float(2 * value)))
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            executor.reload_weights({"raw": torch.ones(3, 3)})
        torch.testing.assert_close(executor.kernel, torch.full((2, 3), 10.0))

    def test_mega_and_mega_se_restore_all_kernel_buffers_in_place(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
            mega_moe,
            mega_moe_se,
        )

        def transform(l1, l2):
            return tuple(tuple(t.clone() for t in pair) for pair in (l1, l2))

        # Only hardware transforms are replaced; real layout ownership, shared
        # expert validation, and the production reload method all execute.
        dg = SimpleNamespace(transform_weights_for_mega_moe=transform)
        for shared in (0, 1, 3):
            with self.subTest(shared=shared):
                module = mega_moe_se if shared else mega_moe
                cls = module.MegaMoeSEExecutor if shared else module.MegaMoeExecutor
                obj = cls.__new__(cls)
                torch.nn.Module.__init__(obj)
                obj.cfg = SimpleNamespace(
                    n_local_experts=2,
                    dim=8,
                    moe_inter_dim=4,
                    moe_w1_layout="gate_up",
                    n_shared_experts=shared,
                )
                raw = raw_weights(shared)
                if shared:
                    for name in ("shared_gate_up", "shared_down"):
                        raw[name] = raw[name].to(torch.float8_e4m3fn)
                    for name in ("shared_gate_up_scale", "shared_down_scale"):
                        raw[name] = raw[name].to(torch.int32)
                weights = adapt_split_moe_weights(
                    dict(raw), 4, shared, {k: k for k in raw}
                )
                with mock.patch.dict(sys.modules, {"deep_gemm": dg}), mock.patch.object(
                    module,
                    "prepare_fp4_weight_scale_for_deepgemm",
                    side_effect=lambda tensor, *args: tensor.to(torch.int32),
                ), mock.patch.object(torch.cuda, "empty_cache"), mock.patch.object(
                    cls, "setup_weights", side_effect=AssertionError("runtime setup")
                ):
                    obj._setup_kernel_weights(dict(weights))
                    expected = {
                        k: v.clone() for k, v in obj.sleep_weight_tensors().items()
                    }
                    ptrs = {
                        k: v.data_ptr() for k, v in obj.sleep_weight_tensors().items()
                    }
                    self.assertEqual(len(expected), 8 if shared else 4)
                    for _ in range(2):
                        for tensor in obj.sleep_weight_tensors().values():
                            tensor.zero_()
                        obj.reload_weights(weights)
                        for name, tensor in obj.sleep_weight_tensors().items():
                            self.assertTrue(
                                torch.equal(
                                    tensor.view(torch.uint8),
                                    expected[name].view(torch.uint8),
                                ),
                                name,
                            )
                            self.assertEqual(tensor.data_ptr(), ptrs[name])


if __name__ == "__main__":
    unittest.main()
