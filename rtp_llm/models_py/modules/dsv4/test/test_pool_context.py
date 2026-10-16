"""CPU-only lifetime and Module-registration contracts for temporary pool views."""

import gc
import importlib.util
import sys
import unittest
import weakref
from pathlib import Path
from unittest import mock

import torch
from torch import nn


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).parents[1]
_ATTN = _load("pool_attn_type_under_test", _ROOT / "attn_type.py")
with mock.patch.dict(sys.modules, {"rtp_llm.models_py.modules.dsv4.attn_type": _ATTN}):
    pool = _load("pool_context_under_test", _ROOT / "fp8/_kv_cache_utils.py")


class PoolContextTest(unittest.TestCase):
    def bind(self, module, state_3d=False):
        kv = torch.randn(2, 4, 132)
        table = torch.tensor([[0, 1]], dtype=torch.int32)
        state = torch.randn(2, 4, 128) if state_3d else torch.randn(8, 128)
        module.set_pool_context(kv, table, 4, state, table, 4, 256, 512, 1024)
        self.assertIs(module._kv_pool_view, kv)
        self.assertIs(module._kv_block_table, table)
        self.assertEqual(module._state_pool_3d.data_ptr(), state.data_ptr())
        self.assertEqual(module._state_pool_3d.shape, (2, 4, 128))
        self.assertEqual(module._kv_owner_tokens_per_block, 1024)
        return kv, table, state

    def assert_cleared(self, module):
        for name, expected in pool._EMPTY_POOL_CONTEXT.items():
            self.assertEqual(getattr(module, name), expected, name)

    def test_rebind_clear_lifetimes_and_unrelated_module_state(self):
        module = pool.PoolBackedModule()
        module.weight = nn.Parameter(torch.ones(3))
        module.register_buffer("persistent", torch.ones(5))
        child = module.child = nn.Linear(3, 2)
        expected = {k: v.clone() for k, v in module.state_dict().items()}
        for state_3d in (False, True):
            values = self.bind(module, state_3d)
            refs = [weakref.ref(t) for t in values]
            del values
            gc.collect()
            self.assertTrue(all(r() is not None for r in refs))
            module.clear_pool_context()
            gc.collect()
            self.assertTrue(all(r() is None for r in refs))
            self.assert_cleared(module)
            module.clear_pool_context()
            self.assert_cleared(module)
        self.assertIs(module.child, child)
        for name, value in module.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]))

    def test_registered_parameter_and_buffer_keep_registration(self):
        for field in (
            "_kv_pool_view",
            "_kv_block_table",
            "_state_pool_3d",
            "_state_block_table",
        ):
            for kind in ("parameter", "buffer"):
                module = pool.PoolBackedModule()
                if kind == "parameter":
                    setattr(module, field, nn.Parameter(torch.ones(2)))
                    registry = module._parameters
                else:
                    delattr(module, field)
                    module.register_buffer(field, torch.ones(2))
                    registry = module._buffers
                module.clear_pool_context()
                self.assertIn(field, registry)
                self.assertIsNone(registry[field])
                self.assertNotIn(field, module.__dict__)
                self.assert_cleared(module)

    def test_registered_scalar_field_preserves_original_type_error(self):
        module = pool.PoolBackedModule()
        del module._kv_eb
        module.register_buffer("_kv_eb", torch.ones(1))
        with self.assertRaises(TypeError):
            module.clear_pool_context()
        self.assertIs(module._kv_eb, module._buffers["_kv_eb"])

    def test_custom_setter_and_missing_field_keep_original_assignment_path(self):
        class TrackingPool(pool.PoolBackedModule):
            def __setattr__(self, name, value):
                if name in pool._EMPTY_POOL_CONTEXT and "assigned" in self.__dict__:
                    self.assigned.append(name)
                super().__setattr__(name, value)

        module = TrackingPool()
        module.assigned = []
        self.bind(module)
        module.assigned.clear()
        module.clear_pool_context()
        self.assertEqual(module.assigned, list(pool._EMPTY_POOL_CONTEXT))
        self.assert_cleared(module)
        normal = pool.PoolBackedModule()
        del normal._kv_pool_view
        normal.clear_pool_context()
        self.assert_cleared(normal)

    def test_binding_parameters_and_buffers_preserves_registration(self):
        for field, index in (
            ("_kv_pool_view", 0),
            ("_kv_block_table", 1),
            ("_state_pool_3d", 3),
            ("_state_block_table", 4),
        ):
            for constructor, registry in (
                (nn.Parameter, "_parameters"),
                (nn.Buffer, "_buffers"),
            ):
                module = pool.PoolBackedModule()
                args = [
                    torch.randn(2, 4, 132),
                    torch.randn(1, 2),
                    4,
                    torch.randn(2, 4, 128),
                    torch.randn(1, 2),
                    4,
                    256,
                    512,
                ]
                value = args[index] = constructor(args[index])
                if constructor is nn.Buffer:
                    # Module rejects automatic buffer registration when an
                    # ordinary attribute of this name already exists.
                    with self.assertRaisesRegex(KeyError, "already exists"):
                        module.set_pool_context(*args)
                    delattr(module, field)
                module.set_pool_context(*args)
                self.assertIs(getattr(module, field), value)
                self.assertIs(getattr(module, registry)[field], value)
                self.assertNotIn(field, module.__dict__)
                module.clear_pool_context()
                self.assertIsNone(getattr(module, registry)[field])

    def test_rebinding_registered_fields_and_custom_setter(self):
        module = pool.PoolBackedModule()
        del module._kv_pool_view
        module.register_buffer("_kv_pool_view", torch.ones(2))
        kv, _, _ = self.bind(module)
        self.assertIs(module._buffers["_kv_pool_view"], kv)
        module.clear_pool_context()
        self.assertIsNone(module._buffers["_kv_pool_view"])

        class TrackingPool(pool.PoolBackedModule):
            def __setattr__(self, name, value):
                if name in pool._EMPTY_POOL_CONTEXT and "assigned" in self.__dict__:
                    self.assigned.append(name)
                super().__setattr__(name, value)

        module = TrackingPool()
        module.assigned = []
        self.bind(module)
        self.assertEqual(module.assigned, list(pool._EMPTY_POOL_CONTEXT))

    def test_binding_existing_parameter_retains_type_error_and_partial_state(self):
        module = pool.PoolBackedModule()
        state = module._state_pool_3d = nn.Parameter(torch.empty(2, 4, 128))
        kv = torch.empty(2, 4, 132)
        with self.assertRaisesRegex(TypeError, "cannot assign"):
            module.set_pool_context(kv, None, 4, torch.empty(8, 128), None, 4, 256, 512)
        self.assertIs(module._kv_pool_view, kv)
        self.assertIs(module._state_pool_3d, state)
        module.clear_pool_context()
        self.assertIsNone(module._parameters["_state_pool_3d"])

    def test_binding_validation_retains_partial_state_and_original_errors(self):
        module = pool.PoolBackedModule()
        old_kv, _, _ = self.bind(module)
        new_kv = torch.randn(2, 4, 132)
        old_state = module._state_pool_3d
        # KV validation happens before any assignments.
        with self.assertRaisesRegex(AssertionError, "KV pool bound"):
            module.set_pool_context(new_kv, None, 0, None, None, 0, 0, 512)
        self.assertIs(module._kv_pool_view, old_kv)
        # State validation happens after installing the new KV fields.
        for state, eb, coverage, message in (
            (torch.empty(8, 128), 0, 256, "state pool bound"),
            (torch.empty(7, 128), 4, 256, "not divisible"),
            (torch.empty(2, 3, 128), 4, 256, "does not match"),
            (torch.empty(7), 4, 256, "expected 2D or 3D"),
        ):
            with self.assertRaisesRegex(AssertionError, message):
                module.set_pool_context(new_kv, None, 4, state, None, eb, coverage, 512)
            self.assertIs(module._kv_pool_view, new_kv)
            self.assertIs(module._state_pool_3d, old_state)
        module.clear_pool_context()
        module.set_pool_context(None, None, 0, None, None, 0, 0, 0)
        self.assert_cleared(module)

    def test_tensor_subclass_and_scalar_subclass_follow_module_path(self):
        class PoolTensor(torch.Tensor):
            pass

        class Entries(int):
            pass

        module = pool.PoolBackedModule()
        state = torch.empty(8, 128).as_subclass(PoolTensor)
        module.set_pool_context(
            None, None, Entries(4), state, None, Entries(4), 256, 512
        )
        self.assertIsInstance(module._state_pool_3d, PoolTensor)
        self.assertIsInstance(module._state_eb, Entries)
        self.assertEqual(module._state_pool_3d.shape, (2, 4, 128))
        self.assertEqual(module._state_pool_3d.data_ptr(), state.data_ptr())
        module.clear_pool_context()
        self.assert_cleared(module)

    def test_failed_binding_can_always_be_cleared(self):
        module = pool.PoolBackedModule()
        with self.assertRaises(AssertionError):
            module.set_pool_context(
                torch.empty(2, 4, 132), None, 4, torch.empty(7, 128), None, 4, 256, 512
            )
        module.clear_pool_context()
        self.assert_cleared(module)
        self.bind(module)
        module.clear_pool_context()
        self.assert_cleared(module)


if __name__ == "__main__":
    unittest.main()
