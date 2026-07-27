import ast
import gc
import sys
import unittest
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _install_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


# deepep_wrapper only needs these classes for annotations/config extraction. Stub
# them so this lifecycle test does not import the full GPU model package.
_install_module("rtp_llm.config", __path__=[])
_install_module("rtp_llm.config.engine_config", EngineConfig=object)
_install_module("rtp_llm.config.model_config", ModelConfig=object)
_install_module("rtp_llm.config.quant_config", QuantizationConfig=object)
_install_module("rtp_llm.device", __path__=[])
_install_module(
    "rtp_llm.device.device_type",
    DeviceType=SimpleNamespace(ROCm="rocm"),
    get_device_type=lambda: "cuda",
)
_install_module("rtp_llm.models_py.modules", __path__=[])
_install_module("rtp_llm.models_py.modules.factory", __path__=[])
_install_module("rtp_llm.models_py.modules.factory.fused_moe", __path__=[])
_install_module("rtp_llm.models_py.modules.factory.fused_moe.defs", __path__=[])
_install_module(
    "rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter",
    MoEConfigAdapter=object,
)
_install_module("rtp_llm.ops", SpeculativeType=SimpleNamespace(NONE=0))

from rtp_llm.models_py.distributed import deepep_wrapper as dw


class _Config:
    def equal(self, other):
        return self is other


class _Buffer:
    def __init__(self, name):
        self.name = name
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1


class _FailingBuffer(_Buffer):
    def destroy(self):
        super().destroy()
        raise RuntimeError(f"destroy failed for {self.name}")


class DeepEPLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.original_instance = dw.DeepEPWrapper._instance
        self.original_initialized = dw.DeepEPWrapper._initialized
        dw.DeepEPWrapper._instance = None
        dw.DeepEPWrapper._initialized = False

    def tearDown(self):
        dw.DeepEPWrapper._instance = self.original_instance
        dw.DeepEPWrapper._initialized = self.original_initialized

    def _active_wrapper(self, buffer=None):
        wrapper = object.__new__(dw.DeepEPWrapper)
        wrapper._config = _Config()
        wrapper._mode = dw.DeepEPMode.LOW_LATENCY
        wrapper._buffer = _Buffer("initial") if buffer is None else buffer
        wrapper._retired_buffer_ref = None
        wrapper._retired_buffer_id = None
        wrapper._lifecycle_error = None
        dw.DeepEPWrapper._instance = wrapper
        dw.DeepEPWrapper._initialized = True
        return wrapper

    def test_destroy_releases_buffer_and_is_idempotent(self):
        wrapper = self._active_wrapper()
        buffer = wrapper.buffer

        dw.destroy_deepep_wrapper()
        dw.destroy_deepep_wrapper()

        self.assertEqual(buffer.destroy_calls, 1)
        self.assertIsNone(wrapper._buffer)
        self.assertFalse(dw.DeepEPWrapper._initialized)

    def test_destroy_drops_the_wrapper_owned_reference(self):
        wrapper = self._active_wrapper()
        buffer_ref = weakref.ref(wrapper.buffer)

        dw.destroy_deepep_wrapper()
        gc.collect()

        self.assertIsNone(buffer_ref())

    def test_destroy_failure_is_fail_closed(self):
        buffer = _FailingBuffer("partial")
        buffer_ref = weakref.ref(buffer)
        wrapper = self._active_wrapper(buffer)

        with self.assertRaisesRegex(RuntimeError, "destroy failed for partial"):
            dw.destroy_deepep_wrapper()

        self.assertEqual(buffer.destroy_calls, 1)
        self.assertIsNone(wrapper._buffer)
        self.assertFalse(dw.DeepEPWrapper._initialized)
        with self.assertRaisesRegex(RuntimeError, "previously failed"):
            dw.destroy_deepep_wrapper()
        self.assertEqual(buffer.destroy_calls, 1)
        with self.assertRaisesRegex(RuntimeError, "failed teardown"):
            dw.rebuild_deepep_wrapper()
        with patch.object(dw.torch.distributed, "is_initialized", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "failed teardown"):
                dw.DeepEPWrapper.create(wrapper._config)
        del buffer
        gc.collect()
        self.assertIsNone(buffer_ref())

    def test_missing_destroy_support_is_fail_closed(self):
        wrapper = self._active_wrapper(SimpleNamespace())

        with self.assertRaisesRegex(RuntimeError, "does not expose destroy"):
            dw.destroy_deepep_wrapper()

        self.assertIsNone(wrapper._buffer)
        self.assertFalse(dw.DeepEPWrapper._initialized)

    def test_three_cycles_preserve_wrapper_and_router_identity(self):
        wrapper = self._active_wrapper()
        rebuilt = [
            _Buffer("rebuild-1"),
            _Buffer("rebuild-2"),
            _Buffer("rebuild-3"),
        ]

        with patch.object(
            dw.torch.distributed, "is_initialized", return_value=True
        ), patch.object(
            wrapper,
            "_init_deepep_buffer",
            side_effect=[(dw.DeepEPMode.LOW_LATENCY, buffer) for buffer in rebuilt],
        ):
            for expected in rebuilt:
                old_buffer = wrapper.buffer
                dw.destroy_deepep_wrapper()
                dw.destroy_deepep_wrapper()
                self.assertIs(dw.DeepEPWrapper._instance, wrapper)
                self.assertEqual(old_buffer.destroy_calls, 1)
                with self.assertRaisesRegex(RuntimeError, "suspended"):
                    _ = wrapper.buffer

                dw.rebuild_deepep_wrapper()
                dw.rebuild_deepep_wrapper()
                self.assertIs(dw.DeepEPWrapper.get_instance(wrapper._config), wrapper)
                self.assertIs(wrapper.buffer, expected)

    def test_low_latency_router_dynamically_dereferences_wrapper_buffer(self):
        router_path = (
            Path(__file__).resolve().parents[2]
            / "modules/factory/fused_moe/impl/cuda/routers/deepep_low_latency_router.py"
        )
        tree = ast.parse(router_path.read_text())
        router_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DeepEpLowLatencyRouter"
        )
        buffer_property = next(
            node
            for node in router_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_buffer"
        )
        returned = buffer_property.body[-1].value
        self.assertIsInstance(returned, ast.Attribute)
        self.assertEqual(returned.attr, "buffer")
        self.assertIsInstance(returned.value, ast.Attribute)
        self.assertEqual(returned.value.attr, "_deepep_buffer_wrapper")

    def test_rebuild_is_noop_when_startup_skipped_initialization(self):
        with patch.object(dw.torch.distributed, "is_initialized", return_value=False):
            dw.rebuild_deepep_wrapper()
        self.assertIsNone(dw.DeepEPWrapper._instance)
        self.assertFalse(dw.DeepEPWrapper._initialized)

    def test_rebuild_inconsistent_state_fails_closed(self):
        dw.DeepEPWrapper._initialized = True

        with self.assertRaisesRegex(RuntimeError, "initialized without an instance"):
            dw.rebuild_deepep_wrapper()

        self.assertFalse(dw.DeepEPWrapper._initialized)

    def test_rebuild_requires_distributed_environment(self):
        wrapper = self._active_wrapper()
        dw.destroy_deepep_wrapper()
        with patch.object(dw.torch.distributed, "is_initialized", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "distributed environment"):
                dw.rebuild_deepep_wrapper()
        self.assertIs(dw.DeepEPWrapper._instance, wrapper)

    def test_rebuild_failure_remains_suspended(self):
        wrapper = self._active_wrapper()
        dw.destroy_deepep_wrapper()

        with patch.object(
            dw.torch.distributed, "is_initialized", return_value=True
        ), patch.object(
            wrapper,
            "_init_deepep_buffer",
            side_effect=RuntimeError("rebuild failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "rebuild failed"):
                dw.rebuild_deepep_wrapper()

        self.assertIsNone(wrapper._buffer)
        self.assertFalse(dw.DeepEPWrapper._initialized)

    def test_rebuild_rejects_the_destroyed_buffer(self):
        wrapper = self._active_wrapper()
        old_buffer = wrapper.buffer
        dw.destroy_deepep_wrapper()

        with patch.object(
            dw.torch.distributed, "is_initialized", return_value=True
        ), patch.object(
            wrapper,
            "_init_deepep_buffer",
            return_value=(dw.DeepEPMode.LOW_LATENCY, old_buffer),
        ):
            with self.assertRaisesRegex(RuntimeError, "previously destroyed"):
                dw.rebuild_deepep_wrapper()

        self.assertEqual(old_buffer.destroy_calls, 1)
        self.assertIsNone(wrapper._buffer)
        self.assertFalse(dw.DeepEPWrapper._initialized)

    def test_rebuild_mode_mismatch_destroys_candidate(self):
        wrapper = self._active_wrapper()
        dw.destroy_deepep_wrapper()
        candidate = _Buffer("wrong-mode")

        with patch.object(
            dw.torch.distributed, "is_initialized", return_value=True
        ), patch.object(
            wrapper,
            "_init_deepep_buffer",
            return_value=(dw.DeepEPMode.NORMAL, candidate),
        ):
            with self.assertRaisesRegex(RuntimeError, "mode changed"):
                dw.rebuild_deepep_wrapper()

        self.assertEqual(candidate.destroy_calls, 1)
        self.assertIsNone(wrapper._buffer)
        self.assertFalse(dw.DeepEPWrapper._initialized)

    def test_rebuild_mode_mismatch_cleanup_failure_is_fail_closed(self):
        wrapper = self._active_wrapper()
        dw.destroy_deepep_wrapper()
        candidate = _FailingBuffer("wrong-mode")

        with patch.object(
            dw.torch.distributed, "is_initialized", return_value=True
        ), patch.object(
            wrapper,
            "_init_deepep_buffer",
            return_value=(dw.DeepEPMode.NORMAL, candidate),
        ):
            with self.assertRaisesRegex(RuntimeError, "rejected buffer"):
                dw.rebuild_deepep_wrapper()

        self.assertEqual(candidate.destroy_calls, 1)
        self.assertIsNone(wrapper._buffer)
        self.assertFalse(dw.DeepEPWrapper._initialized)
        with self.assertRaisesRegex(RuntimeError, "failed teardown"):
            dw.rebuild_deepep_wrapper()

    def test_normal_buffer_requests_explicit_destroy_support(self):
        wrapper = object.__new__(dw.DeepEPWrapper)
        wrapper._config = SimpleNamespace(
            use_deepep_internode=False,
            deep_ep_num_sm=8,
            expert_num=8,
            ep_size=2,
        )
        wrapper._use_accl_ep = False
        captured = {}

        def make_buffer(**kwargs):
            captured.update(kwargs)
            return object()

        with patch.object(dw, "DeepEPBuffer", side_effect=make_buffer):
            wrapper._init_normal_buffer(object())
        self.assertTrue(captured["explicitly_destroy"])

    def test_low_latency_buffers_request_explicit_destroy_support(self):
        wrapper = object.__new__(dw.DeepEPWrapper)
        wrapper._config = SimpleNamespace(
            local_rank=1,
            ll_num_max_token_per_rank=64,
            hidden_size=128,
            ep_size=2,
            expert_num=8,
            attention_dp_size=1,
            attention_tp_size=1,
            ffn_dp_size=1,
            ffn_tp_size=2,
        )
        wrapper._use_accl_ep = False

        with patch.object(dw, "DeepEPBuffer") as buffer_type:
            buffer_type.get_low_latency_rdma_size_hint.return_value = 1024
            buffer_type.get_low_latency_rdma_size_hint_m2n.return_value = 2048

            wrapper._init_low_latency_buffer(object())
            self.assertTrue(buffer_type.call_args.kwargs["explicitly_destroy"])

            buffer_type.reset_mock()
            buffer_type.get_low_latency_rdma_size_hint_m2n.return_value = 2048
            wrapper._init_low_latency_m2n_buffer(object())
            self.assertTrue(buffer_type.call_args.kwargs["explicitly_destroy"])


if __name__ == "__main__":
    unittest.main()
