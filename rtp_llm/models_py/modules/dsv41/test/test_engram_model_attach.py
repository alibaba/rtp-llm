# SPDX-License-Identifier: Apache-2.0
"""CPU control-flow tests. Factories are fakes; no CUDA/engine claim is made."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from rtp_llm.models_py.modules.dsv41 import engram_bootstrap as b


SOURCE = Path(__file__).resolve().parents[1] / "engram_model_attach.py"
SPEC = importlib.util.spec_from_file_location("engram_model_attach_under_test", SOURCE)
attach = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = attach
SPEC.loader.exec_module(attach)


class Storage:
    def __init__(self, trace, *, close_error=False):
        self.trace = trace
        self.close_error = close_error
        self.pins = 0

    def _pin_graph(self):
        self.trace.append("pin")
        self.pins += 1

    def _unpin_graph(self):
        self.trace.append("unpin")
        self.pins -= 1

    def close(self):
        if self.pins:
            raise RuntimeError("pinned storage")
        self.trace.append("close")
        if self.close_error:
            raise RuntimeError("injected close failure")


class Layer:
    def __init__(self, layer_id, trace, *, release_error=False):
        self.layer_id = layer_id
        self.trace = trace
        self.release_error = release_error

    def release_external_views(self):
        self.trace.append(("release", self.layer_id))
        if self.release_error:
            raise RuntimeError("injected view release failure")


class ModelAttachTest(unittest.TestCase):
    def setUp(self):
        self.trace = []
        self.retained = []
        self.layout = SimpleNamespace(layer_ids=(0, 2))
        self.hash_state = object()
        self.storage = Storage(self.trace)
        self.model = SimpleNamespace(
            _v4_args=SimpleNamespace(v41_config={"checkpoint_path": "/small/checkpoint"}),
            v4=SimpleNamespace(
                embed=SimpleNamespace(weight=SimpleNamespace(device="cuda:1")),
                layers=[SimpleNamespace(engram=None) for _ in range(3)],
            ),
            py_hw_kernel_config=SimpleNamespace(enable_cuda_graph=False),
            _engram_hash_state=None,
            _engram_layers=(),
        )
        self.package = {"staged": True}
        b.stage_package(self.package)
        self.addCleanup(b.clear_staged_package)
        self.env = {}
        self.kwargs = dict(
            layout_factory=self.make_layout,
            hash_factory=self.make_hash,
            layer_factory=self.make_layer,
            attach_factory=self.attach_storage,
            retain=self.retained.append,
            environ=self.env,
        )

    def make_layout(self, config):
        self.trace.append("layout")
        self.assertIs(config, self.model._v4_args.v41_config)
        return self.layout

    def make_hash(self, layout, path, *, device):
        self.trace.append("hash")
        self.assertIs(layout, self.layout)
        self.assertEqual((path, device), ("/small/checkpoint", "cuda:1"))
        return self.hash_state

    def attach_storage(self, package, *, layout, hash_state, device, environ):
        self.trace.append("attach")
        self.attached_environ = environ
        self.assertEqual((package, device), (self.package, "cuda:1"))
        self.assertIs(layout, self.layout)
        self.assertIs(hash_state, self.hash_state)
        return self.storage

    def make_layer(self, config, layer_id, path, device, **kwargs):
        self.trace.append(("layer", layer_id))
        # The first constructed module is not visible while the second loads.
        self.assertTrue(all(layer.engram is None for layer in self.model.v4.layers))
        self.assertEqual(kwargs, {"external_storage": self.storage})
        return Layer(layer_id, self.trace)

    def run_init(self, **overrides):
        return attach.initialize_model_engram(self.model, **{**self.kwargs, **overrides})

    def test_explicit_storage_p_environment_reaches_bootstrap(self):
        environment = {**self.env, "ENGRAM_SHARD_COUNT": "3"}
        self.run_init(environ=environment)
        self.assertIs(self.attached_environ, environment)

    def test_p_without_staged_package_cannot_silently_load_local_tables(self):
        b.clear_staged_package()
        with self.assertRaisesRegex(attach.EngramModelInitializationError, "REQUIRES_EXCHANGE"):
            self.run_init(environ={"ENGRAM_SHARD_COUNT": "3"})
        self.assertEqual(self.trace, ["layout"])

    def test_all_layers_bind_one_storage_after_hash_and_attach(self):
        state = self.run_init()
        self.assertEqual(self.trace, ["layout", "hash", "attach", ("layer", 0), ("layer", 2)])
        self.assertEqual(state.phase, "READY")
        self.assertTrue(state.external)
        self.assertIs(state.storage, self.storage)
        self.assertIs(self.model._engram_storage, self.storage)
        self.assertIs(self.model._engram_hash_state, self.hash_state)
        self.assertEqual(self.model._engram_layers, ((0, 0), (2, 1)))
        self.assertEqual(self.retained, [self.storage])
        self.assertIs(self.model.v4.layers[0].engram, state.layers[0][2])
        self.assertIs(self.model.v4.layers[2].engram, state.layers[1][2])

    def test_successful_reinitialize_ignores_changed_launch_environment(self):
        state = self.run_init()
        old_trace = list(self.trace)
        self.assertIs(self.run_init(), state)
        self.assertEqual(self.trace, old_trace)
        self.assertEqual(self.retained, [self.storage])

    def test_legacy_has_no_external_keyword_or_retention(self):
        b.clear_staged_package()

        def legacy(config, layer_id, path, device, **kwargs):
            self.assertEqual(kwargs, {})
            return Layer(layer_id, self.trace)

        state = self.run_init(environ={}, layer_factory=legacy)
        self.assertFalse(state.external)
        self.assertIsNone(state.storage)
        self.assertNotIn("attach", self.trace)
        self.assertEqual(self.retained, [])

    def test_no_layer_model_skips_bundle_graph_and_hash(self):
        self.layout.layer_ids = ()
        self.model.py_hw_kernel_config.enable_cuda_graph = True
        state = self.run_init(environ={})
        self.assertEqual(state.phase, "EMPTY")
        self.assertFalse(state.external)
        self.assertEqual(self.trace, ["layout"])
        self.assertIs(self.run_init(), state)

    def test_external_graph_config_rejected_before_hash_or_import(self):
        self.model.py_hw_kernel_config.enable_cuda_graph = True
        with self.assertRaisesRegex(attach.EngramModelInitializationError, "GRAPH_UNSUPPORTED"):
            self.run_init()
        self.assertEqual(self.trace, ["layout"])

    def graph_env(self):
        return {**self.env, attach.EXTERNAL_GRAPH_MODE_ENV: attach.PROCESS_LIFETIME_GRAPH}

    def test_decode_graph_opt_in_pins_once_before_ready_and_is_process_retained(self):
        self.model.py_hw_kernel_config.enable_cuda_graph = True
        state = self.run_init(environ=self.graph_env())
        self.assertEqual(state.phase, "READY")
        self.assertEqual(self.storage.pins, 1)
        self.assertTrue(state.graph_pin_held and state.process_retained)
        self.assertEqual(self.retained, [self.storage])
        self.assertIs(self.run_init(environ={}), state)
        self.assertEqual(self.storage.pins, 1)
        attach.require_engram_runtime(self.model, is_cuda_graph=True)
        attach.require_engram_runtime(self.model, capture_check=lambda: self.fail("decode does not poll CUDA"))
        with self.assertRaisesRegex(RuntimeError, "pinned"):
            self.storage.close()
        self.assertNotIn("close", self.trace)

    def test_graph_mode_is_explicit_and_cannot_be_enabled_after_initialization(self):
        self.run_init()
        self.run_init(environ=self.graph_env())
        with self.assertRaisesRegex(attach.EngramModelInitializationError, "GRAPH_UNSUPPORTED"):
            attach.require_engram_runtime(self.model, is_cuda_graph=True)
        self.assertEqual(self.storage.pins, 0)

    def test_invalid_graph_mode_fails_before_import(self):
        with self.assertRaisesRegex(attach.EngramModelInitializationError, "INVALID_EXTERNAL_ENGRAM_GRAPH_MODE"):
            self.run_init(environ={**self.env, attach.EXTERNAL_GRAPH_MODE_ENV: "1"})
        self.assertEqual(self.trace, ["layout"])

    def test_opt_in_only_accepts_decode_graph_but_keeps_prefill_eager(self):
        self.run_init(environ=self.graph_env())
        attach.require_engram_runtime(self.model, is_prefill=True, capture_check=lambda: False)
        for fields in (dict(is_cuda_graph=True, is_prefill=True),
                       dict(is_prefill=True, capture_check=lambda: True),
                       dict(is_cuda_graph=True, is_target_verify=True)):
            with self.subTest(fields=fields), self.assertRaisesRegex(
                attach.EngramModelInitializationError, "DECODE_GRAPH_ONLY"
            ):
                attach.require_engram_runtime(self.model, **fields)

    def test_graph_layer_construction_failure_never_pins_or_publishes(self):
        def fail_second(config, layer_id, path, device, **kwargs):
            if layer_id == 2:
                raise RuntimeError("second layer failed")
            return self.make_layer(config, layer_id, path, device, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "second layer failed"):
            self.run_init(environ=self.graph_env(), layer_factory=fail_second)
        self.assertNotIn("pin", self.trace)
        self.assertEqual(self.trace[-2:], [("release", 0), "close"])
        self.assertTrue(all(layer.engram is None for layer in self.model.v4.layers))
        self.assertEqual(self.retained, [])

    def test_graph_pin_failure_leaves_no_ready_binding(self):
        def fail_pin():
            raise RuntimeError("injected pin failure")
        self.storage._pin_graph = fail_pin
        with self.assertRaisesRegex(RuntimeError, "pin failure"):
            self.run_init(environ=self.graph_env())
        self.assertEqual(self.model._engram_attachment.phase, "FAILED")
        self.assertEqual(self.storage.pins, 0)
        self.assertEqual(self.trace[-3:], [("release", 2), ("release", 0), "close"])
        self.assertEqual(self.retained, [])

    def test_graph_retention_failure_unpins_before_pre_ready_close(self):
        def fail_retain(storage):
            self.assertEqual(storage.pins, 1)
            self.assertEqual(self.model._engram_attachment.phase, "INITIALIZING")
            raise RuntimeError("injected retention failure")
        with self.assertRaisesRegex(RuntimeError, "retention failure"):
            self.run_init(environ=self.graph_env(), retain=fail_retain)
        self.assertEqual(self.storage.pins, 0)
        self.assertEqual(self.trace[-4:], ["unpin", ("release", 2), ("release", 0), "close"])
        self.assertTrue(all(layer.engram is None for layer in self.model.v4.layers))

    def test_failed_graph_rollback_retains_resource_without_unmapping(self):
        def fail_unpin():
            raise RuntimeError("injected unpin failure")
        self.storage._unpin_graph = fail_unpin
        def fail_retain(storage):
            if self.model._engram_attachment.phase == "INITIALIZING":
                raise RuntimeError("injected retention failure")
            self.retained.append(storage)
        with self.assertRaisesRegex(attach.EngramModelInitializationError, "CLEANUP_INCOMPLETE"):
            self.run_init(environ=self.graph_env(), retain=fail_retain)
        self.assertEqual(self.storage.pins, 1)
        self.assertNotIn("close", self.trace)
        self.assertEqual(self.retained, [self.storage])

    def test_missing_lifetime_anchor_rejects_capture_without_releasing_pin(self):
        state = self.run_init(environ=self.graph_env())
        state.process_retained = False
        with self.assertRaisesRegex(attach.EngramModelInitializationError, "LIFETIME_UNPINNED"):
            attach.require_engram_runtime(self.model, is_cuda_graph=True)
        self.assertEqual(self.storage.pins, 1)
        self.assertNotIn("close", self.trace)

    def test_ready_binding_cannot_be_replaced_before_capture_or_reinitialized(self):
        state = self.run_init(environ=self.graph_env())
        self.model.v4.layers[0].engram = object()
        for call in (lambda: attach.require_engram_runtime(self.model, is_cuda_graph=True), self.run_init):
            with self.assertRaisesRegex(attach.EngramModelInitializationError, "BINDING_CHANGED"):
                call()
        self.assertEqual(state.phase, "READY")
        self.assertEqual(self.storage.pins, 1)
        self.assertNotIn("close", self.trace)

    def test_distributed_without_staged_package_is_rejected(self):
        b.clear_staged_package()
        with self.assertRaisesRegex(attach.EngramModelInitializationError, "REQUIRES_EXCHANGE"):
            self.run_init(environ={"ENGRAM_SHARD_COUNT": "2"})
        self.assertEqual(self.trace, ["layout"])

    def test_hash_failure_does_not_import_or_publish(self):
        def bad_hash(*args, **kwargs):
            raise ValueError("tokenizer hash ABI mismatch")

        with self.assertRaisesRegex(ValueError, "tokenizer hash"):
            self.run_init(hash_factory=bad_hash)
        self.assertEqual(self.trace, ["layout"])
        self.assertIsNone(self.model._engram_hash_state)
        self.assertEqual(self.model._engram_layers, ())

    def test_failed_attach_does_not_load_private_tables(self):
        def bad_attach(*args, **kwargs):
            raise ValueError("generation mismatch")

        with self.assertRaisesRegex(ValueError, "generation mismatch"):
            self.run_init(attach_factory=bad_attach)
        self.assertEqual(self.trace, ["layout", "hash"])
        self.assertEqual(self.retained, [])

    def test_layer_failure_releases_views_before_close_without_publishing(self):
        def bad_layer(config, layer_id, path, device, **kwargs):
            if layer_id == 2:
                raise ValueError("missing projection")
            return self.make_layer(config, layer_id, path, device, **kwargs)

        with self.assertRaisesRegex(ValueError, "missing projection"):
            self.run_init(layer_factory=bad_layer)
        self.assertEqual(self.trace[-2:], [("release", 0), "close"])
        self.assertTrue(all(layer.engram is None for layer in self.model.v4.layers))
        self.assertIsNone(self.model._engram_hash_state)
        self.assertEqual(self.model._engram_layers, ())
        self.assertEqual(self.retained, [])
        with self.assertRaisesRegex(attach.EngramModelInitializationError, "NOT_RETRYABLE"):
            self.run_init()

    def test_cleanup_failure_retains_mapping_and_reports_resource(self):
        self.storage.close_error = True

        def bad_layer(*args, **kwargs):
            raise ValueError("projection failure")

        with self.assertRaises(attach.EngramModelInitializationError) as raised:
            self.run_init(layer_factory=bad_layer)
        self.assertEqual(raised.exception.code, "ENGRAM_CLEANUP_INCOMPLETE")
        self.assertIs(raised.exception.resource, self.model._engram_attachment)
        self.assertIn(self.storage, raised.exception.resource.failed_resources)
        self.assertEqual(self.retained, [self.storage])
        self.assertEqual(raised.exception.resource.phase, "FAILED")

    def test_attach_cleanup_pending_resource_is_not_lost(self):
        self.storage.close_error = True

        def bad_attach(*args, **kwargs):
            error = RuntimeError("native import cleanup pending")
            error.resource = self.storage
            raise error

        with self.assertRaisesRegex(attach.EngramModelInitializationError, "CLEANUP_INCOMPLETE"):
            self.run_init(attach_factory=bad_attach)
        self.assertEqual(self.retained, [self.storage])

    def test_publication_failure_rolls_back_every_layer(self):
        class RejectNewModule:
            def __init__(self):
                self._engram = None

            @property
            def engram(self):
                return self._engram

            @engram.setter
            def engram(self, value):
                if isinstance(value, Layer):
                    raise ValueError("injected publication failure")
                self._engram = value

        self.model.v4.layers[2] = RejectNewModule()
        with self.assertRaisesRegex(ValueError, "publication failure"):
            self.run_init()
        self.assertTrue(all(layer.engram is None for layer in self.model.v4.layers))
        self.assertEqual(self.trace[-3:], [("release", 2), ("release", 0), "close"])
        self.assertEqual(self.model._engram_layers, ())
        self.assertEqual(self.retained, [])

    def test_lifetime_anchor_failure_rolls_back_before_forward(self):
        def fail_retain(resource):
            raise RuntimeError("injected retention failure")

        with self.assertRaisesRegex(RuntimeError, "retention failure"):
            self.run_init(retain=fail_retain)
        self.assertTrue(all(layer.engram is None for layer in self.model.v4.layers))
        self.assertEqual(self.trace[-3:], [("release", 2), ("release", 0), "close"])
        self.assertIsNone(self.model._engram_hash_state)

    def test_reentrant_initialization_fails_without_second_import(self):
        def reenter(*args, **kwargs):
            return self.run_init()

        with self.assertRaisesRegex(attach.EngramModelInitializationError, "NOT_RETRYABLE"):
            self.run_init(hash_factory=reenter)
        self.assertEqual(self.trace, ["layout"])
        self.assertEqual(self.model._engram_attachment.phase, "FAILED")

    def test_view_release_failure_keeps_module_and_mapping(self):
        def bad_layer(config, layer_id, path, device, **kwargs):
            if layer_id == 2:
                raise ValueError("projection failure")
            return Layer(layer_id, self.trace, release_error=True)

        with self.assertRaisesRegex(attach.EngramModelInitializationError, "CLEANUP_INCOMPLETE"):
            self.run_init(layer_factory=bad_layer)
        resources = self.model._engram_attachment.failed_resources
        self.assertTrue(any(isinstance(item, Layer) for item in resources))
        self.assertIn(self.storage, resources)
        self.assertEqual(self.retained, [self.storage])

    def test_eager_graph_api_rejected_and_normal_eager_allowed(self):
        self.run_init()
        attach.require_eager_engram(self.model, capture_check=lambda: False)
        with self.assertRaisesRegex(attach.EngramModelInitializationError, "GRAPH_UNSUPPORTED"):
            attach.require_eager_engram(self.model, is_cuda_graph=True)
        with self.assertRaisesRegex(attach.EngramModelInitializationError, "GRAPH_UNSUPPORTED"):
            attach.require_eager_engram(self.model, capture_check=lambda: True)

    def test_legacy_guard_does_not_query_cuda_capture(self):
        self.model._engram_attachment = attach.EngramModelAttachment(phase="READY")
        attach.require_eager_engram(
            self.model, is_cuda_graph=True,
            capture_check=lambda: self.fail("legacy mode must not query CUDA"),
        )

    def test_failed_external_model_cannot_forward(self):
        self.model._engram_attachment = attach.EngramModelAttachment(phase="FAILED", external=True)
        with self.assertRaisesRegex(attach.EngramModelInitializationError, "NOT_READY"):
            attach.require_eager_engram(self.model)

    def test_legacy_success_cannot_be_switched_in_process(self):
        b.clear_staged_package()

        def legacy(*args, **kwargs):
            return Layer(args[1], self.trace)

        state = self.run_init(environ={}, layer_factory=legacy)
        self.assertIs(self.run_init(), state)
        self.assertFalse(state.external)
        self.assertNotIn("attach", self.trace)


if __name__ == "__main__":
    unittest.main()
