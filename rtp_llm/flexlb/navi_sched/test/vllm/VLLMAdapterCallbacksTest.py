"""Device-free tests for NaviSched callbacks delegated by native vLLM."""

import logging
import os
import sys
import types
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

ADAPTER = Path(__file__).resolve().parents[2] / "adapters/vllm"


def module(name, file):
    loaded = types.ModuleType(name)
    loaded.__package__ = name.rpartition(".")[0]
    sys.modules[name] = loaded
    exec(compile(file.read_text(), str(file), "exec"), loaded.__dict__)
    return loaded


logger_module = types.ModuleType("vllm.logger")
logger_module.init_logger = logging.getLogger
sys.modules["vllm.logger"] = logger_module
core_callbacks = module("review_core", ADAPTER / "VLLMCoreCallbacks.py")
worker_callbacks = module("review_worker", ADAPTER / "VLLMWorkerDiagnostics.py")
scheduler_callbacks = module("review_scheduler", ADAPTER / "VLLMSchedulerObserver.py")
cache_callbacks = module("review_cache", ADAPTER / "VLLMCacheCallbacks.py")
iteration_callbacks = module("review_metrics", ADAPTER / "VLLMIterationMetrics.py")


class ReviewTests(unittest.TestCase):

    def test_registry_factories_and_duplicate_rejection(self):
        reg = module("review_registry_test", ADAPTER / "VLLMPluginRegistry.py")
        transport = object()
        self.assertIs(reg.create_engine_transport(None, transport), transport)
        self.assertIs(reg.create_renderer_transport(transport), transport)
        self.assertIsNone(reg.create_engine_observer(None, None, 1))
        self.assertFalse(reg.is_frontend_request_intercepted(None, None))
        fn = lambda cfg, core: ("wrapped", core)
        reg.register_engine_transport_factory(fn)
        reg.register_engine_transport_factory(fn)
        self.assertEqual(
            reg.create_engine_transport(None, transport), ("wrapped", transport)
        )
        with self.assertRaises(RuntimeError):
            reg.register_engine_transport_factory(lambda *_: None)
        plugins = types.SimpleNamespace(
            get_engine_adapter=lambda: None,
            register_engine_adapter=lambda a: setattr(plugins, "registered", a),
        )
        reg.install(plugins)
        self.assertIs(plugins.registered, reg)
        plugins.get_engine_adapter = lambda: object()
        with self.assertRaises(RuntimeError):
            reg.install(plugins)

    def test_sample_result_identity_and_original_exception(self):

        class Observer(core_callbacks.VLLMCoreCallbacks):

            def observe_model_output(self, a, b):
                self.observed = (a, b)

        observer = Observer()
        future = Future()
        result = object()
        executor = types.SimpleNamespace(sample_tokens=lambda *a, **kw: result)
        self.assertIs(observer.sample_tokens(executor, future, "grammar"), result)
        self.assertEqual(observer.observed, (future, result))
        failure = ValueError("sampling")

        def fail(*a, **kw):
            raise failure

        executor.sample_tokens = fail
        with self.assertRaises(ValueError) as raised:
            observer.sample_tokens(executor, future, None, True)
        self.assertIs(raised.exception, failure)
        self.assertIs(observer.observed[1].exception(), failure)

    def test_sample_observation_failure_is_isolated(self):

        class Observer(core_callbacks.VLLMCoreCallbacks):

            def observe_model_output(self, *a):
                raise RuntimeError("diagnostic only")

        result = object()
        executor = types.SimpleNamespace(sample_tokens=lambda *a, **kw: result)
        with self.assertLogs("vllm.v1.engine.core", level="ERROR"):
            self.assertIs(Observer().sample_tokens(executor, Future(), None), result)

    def test_wait_details_are_captured_after_completion(self):
        observer = core_callbacks.VLLMCoreCallbacks()
        seen = []
        observer.log_core_stage = lambda *a, **kw: seen.append((a, kw))
        output = types.SimpleNamespace(
            num_scheduled_tokens={"r": 1}, total_num_scheduled_tokens=1
        )
        queue = [1]
        result = object()

        def complete():
            queue.append(2)
            output.total_num_scheduled_tokens = 7
            return result

        future = types.SimpleNamespace(result=complete)
        self.assertIs(observer.wait_for_output(object(), future, output, queue), result)
        self.assertEqual(seen[0][0][1], "batch_queue_future_result")
        self.assertEqual(seen[0][1]["batch_queue_size"], 2)
        self.assertEqual(seen[0][1]["scheduled_tokens"], 7)
        failure = RuntimeError("execution")

        def fail():
            raise failure

        future.result = fail
        with self.assertRaises(RuntimeError) as raised:
            observer.wait_for_output(None, future, output)
        self.assertIs(raised.exception, failure)
        self.assertEqual(len(seen), 2)

    def test_no_diagnostic_failure_can_abort_core(self):
        obs = core_callbacks.VLLMCoreCallbacks()

        def fail(*a, **kw):
            raise ValueError("diagnostic")

        obs._report_core_step_stage = fail
        obs._log_slow_engine_stage = fail
        obs.log_core_stage(
            types.SimpleNamespace(has_work=lambda: True), "engine_step", 0, 2000000000
        )

    def test_worker_exception_does_not_turn_into_success(self):
        d = worker_callbacks.VLLMWorkerDiagnostics(types.SimpleNamespace(rank=0))
        failure = ValueError("rpc failed")

        def fail():
            raise failure

        with self.assertRaises(ValueError) as raised:
            d.execute(fail, (), {})
        self.assertIs(raised.exception, failure)

    def test_iteration_prefill_decode_failed_and_empty(self):
        obs = scheduler_callbacks.VLLMSchedulerObserver(lambda e: None)
        obs.start_iteration()
        self.assertIsNone(obs.iteration_tokens)
        req = types.SimpleNamespace(
            request_id="a", num_output_tokens=0, num_prompt_tokens=100
        )
        obs.before_request_output(req)
        req.num_prompt_tokens = 300
        req.num_output_tokens = 1
        obs.record_output(req, [10])
        self.assertEqual(obs.iteration_tokens, 101)
        req.num_output_tokens = 5
        obs.before_request_output(req)
        obs.record_output(req, [11, 12])
        self.assertEqual(obs.iteration_tokens, 103)
        obs.record_failed_output(
            types.SimpleNamespace(num_output_tokens=0, num_prompt_tokens=50)
        )
        self.assertEqual(obs.iteration_tokens, 153)
        obs.start_iteration()
        obs.before_request_output(req)
        obs.record_output(req, [])
        self.assertEqual(obs.iteration_tokens, 0)

    def test_iteration_source_and_missing_values(self):
        for enabled, scheduler, iteration, expected in [
            (
                True,
                None,
                types.SimpleNamespace(num_prompt_tokens=1, num_generation_tokens=2),
                [],
            ),
            (True, types.SimpleNamespace(num_iteration_tokens=7), None, [7]),
            (
                False,
                None,
                types.SimpleNamespace(num_prompt_tokens=1, num_generation_tokens=2),
                [3],
            ),
            (False, None, None, []),
        ]:
            seen = []
            logger = types.SimpleNamespace(
                engine_adapter=types.SimpleNamespace(
                    scheduler_iteration_metrics_enabled=lambda: enabled
                ),
                histogram_iteration_tokens={
                    0: types.SimpleNamespace(observe=seen.append)
                },
            )
            iteration_callbacks.record_iteration_tokens(logger, scheduler, iteration, 0)
            self.assertEqual(seen, expected)

    def test_cache_event_order_nulls_and_no_stale_buffer(self):
        events = []
        observer = cache_callbacks.VLLMCacheObserver(lambda *args: events.append(args))
        blocks = [
            types.SimpleNamespace(is_null=False),
            types.SimpleNamespace(is_null=True),
            types.SimpleNamespace(is_null=False),
        ]
        observer.stored(blocks, [1, 2, 3], 4, 55)
        observer.stored([], [], 4, 56)
        observer.cleared()
        self.assertEqual(
            events,
            [
                ("stored", 4, (1, 3), "55", "request_store"),
                ("cleared", None, (), None, "prefix_cache_reset"),
            ],
        )


class AdditionalBoundaryTests(unittest.TestCase):

    def test_whale_defaults_and_explicit_values(self):
        env = module("review_whale_env", ADAPTER / "VLLMWhaleEnvironment.py")
        defaults = env.build_environment({}, "7001", "hippo", lambda: "1.2.3.4")
        self.assertEqual(defaults["VLLM_NAVI_SCHED_SERVER_PORT"], "7001")
        self.assertEqual(defaults["VLLM_NAVI_SCHED_AGENT_GIG_PORT"], "7005")
        self.assertEqual(defaults["VLLM_NAVI_SCHED_BIZ_NAME"], "hippo")
        explicit = {
            "VLLM_NAVI_SCHED_SERVER_IP": "5.6.7.8",
            "VLLM_NAVI_SCHED_TARGET_BIZ_NAME": "target",
            "VLLM_NAVI_SCHED_AGENT_GIG_PORT": "9999",
        }

        def unexpected():
            raise AssertionError("must not resolve explicit IP")

        values = env.build_environment(explicit, "7001", "hippo", unexpected)
        self.assertEqual(values["VLLM_NAVI_SCHED_BIZ_NAME"], "target")
        self.assertEqual(values["VLLM_NAVI_SCHED_SERVER_IP"], "5.6.7.8")
        self.assertEqual(values["VLLM_NAVI_SCHED_AGENT_GIG_PORT"], "9999")
        self.assertEqual(values["PYTHONHASHSEED"], "0")

    def test_plugin_registration_selected_and_nonselected_ranks(self):
        import os

        for selected in (True, False):
            package_name = "review_package_" + str(selected)
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package
            reg = module(
                package_name + ".VLLMPluginRegistry", ADAPTER / "VLLMPluginRegistry.py"
            )
            sys.modules[package_name + ".VLLMWorkerDiagnostics"] = worker_callbacks
            sys.modules[package_name + ".VLLMIterationMetrics"] = iteration_callbacks
            expected = object()
            for modname, fn in [
                ("VLLMDeferredRenderer", "create_deferred_renderer"),
                ("VLLMEngineObserver", "create_vllm_engine_observer"),
                ("VLLMEngineTransport", "create_vllm_engine_transport"),
                ("VLLMNativePreparation", "configure_native_preparation"),
            ]:
                stub = types.ModuleType(package_name + "." + modname)
                setattr(stub, fn, lambda *a: expected)
                sys.modules[stub.__name__] = stub
            state = {"provider": None}
            plugins = types.SimpleNamespace(
                get_engine_adapter=lambda: state["provider"],
                register_engine_adapter=lambda p: state.update(provider=p),
            )
            vllm = types.ModuleType("vllm")
            vllm.plugins = plugins
            with patch.dict(
                sys.modules, {"vllm": vllm, "vllm.plugins": plugins}
            ), patch.dict(
                os.environ,
                {
                    "PYTHONHASHSEED": "0",
                    "NAVI_SCHED_RANK_LISTEN": "0",
                    "RANK_ID": "0" if selected else "1",
                    "RANK_SIZE": "2",
                },
                clear=True,
            ):
                entry = module(
                    package_name + ".NaviSchedVLLMPlugin",
                    ADAPTER / "NaviSchedVLLMPlugin.py",
                )
                entry.register_enabled_vllm_plugin()
                self.assertIs(state["provider"], reg)
                self.assertIsInstance(
                    reg.create_worker_diagnostics(types.SimpleNamespace(rank=0)),
                    worker_callbacks.VLLMWorkerDiagnostics,
                )
                if selected:
                    self.assertIs(reg.create_engine_observer(None, None, 1), expected)
                    self.assertTrue(reg.scheduler_iteration_metrics_enabled())
                else:
                    self.assertIsNone(reg.create_engine_observer(None, None, 1))
                    self.assertFalse(reg.scheduler_iteration_metrics_enabled())


class CacheTouchCompatibilityTest(unittest.TestCase):
    def test_two_and_three_field_lookup_keep_touch_free_order(self):
        import ast
        import threading
        from collections import deque

        tree = ast.parse((ADAPTER / "VLLMEngineObserver.py").read_text())
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_promote_arrived_cache_touches"
        )
        namespace = {
            "time": types.SimpleNamespace(monotonic_ns=lambda: 100),
            "logger": logging.getLogger(__name__),
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), "cache touch", "exec"),
            namespace,
        )
        for field_count in (2, 3):
            with self.subTest(field_count=field_count):
                events = []
                request = object()
                blocks = [object(), object()]
                cached = types.SimpleNamespace(blocks=[[blocks[0]], [blocks[1]]])

                def lookup(actual):
                    self.assertIs(actual, request)
                    return (cached,) + (0,) * (field_count - 1)

                pool = types.SimpleNamespace(
                    touch=lambda value: events.append(("touch", value)),
                    free_blocks=lambda value: events.append(("free", value)),
                )
                observer = types.SimpleNamespace(
                    _native_ingress=False,
                    _cache_touch_lock=threading.Lock(),
                    _cache_touch_queue=deque([(1, request, 1)]),
                    _cache_touch_count=0,
                    _cache_touch_promoted_blocks=0,
                    _cache_touch_last_log_ns=100,
                )
                scheduler = types.SimpleNamespace(
                    kv_cache_manager=types.SimpleNamespace(
                        get_computed_blocks=lookup, block_pool=pool
                    )
                )
                namespace["_promote_arrived_cache_touches"](observer, scheduler)
                self.assertEqual(events, [("touch", blocks), ("free", blocks)])
                self.assertEqual(observer._cache_touch_promoted_blocks, 2)
                self.assertFalse(observer._cache_touch_queue)


if __name__ == "__main__":
    unittest.main()
