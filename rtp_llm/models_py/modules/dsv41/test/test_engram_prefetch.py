"""CPU stream-contract tests; no CUDA context or model checkpoint required."""

import ast
import importlib.util
import os
import sys
import unittest
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

_spec = importlib.util.spec_from_file_location(
    "engram_prefetch_test", Path(__file__).resolve().parents[1] / "engram.py"
)
engram = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engram)

_kv_spec = importlib.util.spec_from_file_location(
    "engram_test_kv_cache_utils",
    Path(__file__).resolve().parents[1] / "kv_cache_utils.py",
)
kv_cache_utils = importlib.util.module_from_spec(_kv_spec)
_kv_spec.loader.exec_module(kv_cache_utils)


class Tensor:
    def __init__(self, rows=16, heads=24, *, contiguous=True):
        self.shape = (rows, heads)
        self.ndim = 2
        self.device = torch.device("cuda:0")
        self.dtype = torch.int64
        self.is_cuda = True
        self._version = 0
        self._stride = (heads, 1) if contiguous else (heads * 2, 2)
        self.record_stream = Mock()

    def stride(self):
        return self._stride

    def storage_offset(self):
        return 0

    def is_contiguous(self):
        return self._stride == (self.shape[1], 1)

    def is_inference(self):
        return False


class EngramPrefetchTest(unittest.TestCase):
    def setUp(self):
        # These tests explicitly exercise the optional GPU prefetch path.
        context = patch.dict(os.environ, {"DSV41_CPU_ENGRAM_LOOKUP": "0"})
        context.start()
        self.addCleanup(context.stop)
        self.current, self.side, self.event = Mock(), Mock(), Mock()
        self.rows = Tensor()
        self.embedding = Mock(return_value=self.rows)
        self.embedding._uva = (object(), object())
        self.embedding._pinned = SimpleNamespace(_lookup_events=[])
        self.module = SimpleNamespace(
            q_weight=Tensor(),
            layout=SimpleNamespace(n_hash_cols=24),
            embed_tokens=self.embedding,
            _lookup_stream=self.side,
            _lookup_done=self.event,
            _lookup_work=None,
        )
        self.module.prefetch_lookup = lambda ids: engram.Engram.prefetch_lookup(
            self.module, ids
        )
        self.module.clear_lookup_prefetch = lambda: engram.Engram.clear_lookup_prefetch(
            self.module
        )
        for name, value in (
            ("current_stream", Mock(return_value=self.current)),
            ("stream", lambda stream: nullcontext()),
            ("is_current_stream_capturing", Mock(return_value=False)),
        ):
            context = patch.object(torch.cuda, name, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.modules.dsv41": SimpleNamespace(
                    _profiler=SimpleNamespace(
                        record_function_range=lambda name: nullcontext()
                    )
                )
            },
        )
        context.start()
        self.addCleanup(context.stop)

    def test_producer_and_consumer_dependencies_without_host_wait(self):
        ids = Tensor()
        work = self.module.prefetch_lookup(ids)
        self.side.wait_stream.assert_called_once_with(self.current)
        ids.record_stream.assert_called_once_with(self.side)
        self.event.record.assert_called_once_with(self.side)
        self.embedding.assert_called_once_with(ids, ids.device, prefetch=True)
        self.assertIs(work.consume(ids, Tensor()), self.rows)
        self.current.wait_event.assert_called_once_with(self.event)
        self.rows.record_stream.assert_called_once_with(self.current)
        work.clear()
        self.event.synchronize.assert_not_called()
        self.assertIsNone(work.ids)
        self.assertIsNone(work.embedding)

    def test_pending_owns_inputs_and_table_until_abandoned_work_finishes(self):
        ids = Tensor()
        work = self.module.prefetch_lookup(ids)
        reference = weakref.ref(ids)
        del ids
        self.assertIsNotNone(reference())
        # The mocked embedding call itself retains arguments; remove that owner.
        self.embedding.reset_mock()
        self.module.clear_lookup_prefetch()
        self.event.synchronize.assert_called_once()
        self.assertIsNone(reference())
        self.assertIsNone(work.rows)
        self.assertIsNone(work.embedding)
        self.module.clear_lookup_prefetch()
        self.event.synchronize.assert_called_once()

    def test_mismatch_and_inplace_edit_discard_instead_of_using_stale_rows(self):
        for mismatch in ("identity", "version", "stride", "rows", "device"):
            with self.subTest(mismatch=mismatch):
                ids, hidden = Tensor(), Tensor()
                work = self.module.prefetch_lookup(ids)
                consume_ids = ids
                if mismatch == "identity":
                    consume_ids = Tensor()
                elif mismatch == "version":
                    ids._version += 1
                elif mismatch == "stride":
                    ids._stride = (48, 2)
                elif mismatch == "rows":
                    hidden.shape = (8, 24)
                else:
                    hidden.device = torch.device("cuda:1")
                self.assertIsNone(work.consume(consume_ids, hidden))
                self.module.clear_lookup_prefetch()
        self.assertEqual(self.event.synchronize.call_count, 5)
        self.current.wait_event.assert_not_called()

    def test_unsupported_shapes_and_cold_capture_do_not_launch(self):
        cases = [Tensor(0), Tensor(32769), Tensor(heads=23), Tensor(contiguous=False)]
        cpu = Tensor()
        cpu.is_cuda = False
        cases.append(cpu)
        int32 = Tensor()
        int32.dtype = torch.int32
        cases.append(int32)
        for ids in cases:
            self.assertIsNone(self.module.prefetch_lookup(ids))
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.assertIsNone(self.module.prefetch_lookup(Tensor()))
        self.module._lookup_stream = None
        self.assertIsNone(self.module.prefetch_lookup(Tensor()))
        self.embedding.assert_not_called()
        self.side.wait_stream.assert_not_called()

    def test_exact_chunk_boundary_and_duplicate_prefetch(self):
        ids = Tensor(32768)
        self.assertIsNotNone(self.module.prefetch_lookup(ids))
        self.assertIsNone(self.module.prefetch_lookup(ids))
        self.embedding.assert_called_once()
        self.module.clear_lookup_prefetch()

    def test_launch_error_is_drained_and_propagated(self):
        self.embedding.side_effect = RuntimeError("launch failed")
        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            self.module.prefetch_lookup(Tensor())
        self.side.synchronize.assert_called_once()
        self.assertIsNone(self.module._lookup_work)

    def test_event_record_error_keeps_table_alive_until_stream_is_drained(self):
        self.event.record.side_effect = RuntimeError("event failed")
        with self.assertRaisesRegex(RuntimeError, "event failed"):
            self.module.prefetch_lookup(Tensor())
        self.side.synchronize.assert_called_once()
        self.assertIsNone(self.module._lookup_work)

    def test_warmup_records_lazy_event_and_registers_host_owner_once(self):
        self.module._lookup_stream = None
        with (
            patch.dict(os.environ, {"DSV41_ASYNC_ENGRAM_LOOKUP": "1"}),
            patch.object(torch.cuda, "Stream", return_value=self.side) as stream,
            patch.object(torch.cuda, "Event", return_value=self.event),
        ):
            engram.Engram.prepare_lookup_prefetch(self.module)
            engram.Engram.prepare_lookup_prefetch(self.module)
        stream.assert_called_once_with(device=torch.device("cuda:0"), priority=0)
        self.event.record.assert_called_once_with(self.side)
        self.current.wait_event.assert_called_once_with(self.event)
        self.assertEqual(self.embedding._pinned._lookup_events, [self.event])

    def test_unregister_waits_before_releasing_mapped_host_storage(self):
        order = []
        event = SimpleNamespace(synchronize=lambda: order.append("wait"))
        runtime = SimpleNamespace(
            cudaHostUnregister=lambda pointer: (order.append("unregister") or 0,)
        )
        engram._PinnedSharedTable._unregister(runtime, torch.empty(1), [event])
        self.assertEqual(order, ["wait", "unregister"])

    def _layer(self, module=None):
        return SimpleNamespace(engram=module or self.module, engram_hashes=Tensor())

    def test_first_layer_and_two_layer_lookahead(self):
        first = self._layer()
        second_module = SimpleNamespace(prefetch_lookup=Mock(return_value=None))
        plan = engram.EngramLookupPrefetch(
            [(1, first), (14, self._layer(second_module))]
        )
        plan.before_layer(0)
        self.embedding.assert_called_once()
        plan.before_layer(1)
        second_module.prefetch_lookup.assert_not_called()
        self.module._lookup_work.consume(first.engram_hashes, Tensor())
        self.module._lookup_work = None
        for index in range(2, 12):
            plan.before_layer(index)
        second_module.prefetch_lookup.assert_not_called()
        plan.before_layer(12)
        second_module.prefetch_lookup.assert_called_once()
        plan.clear()
        self.event.synchronize.assert_not_called()

    def test_adjacent_layers_wait_for_pending_consumer(self):
        first = self._layer()
        second_module = SimpleNamespace(prefetch_lookup=Mock(return_value=None))
        plan = engram.EngramLookupPrefetch(
            [(1, first), (2, self._layer(second_module))]
        )
        plan.before_layer(0)
        plan.before_layer(1)
        second_module.prefetch_lookup.assert_not_called()
        self.module._lookup_work.consume(first.engram_hashes, Tensor())
        self.module._lookup_work = None
        plan.before_layer(2)
        second_module.prefetch_lookup.assert_called_once()

    def test_skipped_consumer_drains_and_ced_layers_stay_synchronous(self):
        later = SimpleNamespace(prefetch_lookup=Mock(return_value=None))
        plan = engram.EngramLookupPrefetch(
            [(1, self._layer()), (20, self._layer(later)), (21, self._layer(later))]
        )
        plan.before_layer(0)
        plan.before_layer(12)
        self.event.synchronize.assert_called_once()
        plan.before_layer(21)
        later.prefetch_lookup.assert_not_called()
        plan.clear()

    def test_request_end_clears_pending_and_same_ids_are_looked_up_again(self):
        layer = self._layer()
        ids_reference = weakref.ref(layer.engram_hashes)
        for _ in range(2):
            plan = engram.EngramLookupPrefetch([(1, layer)])
            plan.before_layer(0)
            plan.clear()
            self.assertIsNone(self.module._lookup_work)
            self.assertEqual(plan.layers, ())
        self.assertEqual(self.embedding.call_count, 2)
        self.assertEqual(self.event.synchronize.call_count, 2)
        self.embedding.reset_mock()
        layer.engram_hashes = None
        self.assertIsNone(ids_reference())

    def test_inference_and_regular_tensor_versions(self):
        regular = torch.zeros(1)
        self.assertEqual(engram._tensor_version(regular), 0)
        regular.add_(1)
        self.assertEqual(engram._tensor_version(regular), 1)
        with torch.inference_mode():
            self.assertIsNone(engram._tensor_version(torch.zeros(1)))

    def test_prefetch_helper_is_separate_from_default_sync_lookup(self):
        embedding = engram.HostEngramEmbedding(
            torch.empty(8, 32, dtype=torch.float8_e4m3fn),
            torch.ones(8, 1, dtype=torch.uint8),
        )
        embedding._uva = (object(), object())
        embedding._num_sms = 148
        kernels = SimpleNamespace(lookup_host_rows=Mock(), lookup_prefetch_rows=Mock())
        package = sys.modules["rtp_llm.models_py.modules.dsv41"]
        ids = Tensor()
        with patch.object(package, "_engram_triton", kernels, create=True):
            self.assertIs(
                embedding(ids, ids.device), kernels.lookup_host_rows.return_value
            )
            self.assertIs(
                embedding(ids, ids.device, prefetch=True),
                kernels.lookup_prefetch_rows.return_value,
            )
        kernels.lookup_host_rows.assert_called_once_with(*embedding._uva, ids, 148)
        kernels.lookup_prefetch_rows.assert_called_once_with(*embedding._uva, ids, 148)

    def _carveout_fixture(self):
        path = Path(__file__).resolve().parents[1] / "_engram_triton.py"
        tree = ast.parse(path.read_text())
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_prepare_prefetch_kernel", "_lookup_host_rows")
        ]
        namespace = {
            "PREFETCH_LOOKUP_CARVEOUT": 100,
            "_PREFETCH_LOOKUP_PREPARED": set(),
        }
        exec(compile(ast.Module(functions, []), str(path), "exec"), namespace)
        driver = SimpleNamespace(
            CUfunction=lambda value: value,
            CUfunction_attribute=SimpleNamespace(
                CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT=9
            ),
            CUresult=SimpleNamespace(CUDA_SUCCESS=0),
            cuFuncSetAttribute=Mock(return_value=(0,)),
            cuFuncGetAttribute=Mock(return_value=(0, 100)),
        )
        context = patch.dict(
            sys.modules, {"cuda.bindings": SimpleNamespace(driver=driver)}
        )
        context.start()
        self.addCleanup(context.stop)
        return namespace, driver

    def test_carveout_configuration_is_exercised_before_warmup_returns(self):
        namespace, driver = self._carveout_fixture()
        events = []
        kernel = Mock(function=12)
        output = object()

        class Launcher:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    events.append("launch")
                    return kernel

                return launch

        driver.cuFuncSetAttribute.side_effect = lambda *args: (
            events.append("set") or 0,
        )
        driver.cuFuncGetAttribute.side_effect = lambda *args: (
            events.append("read") or 0,
            100,
        )
        namespace.update(
            torch=SimpleNamespace(empty=lambda *a, **kw: output, bfloat16=object()),
            triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
            _lookup_host_kernel=Launcher(),
        )
        ids = SimpleNamespace(
            shape=(16, 24), device="cuda:0", stride=lambda dim: (24, 1)[dim]
        )
        args = (SimpleNamespace(shape=(100, 256)), object(), ids, 148, 4, 4)
        launch = namespace["_lookup_host_rows"]
        self.assertIs(launch(*args, prepare=True), output)
        self.assertEqual(events, ["launch", "set", "read", "launch"])
        events.clear()
        launch(*args, prepare=True)
        self.assertEqual(events, ["launch"])
        events.clear()
        launch(*args)
        self.assertEqual(events, ["launch"])

    def test_carveout_prepares_each_compiled_function_once(self):
        namespace, driver = self._carveout_fixture()
        prepare = namespace["_prepare_prefetch_kernel"]
        first, second = Mock(function=12), Mock(function=13)
        self.assertTrue(prepare(first))
        self.assertFalse(prepare(first))
        driver.cuFuncSetAttribute.assert_called_once_with(12, 9, 100)
        driver.cuFuncGetAttribute.assert_called_once_with(9, 12)
        prepare(second)
        self.assertEqual(driver.cuFuncSetAttribute.call_count, 2)
        self.assertEqual(namespace["_PREFETCH_LOOKUP_PREPARED"], {first, second})

    def test_carveout_failures_propagate_and_do_not_memoize(self):
        namespace, driver = self._carveout_fixture()
        prepare = namespace["_prepare_prefetch_kernel"]
        kernel = Mock(function=12)
        for set_result, get_result, expected in (
            ((1,), (0, 100), "setup failed"),
            ((0,), (1, 100), "verification failed"),
            ((0,), (0, -1), "verification failed"),
        ):
            with self.subTest(set_result=set_result, get_result=get_result):
                driver.cuFuncSetAttribute.return_value = set_result
                driver.cuFuncGetAttribute.return_value = get_result
                with self.assertRaisesRegex(RuntimeError, expected):
                    prepare(kernel)
                self.assertFalse(namespace["_PREFETCH_LOOKUP_PREPARED"])
        driver.cuFuncSetAttribute.return_value = (0,)
        driver.cuFuncGetAttribute.return_value = (0, 100)
        prepare(kernel)
        self.assertEqual(namespace["_PREFETCH_LOOKUP_PREPARED"], {kernel})

    def _model(self):
        class Base(torch.nn.Module):
            def forward(self, inputs, fmha_impl=None):
                return self.parent_forward(inputs)

        base_name = "rtp_llm.models_py.model_desc.deepseek_v41_base_model"
        module_spec = importlib.util.spec_from_file_location(
            "engram_model_prefetch_test",
            Path(__file__).resolve().parents[3] / "model_desc/deepseek_v41_model.py",
        )
        model_module = importlib.util.module_from_spec(module_spec)
        with patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.modules.dsv41.kv_cache_utils": kv_cache_utils,
                base_name: SimpleNamespace(
                    DeepSeekV4Model=Base,
                    _is_decode_fmha=lambda value: value == "decode",
                ),
            },
        ):
            module_spec.loader.exec_module(model_module)
        model = model_module.DeepSeekV41Model.__new__(model_module.DeepSeekV41Model)
        torch.nn.Module.__init__(model)
        self.layer = self._layer()
        model.v4 = SimpleNamespace(
            layers=[None, self.layer], embed=SimpleNamespace(weight=Tensor())
        )
        model.kv_cache = object()
        model._engram_forward_active = False
        model._is_decode_role = False
        model._engram_layers = ((1, 0),)
        model._image_plan = object()
        model._prepare_engram = Mock()
        model._prepare_image_features = Mock()
        model.parent_forward = Mock(return_value="output")
        context = patch.dict(
            sys.modules, {"rtp_llm.models_py.modules.dsv41.engram": engram}
        )
        context.start()
        self.addCleanup(context.stop)
        context = patch.dict(os.environ, {"DSV41_ASYNC_ENGRAM_LOOKUP": "1"})
        context.start()
        self.addCleanup(context.stop)
        inputs = SimpleNamespace(
            attention_inputs=SimpleNamespace(is_prefill=True, is_target_verify=False)
        )
        return model, inputs

    def test_model_prepare_exception_cleans_metadata_without_launching_lookup(self):
        model, inputs = self._model()
        model._prepare_image_features.side_effect = RuntimeError("image prepare failed")
        with self.assertRaisesRegex(RuntimeError, "image prepare failed"):
            model(inputs)
        self.embedding.assert_not_called()
        self.event.synchronize.assert_not_called()
        self.assertIsNone(self.layer.engram_hashes)
        self.assertIsNone(self.layer.engram_token_mask)
        self.assertIsNone(model._image_plan)
        self.assertIsNone(model.v4._engram_lookup_prefetch)
        self.assertFalse(model._engram_forward_active)
        model.parent_forward.assert_not_called()

    def test_model_layer_exception_cleans_lookup_started_after_preparation(self):
        model, inputs = self._model()

        def fail_at_layer(inputs):
            model._prepare_image_features.assert_called_once()
            self.embedding.assert_not_called()
            model.v4._engram_lookup_prefetch.before_layer(0)
            self.assertIsNotNone(self.module._lookup_work)
            raise RuntimeError("layer failed")

        model.parent_forward.side_effect = fail_at_layer
        with self.assertRaisesRegex(RuntimeError, "layer failed"):
            model(inputs)
        self.event.synchronize.assert_called_once()
        self.assertIsNone(self.layer.engram_hashes)
        self.assertIsNone(self.module._lookup_work)
        self.assertFalse(model._engram_forward_active)

    def test_model_error_after_consume_does_not_add_cpu_wait(self):
        model, inputs = self._model()

        def consume_then_fail(inputs):
            model.v4._engram_lookup_prefetch.before_layer(0)
            work = self.module._lookup_work
            self.module._lookup_work = None
            work.consume(self.layer.engram_hashes, Tensor())
            raise RuntimeError("WKV failed")

        model.parent_forward.side_effect = consume_then_fail
        with self.assertRaisesRegex(RuntimeError, "WKV failed"):
            model(inputs)
        self.event.synchronize.assert_not_called()
        self.rows.record_stream.assert_called_once_with(self.current)
        self.assertIsNone(self.layer.engram_hashes)
        self.assertFalse(model._engram_forward_active)

    def test_model_early_return_before_layers_does_not_launch_lookup(self):
        model, inputs = self._model()
        self.assertEqual(model(inputs), "output")
        self.embedding.assert_not_called()
        self.event.synchronize.assert_not_called()
        self.assertIsNone(self.module._lookup_work)

    def test_model_tagged_attention_inputs_start_prefetch(self):
        model, inputs = self._model()
        inputs.attention_inputs = {"decoder_swa_kv": inputs.attention_inputs}

        def run_layer(inputs):
            prefetch = model.v4._engram_lookup_prefetch
            self.assertIsNotNone(prefetch)
            prefetch.before_layer(0)
            return "output"

        model.parent_forward.side_effect = run_layer
        self.assertEqual(model(inputs), "output")
        self.embedding.assert_called_once()
        self.event.synchronize.assert_called_once()

    def test_model_decode_verify_capture_and_disabled_flag_use_original_path(self):
        model, inputs = self._model()
        for fallback in ("decode", "verify", "fmha", "capture", "disabled", "cpu"):
            with self.subTest(fallback=fallback):
                inputs.attention_inputs.is_prefill = fallback != "decode"
                inputs.attention_inputs.is_target_verify = fallback == "verify"
                model.v4.embed.weight.is_cuda = fallback != "cpu"
                with (
                    patch.object(
                        torch.cuda,
                        "is_current_stream_capturing",
                        return_value=fallback == "capture",
                    ),
                    patch.dict(
                        os.environ,
                        {
                            "DSV41_ASYNC_ENGRAM_LOOKUP": (
                                "0" if fallback == "disabled" else "1"
                            )
                        },
                    ),
                ):
                    self.assertEqual(
                        model(inputs, "decode" if fallback == "fmha" else None),
                        "output",
                    )
        self.embedding.assert_not_called()

    def test_reentrant_forward_fails_before_overwriting_outer_metadata(self):
        model, inputs = self._model()
        model._engram_forward_active = True
        outer_hashes = self.layer.engram_hashes
        with self.assertRaisesRegex(RuntimeError, "not reentrant"):
            model(inputs)
        model._prepare_engram.assert_not_called()
        self.assertIs(self.layer.engram_hashes, outer_hashes)
        self.assertTrue(model._engram_forward_active)

    def test_cpu_mode_submits_before_image_prepare_and_cleans_prepare_failure(self):
        model, inputs = self._model()
        model._v4_args = SimpleNamespace(v41_config={})
        inputs.engram_token_windows_host = torch.zeros(16, 4, dtype=torch.int32)
        plan = Mock()
        model._prepare_image_features.side_effect = RuntimeError("image failed")
        with (
            patch.dict(os.environ, {"DSV41_CPU_ENGRAM_LOOKUP": "1"}),
            patch.object(
                engram, "EngramCPULookupPrefetch", return_value=plan
            ) as prepare,
        ):
            with self.assertRaisesRegex(RuntimeError, "image failed"):
                model(inputs)
        prepare.assert_called_once_with(model, inputs.engram_token_windows_host, 129264)
        plan.clear.assert_called_once()
        self.assertFalse(model._engram_forward_active)
        self.assertIsNone(self.layer.engram_hashes)

    def test_default_cpu_mode_only_runs_for_ordinary_prefill(self):
        model, inputs = self._model()
        model._v4_args = SimpleNamespace(v41_config={})
        inputs.engram_token_windows_host = torch.zeros(16, 4, dtype=torch.int32)
        for case in ("prefill", "decode_role", "decode", "verify", "fmha", "capture"):
            with self.subTest(case=case):
                model._is_decode_role = case == "decode_role"
                inputs.attention_inputs.is_prefill = case != "decode"
                inputs.attention_inputs.is_target_verify = case == "verify"
                plan = Mock()
                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch.object(
                        torch.cuda,
                        "is_current_stream_capturing",
                        return_value=case == "capture",
                    ),
                    patch.object(
                        engram, "EngramCPULookupPrefetch", return_value=plan
                    ) as prepare,
                ):
                    self.assertEqual(
                        model(inputs, "decode" if case == "fmha" else None),
                        "output",
                    )
                if case == "prefill":
                    prepare.assert_called_once_with(
                        model, inputs.engram_token_windows_host, 129264
                    )
                    plan.clear.assert_called_once()
                else:
                    prepare.assert_not_called()

    def test_explicit_cpu_flag_cannot_enable_decode_role_prefetch(self):
        model, inputs = self._model()
        model._is_decode_role = True
        with (
            patch.dict(os.environ, {"DSV41_CPU_ENGRAM_LOOKUP": "1"}),
            patch.object(engram, "EngramCPULookupPrefetch") as prepare,
        ):
            self.assertEqual(model(inputs), "output")
        prepare.assert_not_called()


class EngramLookupDefaultsTest(unittest.TestCase):
    def test_role_defaults_and_explicit_overrides_do_not_mutate_environment(self):
        for decode in (False, True):
            for override in (None, "0", "1"):
                with self.subTest(decode=decode, override=override):
                    env = (
                        {}
                        if override is None
                        else {"DSV41_CPU_ENGRAM_LOOKUP": override}
                    )
                    with patch.dict(os.environ, env, clear=True):
                        self.assertEqual(
                            engram.engram_cpu_lookup_enabled(is_decode_role=decode),
                            not decode if override is None else override == "1",
                        )
                        self.assertFalse(
                            engram.engram_lookup_prefetch_enabled(is_decode_role=decode)
                        )
                        self.assertEqual(dict(os.environ), env)

    def test_cpu_runtime_initializes_once_for_prefill_and_never_for_decode(self):
        for decode in (False, True):
            for override in (None, "0", "1"):
                with self.subTest(decode=decode, override=override):
                    env = (
                        {}
                        if override is None
                        else {"DSV41_CPU_ENGRAM_LOOKUP": override}
                    )
                    model = SimpleNamespace(
                        _is_decode_role=decode, _engram_cpu_runtime=None
                    )
                    runtime = SimpleNamespace(capacity=16, slots={})
                    with (
                        patch.dict(os.environ, env, clear=True),
                        patch.object(
                            engram, "EngramCPULookupRuntime", return_value=runtime
                        ) as create,
                    ):
                        engram.prepare_engram_cpu_lookup(model, max_m=16)
                        engram.prepare_engram_cpu_lookup(model, max_m=16)
                    if not decode and override != "0":
                        create.assert_called_once_with(model, 16)
                        self.assertIs(model._engram_cpu_runtime, runtime)
                    else:
                        create.assert_not_called()
                        self.assertIsNone(model._engram_cpu_runtime)


class EngramCPUPrefetchTest(unittest.TestCase):
    def setUp(self):
        self.current = Mock()
        context = patch.object(torch.cuda, "current_stream", return_value=self.current)
        context.start()
        self.addCleanup(context.stop)
        layout = engram.EngramLayout(
            dict(
                engram_layer_ids=[1, 14],
                engram_num_embeddings=[1000, 1000],
                engram_max_ngram_size=4,
                engram_n_heads=2,
                engram_head_dim=32,
                engram_compressed_vocab_size=32,
                engram_pad_token_id=2,
                engram_vocab_size=13,
            )
        )
        self.state = engram.NgramHashState(layout, token_map=list(range(32)))
        context = patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        )
        context.start()
        self.addCleanup(context.stop)
        for name in ("_lazy_init", "init"):
            context = patch.object(
                torch.cuda, name, side_effect=AssertionError("CUDA in CPU test")
            )
            context.start()
            self.addCleanup(context.stop)

    def test_cpu_hash_matches_existing_hash_with_strides_padding_and_image(self):
        for rows in (0, 1, 17, 16384):
            windows = torch.randint(-1, 32, (rows, 8), dtype=torch.int32)[:, ::2]
            expected = self.state.hash_token_windows(windows, windows == 9)
            torch.testing.assert_close(
                engram._hash_token_windows_cpu(self.state, windows, 9), expected
            )

    def test_cpu_hash_out_of_vocab_matches_cuda_padding_contract(self):
        windows = torch.tensor(
            [[32, 2, 1, 0], [9, 33, 4, 3], [7, 6, 2**63 - 1, 4], [-1, -1, -1, -1]],
            dtype=torch.int64,
        )
        sanitized = windows.clone()
        sanitized[sanitized >= 32] = -1
        torch.testing.assert_close(
            engram._hash_token_windows_cpu(self.state, windows, 9),
            self.state.hash_token_windows(sanitized, sanitized == 9),
        )

    def test_cpu_flag_has_priority_over_gpu_flag(self):
        with patch.dict(
            os.environ,
            {"DSV41_CPU_ENGRAM_LOOKUP": "1", "DSV41_ASYNC_ENGRAM_LOOKUP": "1"},
        ):
            self.assertTrue(engram.engram_cpu_lookup_enabled())
            self.assertFalse(engram.engram_lookup_prefetch_enabled())

    def _work(self):
        rows = torch.empty(16, 192, dtype=torch.bfloat16)
        future = Future()
        future.set_result(rows)
        slot = SimpleNamespace(
            device=torch.device("cuda:0"),
            copied=Mock(),
            collect_timing=False,
            last_timing=None,
        )
        ids = Tensor()
        work = engram._EngramCPULookupWork(slot, ids)
        work.future, work.allocation_stream = future, self.current
        return work, ids, rows

    def test_consumer_joins_copy_and_drops_future_without_record_stream(self):
        work, ids, rows = self._work()
        self.assertIs(work.consume(ids, Tensor()), rows)
        self.current.wait_event.assert_called_once_with(work.slot.copied)
        self.assertIsNone(work.future)
        work.clear()
        work.slot.copied.synchronize.assert_not_called()
        ids.record_stream.assert_not_called()

    def test_consumed_output_is_not_retained_until_plan_cleanup(self):
        work, ids, rows = self._work()
        reference = weakref.ref(rows)
        del rows
        consumed = work.consume(ids, Tensor())
        self.assertIsNotNone(reference())
        del consumed
        self.assertIsNone(reference())
        work.clear()

    def test_forward_releases_lookup_rows_before_gate(self):
        work, ids, rows = self._work()
        reference = weakref.ref(rows)
        del rows
        module = SimpleNamespace(
            _cpu_lookup_work=work,
            _lookup_work=None,
            wkv=lambda value: value.new_empty((value.shape[0], 32)),
            q_weight=None,
            k_weight=None,
            eps=1e-20,
        )

        def gate(hidden, *args, **kwargs):
            self.assertIsNone(reference())
            return hidden

        with patch.object(engram, "gated_engram_residual", side_effect=gate):
            hidden = Tensor()
            self.assertIs(
                engram.Engram._forward_rows(module, hidden, ids, None), hidden
            )
        work.clear()

    def test_packed_forward_quantized_releases_rows_and_pair_before_gate(self):
        work, ids, _ = self._work()
        work.future = Future()
        work.future.set_result(torch.empty((16, 24 * 33), dtype=torch.uint8))
        rows_reference = weakref.ref(work.future.result())
        pair_references = []
        order = []

        def unpack(rows, heads, dim):
            self.assertIs(rows, rows_reference())
            self.assertEqual((heads, dim), (24, 32))
            order.append("unpack")
            values = torch.empty((16, 24 * 32), dtype=torch.float8_e4m3fn)
            scales = torch.empty((16, 6), dtype=torch.int32)
            pair_references.extend((weakref.ref(values), weakref.ref(scales)))
            return values, scales

        test = self

        class WKV:
            def __call__(self, rows):
                raise AssertionError("Packed rows must use forward_quantized")

            def forward_quantized(self, values, scales):
                test.assertIs(values, pair_references[0]())
                test.assertIs(scales, pair_references[1]())
                test.assertEqual(values.dtype, torch.float8_e4m3fn)
                test.assertEqual(scales.dtype, torch.int32)
                order.append("quantized_wkv")
                return torch.empty((16, 32), dtype=torch.bfloat16)

        module = SimpleNamespace(
            _cpu_lookup_work=work,
            _lookup_work=None,
            layout=SimpleNamespace(n_hash_cols=24, head_dim=32),
            wkv=WKV(),
            embed_tokens=Mock(side_effect=AssertionError("Unexpected GPU lookup")),
            q_weight=None,
            k_weight=None,
            eps=1e-20,
        )
        mask, output = object(), object()

        def gate(hidden, kv, q, k, eps, token_mask, *, out):
            self.assertIsNone(rows_reference())
            self.assertTrue(all(reference() is None for reference in pair_references))
            self.assertEqual(kv.dtype, torch.bfloat16)
            self.assertIs(token_mask, mask)
            self.assertIs(out, output)
            order.append("gate")
            return out

        with (
            patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.modules.dsv41._engram_packed_triton": SimpleNamespace(
                        unpack_packed_rows=unpack
                    )
                },
            ),
            patch.object(engram, "gated_engram_residual", side_effect=gate),
        ):
            self.assertIs(
                engram.Engram._forward_rows(module, Tensor(), ids, mask, out=output),
                output,
            )
        self.assertEqual(order, ["unpack", "quantized_wkv", "gate"])
        self.assertIsNone(module._cpu_lookup_work)
        self.assertIsNone(work.future)
        self.current.wait_event.assert_called_once_with(work.slot.copied)
        work.clear()
        work.slot.copied.synchronize.assert_not_called()

    def test_unsupported_packed_unpack_falls_back_to_bf16_gpu_lookup(self):
        work, ids, _ = self._work()
        work.future = Future()
        work.future.set_result(torch.empty((16, 24 * 33), dtype=torch.uint8))
        rows_reference = weakref.ref(work.future.result())
        order = []
        hidden, mask, output = Tensor(), object(), object()

        def unpack(rows, heads, dim):
            self.assertIs(rows, rows_reference())
            self.assertEqual((heads, dim), (24, 32))
            order.append("unpack_unsupported")
            return None

        def lookup(hash_ids, device):
            self.assertIsNone(rows_reference())
            self.assertIsNone(module._cpu_lookup_work)
            self.assertIs(hash_ids, ids)
            self.assertEqual(device, hidden.device)
            order.append("gpu_lookup")
            return torch.empty((16, 24, 32), dtype=torch.bfloat16)

        test = self

        class WKV:
            def __call__(self, rows):
                test.assertEqual(rows.dtype, torch.bfloat16)
                test.assertEqual(rows.shape, (16, 24 * 32))
                order.append("bf16_wkv")
                return rows.new_empty((16, 32))

            def forward_quantized(self, values, scales):
                raise AssertionError("Unsupported unpack cannot call quantized WKV")

        module = SimpleNamespace(
            _cpu_lookup_work=work,
            _lookup_work=None,
            layout=SimpleNamespace(n_hash_cols=24, head_dim=32),
            embed_tokens=lookup,
            wkv=WKV(),
            q_weight=None,
            k_weight=None,
            eps=1e-20,
        )
        module._forward_rows = lambda *args, **kwargs: engram.Engram._forward_rows(
            module, *args, **kwargs
        )

        def gate(value, kv, q, k, eps, token_mask, *, out):
            self.assertIs(value, hidden)
            self.assertIsNone(rows_reference())
            self.assertIs(token_mask, mask)
            self.assertIs(out, output)
            order.append("gate")
            return out

        with (
            patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.modules.dsv41._engram_packed_triton": SimpleNamespace(
                        unpack_packed_rows=unpack
                    )
                },
            ),
            patch.object(engram, "gated_engram_residual", side_effect=gate),
        ):
            self.assertIs(module._forward_rows(hidden, ids, mask, out=output), output)
        self.assertEqual(
            order, ["unpack_unsupported", "gpu_lookup", "bf16_wkv", "gate"]
        )
        self.assertIsNone(work.future)
        self.current.wait_event.assert_called_once_with(work.slot.copied)
        work.clear()
        work.slot.copied.synchronize.assert_not_called()

    def test_abandoned_output_remains_owned_until_copy_is_drained(self):
        work, _, rows = self._work()
        reference = weakref.ref(rows)
        del rows
        work.slot.copied.synchronize.side_effect = lambda: self.assertIsNotNone(
            reference()
        )
        work.clear()
        work.slot.copied.synchronize.assert_called_once()
        self.assertIsNone(reference())

    def test_cross_stream_consumer_drains_before_fallback(self):
        work, ids, _ = self._work()
        with patch.object(torch.cuda, "current_stream", return_value=Mock()):
            self.assertIsNone(work.consume(ids, Tensor()))
        work.slot.copied.synchronize.assert_called_once()
        self.current.wait_event.assert_not_called()

    def test_mismatch_and_mutation_drain_before_fallback(self):
        for reason in ("identity", "version", "stride", "rows", "device"):
            work, ids, _ = self._work()
            hidden = Tensor()
            other = ids
            if reason == "identity":
                other = Tensor()
            if reason == "version":
                ids._version += 1
            if reason == "stride":
                ids._stride = (48, 2)
            if reason == "rows":
                hidden.shape = (8, 24)
            if reason == "device":
                hidden.device = torch.device("cuda:1")
            self.assertIsNone(work.consume(other, hidden))
            work.slot.copied.synchronize.assert_called_once()
            self.assertIsNone(work.ids)

    def test_worker_exception_is_propagated_and_future_references_cleared(self):
        work, _, _ = self._work()
        work.future = Future()
        work.future.set_exception(RuntimeError("native failed"))
        with self.assertRaisesRegex(RuntimeError, "native failed"):
            work.clear()
        self.assertIsNone(work.future)
        self.assertIsNone(work.ids)

    def test_wkv_failure_has_joined_copy_and_does_not_pin_output_in_future(self):
        work, ids, _ = self._work()
        module = SimpleNamespace(
            _cpu_lookup_work=work,
            _lookup_work=None,
            wkv=Mock(side_effect=RuntimeError("WKV failed")),
        )
        with self.assertRaisesRegex(RuntimeError, "WKV failed"):
            engram.Engram._forward_rows(module, Tensor(), ids, None)
        self.current.wait_event.assert_called_once_with(work.slot.copied)
        self.assertIsNone(work.future)
        work.clear()
        work.slot.copied.synchronize.assert_not_called()
        self.assertIsNone(module._cpu_lookup_work)

    def test_slot_waits_before_host_overwrite_and_copy_to_new_allocation(self):
        order = []
        slot = engram._EngramCPULookupSlot.__new__(engram._EngramCPULookupSlot)
        slot.device, slot.hash_index = torch.device("cuda:0"), 0
        slot.collect_timing = False
        slot.num_threads = 8
        slot.embedding = SimpleNamespace(weight=object(), scales=object())
        slot.host = torch.empty(4, 192, dtype=torch.bfloat16)
        output = Mock()
        output.shape = (4, 192)
        output.copy_.side_effect = lambda *a, **kw: order.append("copy")
        slot.copied = Mock()
        slot.copied.synchronize.side_effect = lambda: order.append("host_wait")
        slot.copied.record.side_effect = lambda stream: order.append("copied")
        slot.stream, slot.allocated = Mock(), Mock()
        slot.stream.wait_event.side_effect = lambda event: order.append("device_wait")
        hashes = Future()
        hashes.set_result(
            self.state.hash_token_windows(torch.zeros(4, 4, dtype=torch.int32))
        )
        native = Mock(side_effect=lambda *a, **kw: order.append("native"))
        with (
            patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.modules.dsv41.engram_cpu": SimpleNamespace(
                        lookup_host_rows_cpu=native
                    )
                },
            ),
            patch.object(torch.cuda, "device", lambda device: nullcontext()),
            patch.object(torch.cuda, "stream", lambda stream: nullcontext()),
        ):
            self.assertIs(slot.run(hashes, output), output)
        self.assertEqual(
            order, ["host_wait", "native", "device_wait", "copy", "copied"]
        )
        self.assertTrue(native.call_args.args[2].is_contiguous())
        self.assertEqual(native.call_args.kwargs, {"num_threads": 8})
        slot.stream.wait_event.assert_called_once_with(slot.allocated)
        slot.stream.synchronize.assert_not_called()

    def test_exact_request_allocation_and_ready_event_precede_worker_submit(self):
        slot = engram._EngramCPULookupSlot.__new__(engram._EngramCPULookupSlot)
        slot.device, slot.width, slot.capacity = torch.device("cuda:0"), 6144, 32768
        slot.allocated = Mock()
        slot.run = Mock()
        order = []
        slot.allocated.record.side_effect = lambda stream: order.append("allocated")
        executor = Mock()
        allocations = []
        real_empty = torch.empty

        def allocate(shape, **kwargs):
            self.assertEqual(kwargs, dict(dtype=torch.bfloat16, device=slot.device))
            output = real_empty(shape, dtype=kwargs["dtype"], device="meta")
            allocations.append(output)
            order.append("allocate")
            return output

        def submit(fn, hashes, output, timing):
            self.assertEqual(order[-2:], ["allocate", "allocated"])
            order.append("submit")
            future = Future()
            future.set_result(output)
            return future

        executor.submit.side_effect = submit
        with patch.object(torch, "empty", side_effect=allocate):
            for count in (1, 16384, 32768, 7):
                work = engram._EngramCPULookupWork(slot, Tensor(count))
                work.submit(executor, object(), count, None)
                self.assertEqual(work.future.result().shape, (count, 6144))
                self.assertIs(work.allocation_stream, self.current)
        target_bytes = allocations[1].numel() * allocations[1].element_size()
        self.assertEqual(target_bytes, 192 * 1024**2)
        self.assertEqual(2 * target_bytes, 384 * 1024**2)
        self.assertFalse(hasattr(slot, "output"))

    def test_packed_slot_storage_and_request_bytes_without_device_allocation(self):
        class V41MXFP8Linear:
            pass

        module = SimpleNamespace(
            embed_tokens=object(),
            q_weight=Tensor(),
            layer_hash_index=0,
            layout=SimpleNamespace(n_hash_cols=24, head_dim=256),
            wkv=V41MXFP8Linear(),
        )
        allocations = []
        real_empty = torch.empty

        def allocate(shape, **kwargs):
            self.assertEqual(kwargs["dtype"], torch.uint8)
            allocations.append((shape, kwargs.copy()))
            return real_empty(shape, dtype=torch.uint8, device="meta")

        with (
            patch.dict(
                os.environ,
                {
                    "OMP_NUM_THREADS": "8",
                    "DSV41_CPU_ENGRAM_LOOKUP_TIMING": "0",
                },
                clear=True,
            ),
            patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.modules.dsv41.utils": SimpleNamespace(
                        V41MXFP8Linear=V41MXFP8Linear
                    )
                },
            ),
            patch.object(torch.cuda, "Stream", side_effect=lambda **kwargs: Mock()),
            patch.object(torch.cuda, "Event", side_effect=lambda: Mock()),
            patch.object(torch, "empty", side_effect=allocate),
        ):
            slots = [engram._EngramCPULookupSlot(module, 32768) for _ in range(2)]
            output, allocation_stream = slots[0].allocate_output(16384)
        self.assertEqual(len(allocations), 3)
        for shape, kwargs in allocations[:2]:
            self.assertEqual(shape, (32768, 6336))
            self.assertEqual(
                kwargs, dict(dtype=torch.uint8, device="cpu", pin_memory=True)
            )
        self.assertEqual(
            allocations[2],
            ((16384, 6336), dict(dtype=torch.uint8, device=torch.device("cuda:0"))),
        )
        self.assertTrue(all(slot.packed for slot in slots))
        self.assertTrue(all(slot.num_threads == 16 for slot in slots))
        self.assertEqual(
            sum(slot.host.numel() * slot.host.element_size() for slot in slots),
            396 * 1024**2,
        )
        self.assertEqual(output.numel() * output.element_size(), 99 * 1024**2)
        self.assertIs(allocation_stream, self.current)
        self.assertEqual(slots[0].allocated.record.call_count, 2)
        self.assertTrue(all(not hasattr(slot, "output") for slot in slots))
        self.assertFalse(torch.cuda.is_initialized())

    def test_runtime_warmup_does_not_retain_outputs_or_completed_futures(self):
        references = []

        class Slot:
            def __init__(self, module, capacity):
                self.copied = Mock()
                self.close = Mock()

            def allocate_output(self, count):
                output = torch.empty(count, 192)
                references.append(weakref.ref(output))
                return output, None

            def run(self, hashes, output):
                hashes.result()
                return output

        model = SimpleNamespace(
            _engram_hash_state=self.state,
            _engram_layers=((1, 0), (14, 1)),
            v4=SimpleNamespace(
                layers={
                    index: SimpleNamespace(engram=SimpleNamespace(q_weight=Tensor()))
                    for index in (1, 14)
                }
            ),
        )
        with (
            patch.object(engram, "_EngramCPULookupSlot", Slot),
            patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.modules.dsv41.engram_cpu": SimpleNamespace(
                        prepare_cpu_lookup=Mock(),
                        hash_token_windows_cpu=lambda windows, *args, **kwargs: self.state.hash_token_windows(
                            windows
                        ),
                    )
                },
            ),
        ):
            runtime = engram.EngramCPULookupRuntime(model, 8)
        try:
            # A completed executor WorkItem can briefly retain its arguments;
            # drain that frame as well as the user-visible startup Futures.
            runtime.executor.submit(lambda: None).result()
            self.assertEqual(len(references), 2)
            self.assertTrue(all(reference() is None for reference in references))
            for slot in runtime.slots.values():
                slot.copied.synchronize.assert_called_once()
        finally:
            runtime.close()

    def test_plan_queues_layers_in_order_and_recomputes_same_inputs(self):
        order = []
        windows = torch.zeros(4, 4, dtype=torch.int32)
        slots = {
            index: SimpleNamespace(
                device=torch.device("cuda:0"),
                heads=6,
                collect_timing=False,
                copied=Mock(),
                allocate_output=lambda count: (torch.empty(count, 192), self.current),
                run=lambda hashes, output, timing, index=index: (
                    hashes.result(),
                    order.append(index),
                    output,
                )[-1],
            )
            for index in (1, 14)
        }
        with ThreadPoolExecutor(max_workers=1) as executor:
            model = SimpleNamespace(
                _engram_hash_state=self.state,
                _engram_cpu_runtime=SimpleNamespace(
                    capacity=8, executor=executor, slots=slots
                ),
                v4=SimpleNamespace(
                    layers={
                        index: SimpleNamespace(
                            engram=SimpleNamespace(_cpu_lookup_work=None),
                            engram_hashes=Tensor(4, 6),
                        )
                        for index in slots
                    }
                ),
            )
            for _ in range(2):
                plan = engram.EngramCPULookupPrefetch(model, windows, 9)
                plan.clear()
                self.assertFalse(plan.work)
                self.assertIsNone(plan.hashes)
                for layer in model.v4.layers.values():
                    self.assertIsNone(layer.engram._cpu_lookup_work)
            self.assertEqual(order, [1, 14, 1, 14])
            model.v4.layers[1].engram_hashes = Tensor(3, 6)
            plan = engram.EngramCPULookupPrefetch(model, windows, 9)
            plan.clear()
            self.assertEqual(order[-1], 14)
            self.assertEqual(order.count(1), 2)

    def test_cold_or_cuda_windows_fall_back_without_submission(self):
        executor = Mock()
        model = SimpleNamespace(
            _engram_hash_state=self.state,
            _engram_cpu_runtime=SimpleNamespace(
                capacity=8, executor=executor, slots={}
            ),
        )
        for windows in (
            None,
            Tensor(),
            torch.zeros(9, 4),
            torch.zeros(9, 4, dtype=torch.int32),
            torch.zeros(4, 3, dtype=torch.int32),
            torch.zeros(0, 4, dtype=torch.int32),
        ):
            plan = engram.EngramCPULookupPrefetch(model, windows, 9)
            plan.clear()
        executor.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
