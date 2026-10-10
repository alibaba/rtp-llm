"""CPU stream-contract tests; no CUDA context or model checkpoint required."""

import ast
import importlib.util
import os
import sys
import unittest
import weakref
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
                "rtp_llm.models_py.modules.dsv4": SimpleNamespace(
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
        package = sys.modules["rtp_llm.models_py.modules.dsv4"]
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

        base_name = "rtp_llm.models_py.model_desc.deepseek_v4_model"
        module_spec = importlib.util.spec_from_file_location(
            "engram_model_prefetch_test",
            Path(__file__).resolve().parents[3] / "model_desc/deepseek_v41_model.py",
        )
        model_module = importlib.util.module_from_spec(module_spec)
        with patch.dict(
            sys.modules,
            {
                base_name: SimpleNamespace(
                    DeepSeekV4Model=Base,
                    _is_decode_fmha=lambda value: value == "decode",
                )
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
        model._engram_layers = ((1, 0),)
        model._image_plan = object()
        model._prepare_engram = Mock()
        model._prepare_image_features = Mock()
        model.parent_forward = Mock(return_value="output")
        context = patch.dict(
            sys.modules, {"rtp_llm.models_py.modules.dsv4.engram": engram}
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


if __name__ == "__main__":
    unittest.main()
