import concurrent.futures
import os
import pickle
import threading
import time
from typing import List
from unittest import TestCase, main, mock

import PIL
import pillow_avif
import pillow_heif
import torch
from PIL import Image, ImageFile

from rtp_llm.config.exceptions import (
    AdmissionRejectReason,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import (
    ProfilingDebugLoggingConfig,
    PyEnvConfigs,
    VitConfig,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputPB,
    MultimodalInputsPB,
)
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine, MMWorkItem
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_vl.qwen2_vl_mixin import (
    Qwen2_VLImageEmbedding,
)
from rtp_llm.multimodal.multimodal_util import vit_emb_cache_
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType


class FakeMultiModalEmbeddingInterface(Qwen2_VLImageEmbedding):
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.data_type = config.compute_dtype
        self.image_processor: Qwen2VLImageProcessor = (
            Qwen2VLImageProcessor.from_pretrained(
                "./rtp_llm/multimodal/test/testdata/qwen2_vl/"
            )
        )
        self.spatial_merge_size = 2

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        return torch.tensor(0), None

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        return mm_inputs, kwargs

    def get_preprocess_params(self):
        return {}


class PreprcoesException(Exception):
    pass


class FakeMultiModalEmbeddingInterfacePreprocessException(
    FakeMultiModalEmbeddingInterface
):
    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        raise PreprcoesException(kwargs)

    def get_preprocess_params(self):
        return {"test": "hello"}


class FakeMultiModalEmbeddingInterfaceSlow(FakeMultiModalEmbeddingInterface):
    """Preprocess function that sleeps to guarantee timeout."""

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        time.sleep(5)
        return mm_inputs, kwargs

    def get_preprocess_params(self):
        return {}


class FakeMultiModalEmbeddingInterfaceSlowEmbedding(FakeMultiModalEmbeddingInterface):
    """batched_embedding sleeps, to exercise the embedding-level timeout on the
    default serial scheduler path."""

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        time.sleep(0.2)
        return [(torch.tensor(0), None) for _ in data_list]


class FakeMultiModalEmbeddingInterfaceProcessCrash(FakeMultiModalEmbeddingInterface):
    """Preprocess function that crashes the worker process to trigger BrokenProcessPool."""

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        os._exit(1)

    def get_preprocess_params(self):
        return {}


class FakeMultiModalEmbeddingInterfaceBadCount(FakeMultiModalEmbeddingInterface):
    """batched_embedding returns the wrong number of outputs."""

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        # One fewer than requested, to trip the count guard.
        return [(torch.tensor(0), None) for _ in range(len(data_list) - 1)]


class FakeModel:
    def __init__(self, mm_part: MultiModalEmbeddingInterface = None):
        self.model_config = ModelConfig()
        self.model_config.mm_model_config.mm_position_ids_style = 2
        self.mm_part = mm_part


class MMProcessEngineTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = FakeModel(FakeMultiModalEmbeddingInterface())
        self.mm_process_engine = MMProcessEngine(
            self.model.mm_part,
            self.model.model_config,
            VitConfig(),
            ProfilingDebugLoggingConfig(),
        )

    def setUp(self):
        # Timeout is intentionally excluded from the embedding cache key. Keep
        # cache hits from one test from bypassing another test's preprocess path.
        vit_emb_cache_.resize_cache(0)

    def tearDown(self):
        vit_emb_cache_.resize_cache(0)

    def test_embedding(self):
        res = self.mm_process_engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        self.assertEqual(res.position_ids, [])

        mm_inputs = MultimodalInputsPB()
        mm_input = MultimodalInputPB()
        mm_input.multimodal_url = "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"
        mm_input.multimodal_type = MMUrlType.IMAGE
        mm_input.mm_preprocess_config.mm_timeout_ms = 30000
        mm_inputs.multimodal_inputs.append(mm_input)
        res = self.mm_process_engine.mm_embedding_rpc(mm_inputs)
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        self.assertEqual(res.position_ids, [])

    def test_timeout(self):
        model = FakeModel(FakeMultiModalEmbeddingInterfaceSlow())
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            VitConfig(),
            ProfilingDebugLoggingConfig(),
        )
        with self.assertRaises(TimeoutError):
            engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [
                    [-1, -1, -1, -1, -1, -1, -1, [], 1],
                ],
            )
        engine.stop()

    def test_preprocess(self):
        model = FakeModel(FakeMultiModalEmbeddingInterfacePreprocessException())
        mm_process_engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            VitConfig(),
            ProfilingDebugLoggingConfig(),
        )
        try:
            mm_process_engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [
                    [-1, -1, -1, -1, -1, -1, -1, [], 30000],
                ],
            )
        except PreprcoesException as e:
            self.assertEqual(str(e), "{'test': 'hello'}")

    def test_local_preprocess_mode(self):
        """LocalPreprocessExecutor path: use_local_preprocess=True bypasses the worker pool."""
        model = FakeModel(FakeMultiModalEmbeddingInterface())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        res = engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        engine.stop()

    def test_query_counter(self):
        self.assertEqual(self.mm_process_engine.get_query_num(), 0)
        self.mm_process_engine.inc_query_num()
        self.mm_process_engine.inc_query_num()
        self.assertEqual(self.mm_process_engine.get_query_num(), 2)
        self.mm_process_engine.dec_query_num()
        self.assertEqual(self.mm_process_engine.get_query_num(), 1)
        self.mm_process_engine.dec_query_num()
        self.assertEqual(self.mm_process_engine.get_query_num(), 0)

    def test_work_item_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            MMWorkItem([])

    def test_work_item_uses_global_timeout_when_request_timeout_is_unset(self):
        preprocess_config = MMPreprocessConfig()
        mm_input = MultimodalInput(
            "", MMUrlType.IMAGE, torch.empty(0), preprocess_config
        )

        self.assertEqual(preprocess_config.mm_timeout_ms, -1)
        self.assertEqual(
            MMWorkItem([mm_input], mm_timeout_ms=123000).mm_timeout_ms, 123000
        )

    def test_work_item_uses_largest_resolved_batch_timeout(self):
        first = MultimodalInput(
            "", MMUrlType.IMAGE, torch.empty(0), MMPreprocessConfig(mm_timeout_ms=1000)
        )
        second = MultimodalInput(
            "", MMUrlType.IMAGE, torch.empty(0), MMPreprocessConfig(mm_timeout_ms=3000)
        )
        inherited = MultimodalInput(
            "", MMUrlType.IMAGE, torch.empty(0), MMPreprocessConfig()
        )

        self.assertEqual(
            MMWorkItem([first, second, inherited], mm_timeout_ms=2000).mm_timeout_ms,
            3000,
        )

    def test_embedding_timeout_default_path(self):
        """Default (non-gpu-batch) serial path enforces an embedding-level timeout.

        A slow batched_embedding must surface as TimeoutError rather than block the
        caller indefinitely.
        """
        model = FakeModel(FakeMultiModalEmbeddingInterfaceSlowEmbedding())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True  # fast preprocess; isolate embedding
        vit_config.mm_cache_item_num = 0  # no cache hit to short-circuit the forward
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        try:
            with self.assertRaises(TimeoutError):
                engine.mm_embedding_cpp(
                    ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                    [MMUrlType.IMAGE],
                    [torch.empty(0)],
                    [[-1, -1, -1, -1, -1, -1, -1, [], 20]],
                )
        finally:
            engine.stop()

    def test_batched_embedding_count_mismatch(self):
        """Serial-mode scheduler path fails fast when batched_embedding returns wrong count."""
        model = FakeModel(FakeMultiModalEmbeddingInterfaceBadCount())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True  # local preprocess, serial scheduler
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        try:
            with self.assertRaises(RuntimeError):
                engine.mm_embedding_cpp(
                    ["url0", "url1"],
                    [MMUrlType.IMAGE, MMUrlType.IMAGE],
                    [torch.empty(0), torch.empty(0)],
                    [[-1, -1, -1, -1, -1, -1, -1, [], 30000]] * 2,
                )
        finally:
            engine.stop()

    def test_worker_crash_recovery(self):
        """Pool rebuilds after worker process crash and subsequent requests succeed."""
        model = FakeModel(FakeMultiModalEmbeddingInterfaceProcessCrash())
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            VitConfig(),
            ProfilingDebugLoggingConfig(),
        )

        # First call crashes the worker — should raise but pool rebuilds internally
        with self.assertRaises(Exception):
            engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [[-1, -1, -1, -1, -1, -1, -1, [], 5000]],
            )

        # Swap to a working mm_part so the rebuilt pool can serve requests
        working_model = FakeModel(FakeMultiModalEmbeddingInterface())
        engine.preprocess_executor.preprocess_func = (
            working_model.mm_part.preprocess_input
        )
        engine.preprocess_executor._rebuild_pool()

        # Subsequent request should succeed after pool recovery
        res = engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        engine.stop()

    def test_consecutive_timeout_triggers_rebuild(self):
        """Pool rebuilds after consecutive timeouts reach the threshold."""
        from rtp_llm.multimodal.mm_process_engine import MultiprocessPreprocessExecutor

        model = FakeModel(FakeMultiModalEmbeddingInterfaceSlow())
        vit_config = VitConfig()
        vit_config.mm_preprocess_max_workers = 2
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )

        executor = engine.preprocess_executor
        if not isinstance(executor, MultiprocessPreprocessExecutor):
            self.skipTest("Not using multiprocess executor")

        old_pool = executor.pool

        # Simulate consecutive timeouts reaching the threshold
        executor._consecutive_timeouts = executor._max_consecutive_timeouts - 1

        # This timeout should trigger a rebuild
        with self.assertRaises(TimeoutError):
            engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [[-1, -1, -1, -1, -1, -1, -1, [], 1]],
            )

        # Pool should have been rebuilt
        self.assertIsNot(executor.pool, old_pool)
        self.assertEqual(executor._consecutive_timeouts, 0)
        engine.stop()


_DEFAULT_CONFIG = [-1, -1, -1, -1, -1, -1, -1, [], 30000]


class FakeBatchMMPart(MultiModalEmbeddingInterface):
    """mm_part returning identity-encoded (emb, pos, extra) tuples.

    Each input carries an index in its url ("fake://<i>"); embedding echoes that
    index into all three output tensors so tests can assert ordering, and counts
    embedding/batched_embedding invocations to observe cache hits and batching.
    """

    def __init__(self):
        self.embedding_calls = 0
        self.batch_sizes: List[int] = []
        self._lock = threading.Lock()

    @staticmethod
    def preprocess_input(mm_inputs, vit_config, **kwargs):
        # Carry the inputs through; embedding derives identity from the url.
        return mm_inputs, kwargs

    def get_preprocess_params(self):
        return {}

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        mm_inputs, _ = data
        idx = float(int(mm_inputs[0].url.split("://")[1]))
        with self._lock:
            self.embedding_calls += 1
        emb = torch.tensor([[idx]])  # (1, 1) -> one embedding per work item
        pos = torch.tensor([[idx]])  # (1, 1)
        extra = torch.tensor([idx])  # (1,) -> one flat extra tensor
        return emb, pos, extra

    def batched_embedding(self, data_list, mm_types, **kwargs):
        with self._lock:
            self.batch_sizes.append(len(data_list))
        return super().batched_embedding(data_list, mm_types, **kwargs)


class MMProcessEngineGpuBatchTest(TestCase):
    def setUp(self):
        # vit_emb_cache_ is a process-global; isolate it so cache state never
        # leaks between these tests (or into other test classes in the process).
        vit_emb_cache_.resize_cache(0)

    def tearDown(self):
        vit_emb_cache_.resize_cache(0)

    def _make_engine(self, **vit_overrides):
        model = FakeModel(FakeBatchMMPart())
        vit_config = VitConfig()
        # Enable cross-request batching (the --use_gpu_batch boolean was removed;
        # batching is now inferred from gpu_max_batch_size > 1). Individual tests
        # override gpu_max_batch_size via vit_overrides as needed.
        vit_config.gpu_max_batch_size = 8
        # Local preprocess keeps the test in-process and deterministic.
        vit_config.use_local_preprocess = True
        # Cache off by default; the cache test opts in explicitly.
        vit_config.mm_cache_item_num = 0
        for key, value in vit_overrides.items():
            setattr(vit_config, key, value)
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        self.addCleanup(engine.stop)
        return engine, model.mm_part

    def _embed(self, engine, urls):
        n = len(urls)
        return engine.mm_embedding_cpp(
            urls,
            [MMUrlType.IMAGE] * n,
            [torch.empty(0)] * n,
            [list(_DEFAULT_CONFIG) for _ in range(n)],
        )

    def test_gpu_batch_order_and_outputs(self):
        """Single multi-image request: emb/pos/extra preserve input order."""
        engine, _ = self._make_engine()
        urls = [f"fake://{i}" for i in range(4)]
        res = self._embed(engine, urls)

        self.assertEqual([e.item() for e in res.embeddings], [0, 1, 2, 3])
        self.assertEqual([p.item() for p in res.position_ids], [0, 1, 2, 3])
        self.assertEqual([x.item() for x in res.extra_input], [0, 1, 2, 3])

    def test_gpu_batch_multi_request(self):
        """Concurrent requests are batched yet each gets its own correct result."""
        engine, part = self._make_engine(gpu_batch_wait_ms=400, gpu_max_batch_size=16)
        n = 5
        results: List[float] = [None] * n

        def run(i: int):
            res = self._embed(engine, [f"fake://{i}"])
            results[i] = res.embeddings[0].item()

        threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(results, [0.0, 1.0, 2.0, 3.0, 4.0])
        # The wait window should let at least one forward serve >1 request.
        self.assertGreaterEqual(max(part.batch_sizes), 2)

    def test_gpu_batch_cache_hit(self):
        """A repeated url is served from cache without a second embedding call."""
        engine, part = self._make_engine(mm_cache_cpu_max_bytes=4096)
        # tearDown restores the global cache to disabled for other tests.

        url = "fake://7"
        r1 = self._embed(engine, [url])
        r2 = self._embed(engine, [url])

        self.assertEqual(r1.embeddings[0].item(), 7)
        self.assertEqual(r2.embeddings[0].item(), 7)
        self.assertEqual(part.embedding_calls, 1)


class PreprocessMetricTest(TestCase):
    @mock.patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_download_components_preserve_local_and_worker_samples(self, report):
        from rtp_llm.multimodal import mm_process_engine as engine_module
        from rtp_llm.multimodal.multimodal_util import _download_timing
        from rtp_llm.multimodal.vit_metrics import record_vit_preprocess_value

        def preprocess(inputs, config):
            _download_timing.get().elapsed_ms = 3.0
            record_vit_preprocess_value(
                GaugeMetrics.VIT_IMAGE_FETCH_RT_US_METRIC, 3000, {"mm_type": "video"}
            )
            return "prepared"

        with mock.patch.object(engine_module, "Timer") as timer:
            timer.return_value.cost_ms.return_value = 10.0
            timer.return_value.__enter__.return_value = timer.return_value
            local = engine_module.LocalPreprocessExecutor(preprocess, VitConfig(), {})
            item = mock.Mock(should_preprocess=True, mm_inputs=[], mm_timeout_ms=30000)
            local.submit(item)
            local.get_result(item)
            with mock.patch.multiple(
                engine_module,
                _worker_preprocess_func=preprocess,
                _worker_vit_config=VitConfig(),
                _worker_preprocess_params={},
            ):
                payload = engine_module._worker_process_task([])
            # The worker wire format keeps the sample list in its third field.
            restored = pickle.loads(pickle.dumps(payload))
            executor = object.__new__(engine_module.MultiprocessPreprocessExecutor)
            executor._pool_lock = threading.Lock()
            item.future = mock.Mock()
            item.future.get.return_value = restored
            executor.get_result(item)
        for metric, expected in (
            (GaugeMetrics.VIT_DOWNLOAD_RT_METRIC, 3.0),
            (GaugeMetrics.VIT_PREPROCESS_OTHER_RT_METRIC, 7.0),
            (GaugeMetrics.VIT_IMAGE_FETCH_RT_US_METRIC, 3000.0),
        ):
            values = [c.args[1] for c in report.call_args_list if c.args[0] == metric]
            self.assertEqual(values, [expected, expected])

    @mock.patch("rtp_llm.multimodal.mm_process_engine.logging.exception")
    @mock.patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_metrics_backend_failure_does_not_fail_preprocessing(self, report, log):
        from rtp_llm.multimodal import mm_process_engine as engine_module

        report.side_effect = RuntimeError("metrics unavailable")
        local = engine_module.LocalPreprocessExecutor(
            lambda inputs, config: "prepared", VitConfig(), {}
        )
        item = mock.Mock(should_preprocess=True, mm_inputs=[], mm_timeout_ms=30000)
        local.submit(item)
        payload = item.future.get()
        local.get_result(item)
        self.assertEqual(item.preprocess_result, "prepared")
        executor = object.__new__(engine_module.MultiprocessPreprocessExecutor)
        executor._pool_lock = threading.Lock()
        item.future = mock.Mock()
        item.future.get.return_value = payload
        executor.get_result(item)
        self.assertEqual(item.preprocess_result, "prepared")
        reported_metrics = [call.args[0] for call in report.call_args_list]
        for metric in (
            GaugeMetrics.VIT_PREPROCESS_RT_METRIC,
            GaugeMetrics.VIT_DOWNLOAD_RT_METRIC,
            GaugeMetrics.VIT_PREPROCESS_OTHER_RT_METRIC,
        ):
            self.assertEqual(reported_metrics.count(metric), 2)

    def test_preprocess_components_cannot_be_negative(self):
        from rtp_llm.multimodal import mm_process_engine as engine_module
        from rtp_llm.multimodal.multimodal_util import _download_timing

        def preprocess(inputs, config):
            _download_timing.get().elapsed_ms = 11.0
            return "prepared"

        with mock.patch.object(engine_module, "Timer") as timer:
            timer.return_value.cost_ms.return_value = 10.0
            timer.return_value.__enter__.return_value = timer.return_value
            _, total, samples = engine_module._run_preprocess_task(
                preprocess, [], VitConfig(), {}
            )
        values = {s.metric: s.value for s in samples}
        self.assertEqual(values[GaugeMetrics.VIT_DOWNLOAD_RT_METRIC], total)
        self.assertEqual(values[GaugeMetrics.VIT_PREPROCESS_OTHER_RT_METRIC], 0.0)

    @mock.patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_process_pool_restart_only_after_success(self, report):
        from rtp_llm.multimodal.mm_process_engine import MultiprocessPreprocessExecutor

        executor = object.__new__(MultiprocessPreprocessExecutor)
        executor.pool = mock.Mock()
        old_pool = executor.pool
        executor._clear_preprocess_tasks = mock.Mock()
        executor._create_pool = mock.Mock()
        executor._rebuild_pool()
        old_pool.terminate.assert_called_once()
        old_pool.join.assert_called_once()
        report.assert_called_once_with(
            AccMetrics.VIT_PROCESS_POOL_RESTART_QPS_METRIC, 1
        )
        report.reset_mock()
        executor._create_pool.side_effect = OSError("cannot create pool")
        with self.assertRaises(OSError):
            executor._rebuild_pool()
        report.assert_not_called()
        # Metrics failure must not turn a successful rebuild into a failure.
        executor._create_pool.side_effect = None
        report.side_effect = RuntimeError("metrics unavailable")
        executor._rebuild_pool()

    @mock.patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    @mock.patch("rtp_llm.multimodal.mm_process_engine._feature_hashes_from_result")
    def test_embedding_length_once_for_sync_async_cache_and_hash_only(
        self, hashes, report
    ):
        hashes.side_effect = lambda result: [
            torch.zeros(result[0].shape[0], dtype=torch.int64)
        ]
        model = FakeModel(FakeBatchMMPart())
        config = VitConfig()
        config.use_local_preprocess = True
        config.mm_cache_gpu_max_bytes = 0
        config.mm_cache_cpu_max_bytes = 4096
        config.disable_access_log = True
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            config,
            ProfilingDebugLoggingConfig(),
            device="cpu",
        )
        preprocess = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000)

        def inputs(index):
            return [
                MultimodalInput(
                    f"fake://{index}", MMUrlType.IMAGE, torch.empty(0), preprocess
                )
            ]

        try:
            engine.mm_embedding_impl(inputs(0))
            engine.mm_embedding_impl(inputs(0))  # synchronous cache hit
            engine.get_embedding_result(inputs(1), request_id=1)  # new async task
            engine.get_embedding_result(inputs(1), request_id=2)  # cache hit
            result = engine.get_embedding_result(
                inputs(1), request_id=3, hashes_only=True
            )
            self.assertEqual(result[0].embeddings, [])
            self.assertEqual(model.mm_part.embedding_calls, 2)
        finally:
            engine.stop()
        values = [
            call.args[1]
            for call in report.call_args_list
            if call.args[0] == GaugeMetrics.VIT_EMBEDDING_LENGTH_METRIC
        ]
        self.assertEqual(values, [1, 1, 1, 1, 1])

    def test_embedding_length_counts_tokens_not_hidden_width(self):
        from rtp_llm.multimodal.mm_process_engine import _embedding_token_length

        self.assertEqual(
            _embedding_token_length(
                [torch.zeros(3, 8), torch.zeros(8), torch.empty(0, 8)]
            ),
            4,
        )

    @mock.patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_preprocess_queue_metric_tracks_pending_tasks(self, report):
        from rtp_llm.multimodal.mm_process_engine import MultiprocessPreprocessExecutor

        class FakePool:
            def __init__(self):
                self.callbacks = []

            def apply_async(self, *args, **kwargs):
                self.callbacks.append((kwargs["callback"], kwargs["error_callback"]))
                return object()

        executor = object.__new__(MultiprocessPreprocessExecutor)
        executor.pool = FakePool()
        executor._pool_lock = threading.Lock()
        executor._preprocess_queue_lock = threading.Lock()
        executor._pending_preprocess_tasks = set()
        executor._next_preprocess_task_id = 0

        config = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000)
        work_items = [
            MMWorkItem(
                [
                    MultimodalInput(
                        f"fake://queue-{index}",
                        MMUrlType.IMAGE,
                        torch.empty(0),
                        config,
                    )
                ],
                mm_timeout_ms=30000,
            )
            for index in range(2)
        ]

        executor.submit(work_items[0])
        executor.submit(work_items[1])
        depth_values = [
            call.args[1]
            for call in report.call_args_list
            if call.args
            and call.args[0] == GaugeMetrics.VIT_PREPROCESS_QUEUE_SIZE_METRIC
        ]
        self.assertEqual(depth_values[-1], 2)

        executor.pool.callbacks[0][0](None)
        executor.pool.callbacks[1][1](RuntimeError("preprocess failed"))
        depth_values = [
            call.args[1]
            for call in report.call_args_list
            if call.args
            and call.args[0] == GaugeMetrics.VIT_PREPROCESS_QUEUE_SIZE_METRIC
        ]
        self.assertEqual(depth_values[-1], 0)

    @mock.patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_image_count_once_per_logical_sync_and_async_request(self, report):
        model = FakeModel(FakeMultiModalEmbeddingInterface())
        config = VitConfig()
        config.use_local_preprocess = True
        config.mm_cache_gpu_max_bytes = 0
        config.mm_cache_cpu_max_bytes = 0
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            config,
            ProfilingDebugLoggingConfig(),
            device="cpu",
        )
        preprocess = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000)
        inputs = [
            MultimodalInput(
                "fake://image", MMUrlType.IMAGE, torch.empty(0), preprocess
            ),
            MultimodalInput(
                "fake://video", MMUrlType.VIDEO, torch.empty(0), preprocess
            ),
        ]
        try:
            engine.mm_embedding_impl(inputs)
            engine.get_embedding_result(inputs, request_id=17)
        finally:
            engine.stop()
        counts = [
            call.args[1]
            for call in report.call_args_list
            if call.args[0] == GaugeMetrics.VIT_IMAGE_COUNT_METRIC
        ]
        self.assertEqual(counts, [1, 1])


class FtRuntimeExceptionSerializationTest(TestCase):
    def test_pickle_round_trip_preserves_error(self):
        for reason in AdmissionRejectReason:
            with self.subTest(reason=reason):
                error = FtRuntimeException(
                    ExceptionType.MM_DOWNLOAD_FAILED,
                    "Failed to download multimodal content",
                    reason,
                )

                restored = pickle.loads(pickle.dumps(error))

                self.assertIsInstance(restored, FtRuntimeException)
                self.assertEqual(
                    restored.exception_type, ExceptionType.MM_DOWNLOAD_FAILED
                )
                self.assertEqual(restored.message, error.message)
                self.assertEqual(restored.admission_reject_reason, reason)
                self.assertEqual(str(restored), error.message)

    def test_remote_rpc_error_code_is_registered(self):
        self.assertEqual(ExceptionType.from_value(907), "MM_REMOTE_RPC_FAILED")
        self.assertEqual(ExceptionType(907), ExceptionType.MM_REMOTE_RPC_FAILED)


if __name__ == "__main__":
    main()
