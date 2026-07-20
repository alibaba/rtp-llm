import concurrent.futures
import os
import threading
import time
from typing import List
from unittest import TestCase, main

import PIL
import pillow_avif
import pillow_heif
import torch
from PIL import Image, ImageFile

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
        engine, part = self._make_engine(mm_cache_item_num=10)
        # tearDown restores the global cache to disabled for other tests.

        url = "fake://7"
        r1 = self._embed(engine, [url])
        r2 = self._embed(engine, [url])

        self.assertEqual(r1.embeddings[0].item(), 7)
        self.assertEqual(r2.embeddings[0].item(), 7)
        self.assertEqual(part.embedding_calls, 1)


if __name__ == "__main__":
    main()
