import concurrent.futures
import os
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
from rtp_llm.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from rtp_llm.models.qwen2_vl.qwen2_vl import QWen2_VL
from rtp_llm.models.qwen2_vl.qwen2_vl_vit import Qwen2VLImageEmbedding
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine, MMWorkItem
from rtp_llm.multimodal.multimodal_common import MultiModalEmbeddingInterface
from rtp_llm.utils.base_model_datatypes import (
    MMPreprocessConfig,
    MMUrlType,
    MultimodalInput,
)


class FakeMultiModalEmbeddingInterface(Qwen2VLImageEmbedding):
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


class FakeMultiModalEmbeddingInterfaceProcessCrash(FakeMultiModalEmbeddingInterface):
    """Preprocess function that crashes the worker process to trigger BrokenProcessPool."""

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        os._exit(1)

    def get_preprocess_params(self):
        return {}


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
            self.model, VitConfig(), ProfilingDebugLoggingConfig()
        )

    def test_embedding(self):
        res = self.mm_process_engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
            [MMUrlType.IMAGE],
            [None],
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
        with self.assertRaises(TimeoutError):
            self.mm_process_engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [None],
                [
                    [-1, -1, -1, -1, -1, -1, -1, [], 1],
                ],
            )

    def test_preprocess(self):
        model = FakeModel(FakeMultiModalEmbeddingInterfacePreprocessException())
        mm_process_engine = MMProcessEngine(
            model, VitConfig(), ProfilingDebugLoggingConfig()
        )
        try:
            mm_process_engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [None],
                [
                    [-1, -1, -1, -1, -1, -1, -1, [], 30000],
                ],
            )
        except PreprcoesException as e:
            self.assertEqual(str(e), "{'test': 'hello'}")

    def test_broken_process_pool_recovery(self):
        """Test comprehensive recovery from BrokenProcessPool in various scenarios."""
        model_crash = FakeModel(FakeMultiModalEmbeddingInterfaceProcessCrash())
        mm_process_engine_crash = MMProcessEngine(
            model_crash, VitConfig(), ProfilingDebugLoggingConfig()
        )

        original_executor = mm_process_engine_crash.mm_preprocess_executor
        original_executor_id = id(original_executor)

        with self.assertRaises(concurrent.futures.process.BrokenProcessPool):
            try:
                mm_process_engine_crash.mm_embedding_cpp(
                    ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                    [MMUrlType.IMAGE],
                    [None],
                    [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
                )
            except concurrent.futures.process.BrokenProcessPool:
                new_executor = mm_process_engine_crash.mm_preprocess_executor
                self.assertIsNotNone(new_executor)
                self.assertNotEqual(original_executor_id, id(new_executor))
                raise

        work_item = MMWorkItem(
            [
                MultimodalInput(
                    "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg",
                    MMUrlType.IMAGE,
                    None,
                    MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
                )
            ]
        )

        work_item.may_submit_preprocess(
            model_crash.mm_part,
            mm_process_engine_crash.vit_config,
            mm_process_engine_crash.mm_preprocess_executor,
        )
        future = work_item.future
        self.assertIsNotNone(future)

        with self.assertRaises(concurrent.futures.process.BrokenProcessPool):
            try:
                work_item.may_get_preprocess_result()
            except concurrent.futures.process.BrokenProcessPool:
                self.assertIsNotNone(mm_process_engine_crash.mm_preprocess_executor)
                raise

        model_normal = FakeModel(FakeMultiModalEmbeddingInterface())
        mm_process_engine_normal = MMProcessEngine(
            model_normal, VitConfig(), ProfilingDebugLoggingConfig()
        )

        work_item_normal = MMWorkItem(
            [
                MultimodalInput(
                    "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg",
                    MMUrlType.IMAGE,
                    None,
                    MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
                )
            ]
        )

        work_item_normal.may_submit_preprocess(
            model_normal.mm_part,
            mm_process_engine_normal.vit_config,
            mm_process_engine_normal.mm_preprocess_executor,
        )
        self.assertIsNotNone(work_item_normal.future)

        executor = mm_process_engine_normal.mm_preprocess_executor
        child_pids = mm_process_engine_normal._get_child_pids(executor)

        executor_id_before = id(executor)
        mm_process_engine_normal._recover_from_broken_process_pool()
        executor_id_after = id(mm_process_engine_normal.mm_preprocess_executor)

        self.assertNotEqual(executor_id_before, executor_id_after)
        self.assertIsNotNone(mm_process_engine_normal.mm_preprocess_executor)

        if child_pids:
            mm_process_engine_normal._kill_child_processes(child_pids)
            time.sleep(0.2)

            for pid in child_pids:
                try:
                    os.kill(pid, 0)
                    self.fail(f"Process {pid} should have been killed")
                except ProcessLookupError:
                    pass
                except OSError:
                    pass

    def test_broken_process_pool_concurrent_recovery(self):
        """Test that concurrent recovery attempts are handled correctly."""
        import threading

        model = FakeModel(FakeMultiModalEmbeddingInterface())
        mm_process_engine = MMProcessEngine(
            model, VitConfig(), ProfilingDebugLoggingConfig()
        )

        recovery_count = [0]
        lock = threading.Lock()

        def recover():
            mm_process_engine._recover_from_broken_process_pool()
            with lock:
                recovery_count[0] += 1

        threads = [threading.Thread(target=recover) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(recovery_count[0], 5)
        self.assertIsNotNone(mm_process_engine.mm_preprocess_executor)


if __name__ == "__main__":
    main()
