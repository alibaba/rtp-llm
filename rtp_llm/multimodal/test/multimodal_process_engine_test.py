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
        with self.assertRaises(TimeoutError):
            self.mm_process_engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [
                    [-1, -1, -1, -1, -1, -1, -1, [], 1],
                ],
            )

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


if __name__ == "__main__":
    main()
