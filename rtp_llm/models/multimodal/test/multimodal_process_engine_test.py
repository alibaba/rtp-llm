import concurrent.futures
import os
import tempfile
from random import randint
from typing import List
from unittest import TestCase, main

import PIL
import pillow_avif
import pillow_heif
import torch
from PIL import Image, ImageFile

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputPB,
    MultimodalInputsPB,
)
from rtp_llm.models.multimodal.mm_process_engine import MMProcessEngine, MMWorkItem
from rtp_llm.models.multimodal.multimodal_common import MultiModalEmbeddingInterface
from rtp_llm.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from rtp_llm.models.qwen2_vl.qwen2_vl import QWen2_VL
from rtp_llm.models.qwen2_vl.qwen2_vl_vit import Qwen2VLImageEmbedding
from rtp_llm.utils.base_model_datatypes import (
    MMPreprocessConfig,
    MMUrlType,
    MultimodalInput,
)


class FakeMultiModalEmbeddingInterface(Qwen2VLImageEmbedding):
    def __init__(self):
        self.image_processor: Qwen2VLImageProcessor = (
            Qwen2VLImageProcessor.from_pretrained(
                "./rtp_llm/models/multimodal/test/testdata/qwen2_vl/"
            )
        )
        self.spatial_merge_size = 2

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        return torch.tensor(0), None

    @staticmethod
    def preprocess_input(mm_inputs: List[MultimodalInput], **kwargs):
        return mm_inputs, kwargs

    def get_preprocess_params(self):
        return {}


class PreprcoesException(Exception):
    pass


class FakeMultiModalEmbeddingInterfacePreprocessException(
    FakeMultiModalEmbeddingInterface
):
    @staticmethod
    def preprocess_input(mm_inputs: List[MultimodalInput], **kwargs):
        raise PreprcoesException(kwargs)

    def get_preprocess_params(self):
        return {"test": "hello"}


class FakeModel:
    def __init__(self, mm_part: MultiModalEmbeddingInterface = None):
        self.config = GptInitModelParameters(0, 0, 0, 0, 0)
        self.config.mm_position_ids_style = 2
        self.config.py_env_configs = PyEnvConfigs()
        self.config.py_env_configs.update_from_env()
        self.mm_part = mm_part


class MMProcessEngineTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = FakeModel(FakeMultiModalEmbeddingInterface())
        self.mm_process_engine = MMProcessEngine(self.model)

    def test_embedding(self):
        res = self.mm_process_engine.mm_embedding_cpp(
            ["./rtp_llm/models/multimodal/test/testdata/qwen2_vl/1.jpg"],
            [MMUrlType.IMAGE],
            [None],
            [[-1, -1, -1, -1, -1, -1, -1, 30000]],
        )
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        self.assertEqual(res.position_ids, [])

        mm_inputs = MultimodalInputsPB()
        mm_input = MultimodalInputPB()
        mm_input.multimodal_url = (
            "./rtp_llm/models/multimodal/test/testdata/qwen2_vl/1.jpg"
        )
        mm_input.multimodal_type = MMUrlType.IMAGE
        mm_input.mm_preprocess_config.mm_timeout_ms = 30000
        mm_inputs.multimodal_inputs.append(mm_input)
        res = self.mm_process_engine.mm_embedding_rpc(mm_inputs)
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        self.assertEqual(res.position_ids, [])

    def test_timeout(self):
        with self.assertRaises(concurrent.futures._base.TimeoutError):
            self.mm_process_engine.mm_embedding_cpp(
                ["./rtp_llm/models/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [None],
                [
                    [-1, -1, -1, -1, -1, -1, -1, 1],
                ],
            )

    def test_preprocess(self):
        model = FakeModel(FakeMultiModalEmbeddingInterfacePreprocessException())
        mm_process_engine = MMProcessEngine(model)
        try:
            mm_process_engine.mm_embedding_cpp(
                ["./rtp_llm/models/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [None],
                [
                    [-1, -1, -1, -1, -1, -1, -1, 30000],
                ],
            )
        except PreprcoesException as e:
            self.assertEqual(str(e), "{'test': 'hello'}")


if __name__ == "__main__":
    main()
