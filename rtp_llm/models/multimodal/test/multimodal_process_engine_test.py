import os
import tempfile
from random import randint
from unittest import TestCase, main

import PIL
import pillow_avif
import pillow_heif
import torch
from PIL import Image, ImageFile

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.models.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.models.multimodal.multimodal_common import MultiModalEmbeddingInterface
from rtp_llm.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from rtp_llm.models.qwen2_vl.qwen2_vl import QWen2_VL
from rtp_llm.models.qwen2_vl.qwen2_vl_vit import Qwen2VLImageEmbedding
from rtp_llm.utils.multimodal_util import MMPreprocessConfig, MMUrlType


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
        return data


class FakeModel:
    def __init__(self):
        self.config = GptInitModelParameters(0, 0, 0, 0, 0)
        self.config.mm_position_ids_style = 2
        self.config.py_env_configs = PyEnvConfigs()
        self.config.py_env_configs.update_from_env()
        self.mm_part = FakeMultiModalEmbeddingInterface()


class MMProcessEngineTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = FakeModel()
        self.mm_process_engine = MMProcessEngine(self.model)

    def test(self):
        self.mm_process_engine.submit(
            ["./rtp_llm/models/multimodal/test/testdata/qwen2_vl/1.jpg"],
            [MMUrlType.IMAGE],
            [None],
            [],
        )

    def test_timeout(self):
        with self.assertRaises(TimeoutError):
            self.mm_process_engine.submit(
                ["./rtp_llm/models/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [None],
                [
                    [-1, -1, -1, -1, -1, -1, -1, 1],
                ],
            )


if __name__ == "__main__":
    main()
