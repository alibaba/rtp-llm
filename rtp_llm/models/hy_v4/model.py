"""HY V4 target and draft model entry points."""

import json
import os

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.hy_v4.config import create_hy4_config
from rtp_llm.models.hy_v4.mtp_weight import Hy4MtpWeight
from rtp_llm.models.hy_v4.weight import Hy4Weight


class Hy4(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        return create_hy4_config(ckpt_path)

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.hy_v4 import Hy4Model

        self.py_model = Hy4Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    def support_cuda_graph(self) -> bool:
        return True

    @staticmethod
    def get_weight_cls():
        return Hy4Weight


class Hy4Mtp(Hy4):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        raw_path = os.path.join(ckpt_path, "config.json")
        with open(raw_path) as reader:
            raw = json.load(reader)
        config.num_layers = int(raw.get("num_nextn_predict_layers", 1)) or 1
        if config.num_layers != 1:
            raise ValueError(
                f"HY V4 RTP draft currently supports one MTP layer, got {config.num_layers}"
            )
        config.enable_ihc = False
        # HY V4 applies hc_head in the target model before handing hidden states
        # to MTP.  Its draft therefore consumes one hidden-size stream, unlike
        # DeepSeek V4 MTP which consumes the pre-hc_head hc_mult streams.
        config.hc_mult = 1
        config.mlp_layer_types = ["sparse"]
        config.moe_layer_index = [0]
        config.indexer_types = ["full"]
        config.index_share_for_mtp_iteration = False
        config.is_mtp = True
        config.model_type = "hy_v4_mtp"
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.hy_v4 import Hy4MtpModel

        self.py_model = Hy4MtpModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    @staticmethod
    def get_weight_cls():
        return Hy4MtpWeight
