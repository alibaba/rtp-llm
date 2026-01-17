import json
import os
from typing import List, Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.qwen3_next.qwen3_next_weight import Qwen3NextWeight
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.utils.arch import is_cuda
from rtp_llm.ops import HybridAttentionType


class Qwen3Next(BaseModel):
    @staticmethod
    def get_weight_cls():
        return Qwen3NextWeight

    def _create_python_model(self) -> Optional[GptModelBase]:
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        max_generate_batch_size = self.max_generate_batch_size
        if not is_cuda():
            raise RuntimeError("Qwen3Next is only supported in cuda arch")
        from rtp_llm.models_py.model_desc.qwen_next import Qwen3NextModel

        self.py_model = Qwen3NextModel(
            model_config,
            parallelism_config,
            self.weight,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    def support_cuda_graph(self) -> bool:
        return True

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.qwen3_next import create_qwen3_next_config

        config = create_qwen3_next_config(ckpt_path)
        return config


register_model("qwen3_next", Qwen3Next, ["Qwen3NextForCausalLM"])
