import logging
from typing import Any, Dict, List, NamedTuple, Optional, Union

import torch
from pydantic import BaseModel as PyBaseModel

from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.config.gpt_init_model_parameters import ConfigMode, GptInitModelParameters
from rtp_llm.config.task_type import TaskType
from rtp_llm.distribute.gang_info import get_gang_info
from rtp_llm.distribute.worker_info import ParallelInfo, g_parallel_info
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import (
    BaseTokenizer,
    TokenizerFactory,
)
from rtp_llm.model_loader.loader import ModelLoader, get_model_loader
from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo, ModelWeights
from rtp_llm.models.downstream_modules.custom_module import CustomModule
from rtp_llm.models.downstream_modules.utils import create_custom_module
from rtp_llm.models.multimodal.multimodal_mixin import MultiModalMixin
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.config.model_config import ModelConfig as PyModelConfig
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    EmbeddingOutput,
    GenerateContext,
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
    GenerateResponse,
    LegacyModelConfig as ModelConfig,
)
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.multimodal_util import MultimodalInput
from rtp_llm.utils.time_util import timer_wrapper
from rtp_llm.utils.util import to_torch_dtype

FT_DEFAULT_MAX_NEW_TOKENS = 2048


class BaseModel(object):

    config: GptInitModelParameters
    device: str

    def __init__(self, config: GptInitModelParameters) -> None:
        self.config = config
        self.weight = None

        self.linear_bias_slopes: Optional[torch.Tensor] = None
        self.prefix_tokens: Optional[torch.Tensor] = None
        self.tokenizer: Optional[BaseTokenizer] = None
        self.max_input_buffer_len: int = 0

        self.task_type: TaskType = TaskType.LANGUAGE_MODEL
        self.custom_module: Optional[CustomModule] = None
        self.is_attn_model = False
        if self.config:
            self.is_attn_model = (
                config.gpt_init_params.ffn_disaggregate_config.enable_ffn_disaggregate
                and not config.gpt_init_params.ffn_disaggregate_config.is_ffn_service()
            )

        self.default_generate_config: GenerateConfig = GenerateConfig()
        self.load_tokenizer()

        self.py_model: Optional[GptModelBase] = None

    @timer_wrapper(description="load model")
    def load(self, parallel_info: ParallelInfo = g_parallel_info):
        if (
            self.config.model_specific_config.load_python_model
            and self.config.hw_kernel_config.enable_cuda_graph
            and self.support_cuda_graph() is False
        ):
            raise Exception("current model can't support cuda graph in py model mode")

        self.model_weights_loader = self.create_model_loader(parallel_info)
        self._load(self.device)

        if self.config.model_specific_config.load_python_model:
            logging.info(
                f"Creating python model for {self.config.py_model_config.ckpt_path} on {self.device}"
            )
            self._create_python_model()
        else:
            logging.info(f"Skip creating python model, use legacy cpp GptModel")

    def _create_python_model(self) -> Optional[GptModelBase]:
        raise NotImplementedError("Python Model is not implemented for this model.")

    def support_cuda_graph(self) -> bool:
        return False

    def _load(self, device: str):
        # set empty weights for attention service
        self.weight: ModelWeights = self.model_weights_loader.load_weights(
            device=self.device
        )
        self._load_custom_module()
        self._load_multimodal()
        self.model_weights_loader.force_clean_cuda_memory()

    @classmethod
    def create_config(
        cls,
        model_config: ModelConfig,
        parallel_info: ParallelInfo = g_parallel_info,
        config_mode: ConfigMode = ConfigMode.ComplexMode,
    ) -> GptInitModelParameters:
        # Call _create_config to get PyModelConfig from checkpoint
        py_model_config: PyModelConfig = cls._create_config(model_config.ckpt_path)
        # Update the config with model-specific settings
        py_model_config = cls._update_config(py_model_config)
        
        # Calculate hidden_size if not set
        if py_model_config.hidden_size == 0 and py_model_config.head_num > 0 and py_model_config.size_per_head > 0:
            py_model_config.hidden_size = py_model_config.head_num * py_model_config.size_per_head
        
        gpt_config = GptInitModelParameters()
        gpt_config.py_model_config = py_model_config
        
        # Update common settings from legacy ModelConfig
        gpt_config.update_common(
            ckpt_path=model_config.ckpt_path,
            tokenizer_path=model_config.tokenizer_path,
            quantization=model_config.quantization,
            data_type=model_config.act_type,
            kv_cache_type=model_config.kv_cache_type,
            max_seq_len=model_config.max_seq_len,
            seq_size_per_block=model_config.seq_size_per_block,
            gen_num_per_circle=model_config.gen_num_per_circle,
            lora_infos=model_config.lora_infos,
            ptuning_path=model_config.ptuning_path,
            parallel_info=parallel_info,
            gang_info=get_gang_info(),
            config_mode=config_mode,
        )
        
        return gpt_config

    @classmethod
    def _create_config(cls, ckpt_path: str) -> PyModelConfig:
        raise NotImplementedError()

    @classmethod
    def _update_config(cls, model_config: PyModelConfig) -> PyModelConfig:
        return model_config

    @classmethod
    def from_config(
        cls, config: Any, parallel_info: ParallelInfo = g_parallel_info
    ) -> "BaseModel":
        model = cls(config)
        model.load(parallel_info)
        return model

    @staticmethod
    def get_weight_cls() -> ModelDeployWeightInfo:
        raise NotImplementedError

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        assert self.weight is not None
        return self.weight.dtype

    @timer_wrapper(description="init mutlimodal")
    def _may_init_multimodal(self):
        if self.is_multimodal():
            assert isinstance(self, MultiModalMixin)  # for syntax check
            self.config.is_multimodal = True
            if self.parallel_info.tp_rank == 0:
                self.init_multimodal(self.config, self.device)

    @timer_wrapper(description="init custom_module")
    def _init_misc(self):
        self._may_init_multimodal()
        # Get task_type from C++ ModelConfig and convert to enum
        task_type_str = self.config.gpt_init_params.model_config.get_task_type()
        try:
            self.task_type = TaskType.from_str(task_type_str)
        except:
            self.task_type = TaskType.LANGUAGE_MODEL
        self.custom_module = self._init_custom_module()
        self.compute_dtype: torch.dtype = to_torch_dtype(self.config.gpt_init_params.model_config.data_type)

    def _init_custom_module(self) -> Optional[CustomModule]:
        return create_custom_module(self.task_type, self.config, self.tokenizer)

    def load_tokenizer(self) -> None:
        # Get tokenizer parameters from config
        ckpt_path = self.config.py_model_config.ckpt_path
        tokenizer_path = self.config.py_model_config.tokenizer_path_
        if not tokenizer_path:
            tokenizer_path = ckpt_path
        
        # Get model_type from config.json
        import json
        import os
        config_json_path = os.path.join(ckpt_path, "config.json")
        model_type = ""
        if os.path.exists(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as reader:
                config_json = json.loads(reader.read())
                # Try to get model_type from architectures field
                if "architectures" in config_json and config_json["architectures"]:
                    model_type = config_json["architectures"][0].lower()
                elif "model_type" in config_json:
                    model_type = config_json["model_type"].lower()
        
        # Fallback to StaticConfig if not found in config.json
        if not model_type:
            from rtp_llm.config.py_config_modules import StaticConfig
            model_type = StaticConfig.model_config.model_type
        
        from rtp_llm.utils.fuser import fetch_remote_file_to_local
        tokenizer_path = fetch_remote_file_to_local(tokenizer_path)
        ckpt_path = fetch_remote_file_to_local(ckpt_path)
        
        self.tokenizer = TokenizerFactory.create(ckpt_path, tokenizer_path, model_type)
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id:
            self.config.gpt_init_params.model_config.special_tokens.eos_token_id = self.tokenizer.eos_token_id
            self.config.update_task_prompt_tokens_id(self.tokenizer)

    def is_multimodal(self) -> bool:
        return isinstance(self, MultiModalMixin)

    def _init_database(self):
        self.database = CkptDatabase(self.config.py_model_config.ckpt_path, self.config.py_model_config.ptuning_path)
        # static lora load
        self.static_lora: bool = (
            self.config.lora_infos is not None and len(self.config.lora_infos) == 1
        )
        if self.static_lora:
            for name, path in self.config.lora_infos.items():
                self.database.load_lora(name, path)
            self.database.dump_lora_info()

    def _load_model_weights(self):
        self.weight: ModelWeights = self.model_weights_loader.load_weights(
            device=self.device
        )

    @timer_wrapper(description="load custom module")
    def _load_custom_module(self):
        if self.custom_module is not None:
            self.custom_module.init(self.weight)

    @timer_wrapper(description="load multimodal")
    def _load_multimodal(self):
        if self.config.vit_separation != 2 and self.is_multimodal():
            self.load_mm_weight(
                self.compute_dtype,
                self.config.tp_size,
                self.config.tp_rank,
                self.device,
            )


\

    def create_model_loader(self, parallel_info: ParallelInfo) -> ModelLoader:
        self.parallel_info = parallel_info
        self.device = self.parallel_info.device

        self._init_misc()
        self._init_database()

        tp_rank = self.parallel_info.tp_rank
        tp_size = self.parallel_info.tp_size

        weights_info: ModelDeployWeightInfo = self.get_weight_cls()(
            self.config, tp_size, tp_rank
        )
        misc_weights_info = (
            self.custom_module.get_custom_weight_info() if self.custom_module else []
        )
        return get_model_loader(
            self.task_type,
            weights_info,
            misc_weights_info,
            self.compute_dtype,
            self.database,
            self.is_attn_model,
        )

    @staticmethod
    def eval_model_size(config: GptInitModelParameters):
        # Get task_type from C++ ModelConfig
        task_type_str = config.gpt_init_params.model_config.get_task_type()
        try:
            task_type = TaskType.from_str(task_type_str)
        except:
            task_type = TaskType.LANGUAGE_MODEL
        return config.py_model_config.eval_model_size(
            config.gpt_init_params.model_config.quant_algo_, task_type, config.gpt_init_params.model_config.vocab_size_
        )

    @staticmethod
    def eval_model_param_count(config: GptInitModelParameters):
        return config.py_model_config.model_param_count(config.gpt_init_params.model_config.vocab_size_)
