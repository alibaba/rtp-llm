import logging
from typing import Optional, Union, Any
import json
import os

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.engine_config import EngineConfig
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
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.ops import VitSeparation
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.time_util import timer_wrapper
from rtp_llm.utils.util import to_torch_dtype
from rtp_llm.models.config_wrapper import ConfigWrapper

class BaseModel(object):

    # Independent configuration objects
    model_config: ModelConfig
    engine_config: EngineConfig

    def __init__(
        self,
        model_config: ModelConfig,
        engine_config: EngineConfig,
        vit_config: Optional[VitConfig] = None,
        merge_lora: bool = False,
    ) -> None:
        """Initialize BaseModel with independent configuration objects.
        Args:
            model_config: Model configuration (contains template_type, model_name, lora_infos, mm_model_config)
            engine_config: Engine configuration
            vit_config: Optional VitConfig (needed for multimodal models)
            merge_lora: Whether to merge LoRA weights
        """
        self.model_config = model_config
        self.engine_config = engine_config
        self.vit_config = vit_config
        self.merge_lora = merge_lora

        self.weight = None
        self.tokenizer: Optional[BaseTokenizer] = None
        self.custom_module: Optional[CustomModule] = None
        self.default_generate_config: GenerateConfig = GenerateConfig()
        self.load_tokenizer()

        self.py_model: Optional[GptModelBase] = None

    def load_default_generate_config(self, generate_env_config: Optional[Any] = None):
        """Load default generate config from GenerateEnvConfig.
        
        Args:
            generate_env_config: Optional GenerateEnvConfig object
        """
        if generate_env_config is None:
            return
        generation_config_path = generate_env_config.generation_config_path
        if generation_config_path:
            self.default_generate_config.update(
                json.load(
                    open(os.path.join(generation_config_path, "generation_config.json"))
                )
            )
            logging.info(
                f"load generate config:{generation_config_path}/generation_config.json: \n\
                         {json.dumps(self.default_generate_config.model_dump(), indent=4)}"
            )


    def _get_device_str(self) -> str:
        """Get device string from parallelism_config."""
        return f"cuda:{self.engine_config.parallelism_config.local_rank}"

    @timer_wrapper(description="load model")
    def load(self, parallel_info: ParallelInfo = g_parallel_info):
        if (
            self.engine_config.model_specific_config.load_python_model
            and self.engine_config.hw_kernel_config.enable_cuda_graph
            and self.support_cuda_graph() is False
        ):
            raise Exception("current model can't support cuda graph in py model mode")

        self.model_weights_loader = self.create_model_loader(parallel_info)
        device_str = self._get_device_str()
        self._load(device_str)

        if self.engine_config.model_specific_config.load_python_model:
            logging.info(
                f"Creating python model for {self.model_config.ckpt_path} on {device_str}"
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
            device=device
        )
        self._load_custom_module()
        self._load_multimodal()
        self.model_weights_loader.force_clean_cuda_memory()

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        raise NotImplementedError()

    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        engine_config: EngineConfig,
        parallel_info: ParallelInfo = g_parallel_info,
        vit_config: Optional[VitConfig] = None,
        merge_lora: bool = False,
    ) -> "BaseModel":
        """Create model from independent configuration objects.
        
        Args:
            model_config: Model configuration (contains template_type, model_name, lora_infos, mm_model_config)
            engine_config: Engine configuration
            parallel_info: Parallel information for loading
            vit_config: Optional VitConfig (needed for multimodal models)
            merge_lora: Whether to merge LoRA weights
        """
        # All metadata is in model_config
        model = cls(
            model_config=model_config,
            engine_config=engine_config,
            vit_config=vit_config,
            merge_lora=merge_lora,
        )
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
            self.model_config.mm_model_config.is_multimodal = True
            if self.parallel_info.tp_rank == 0:
                if self.vit_config is None:
                    raise ValueError("vit_config is required for multimodal models")
                # Only initialize multimodal if vit_separation != REMOTE
                vit_separation = self.vit_config.vit_separation
                if vit_separation != VitSeparation.VIT_SEPARATION_REMOTE:
                    self.init_multimodal(
                        mm_model_config=self.model_config.mm_model_config,
                        vit_config=self.vit_config,
                        device=self._get_device_str(),
                    )
    @timer_wrapper(description="init custom_module")
    def _init_misc(self):
        self._may_init_multimodal()
        self.custom_module = self._init_custom_module()

    def _init_custom_module(self) -> Optional[CustomModule]:
        return create_custom_module(self.model_config.task_type, self, self.tokenizer)

    def load_tokenizer(self) -> None:
        # Get tokenizer parameters from config
        ckpt_path = self.model_config.ckpt_path
        tokenizer_path = self.model_config.tokenizer_path
        
        model_type = self.model_config.model_type
        self.tokenizer = TokenizerFactory.create(ckpt_path, tokenizer_path, model_type)
    
        if self.engine_config.kv_cache_config.multi_task_prompt or self.engine_config.kv_cache_config.multi_task_prompt_str:
            self.engine_config.kv_cache_config.load_and_update_task_prompt_config(self.tokenizer)

        if self.tokenizer.eos_token_id:
            self.model_config.special_tokens.eos_token_id = self.tokenizer.eos_token_id

    def is_multimodal(self) -> bool:
        return isinstance(self, MultiModalMixin)

    def _init_database(self):
        self.database = CkptDatabase(self.model_config.ckpt_path, self.model_config.ptuning_path)
        lora_infos = self.model_config.lora_infos
        self.static_lora: bool = len(lora_infos) == 1
        if self.static_lora:
            for name, path in lora_infos.items():
                self.database.load_lora(name, path)
            self.database.dump_lora_info()
                
    def _load_model_weights(self):
        self.weight: ModelWeights = self.model_weights_loader.load_weights(
            device=self._get_device_str()
        )

    @timer_wrapper(description="load custom module")
    def _load_custom_module(self):
        if self.custom_module is not None:
            self.custom_module.init(self.weight)

    @timer_wrapper(description="load multimodal")
    def _load_multimodal(self):
        if self.vit_config is not None and self.vit_config.vit_separation != VitSeparation.VIT_SEPARATION_REMOTE and self.is_multimodal():
            assert isinstance(self, MultiModalMixin)  # for syntax check
            # Convert torch.dtype to string for load_mm_weight
            dtype_str = self.model_config.data_type
            self.load_mm_weight(
                model_config=self.model_config,
                ctype=dtype_str,
                tp_size=self.engine_config.parallelism_config.tp_size,
                tp_rank=self.engine_config.parallelism_config.tp_rank,
                device=self._get_device_str(),
            )

    def create_model_loader(self, parallel_info: ParallelInfo) -> ModelLoader:
        self.parallel_info = parallel_info

        self._init_misc()
        self._init_database()

        vit_weights = None
        if self.model_config.mm_related_params is not None:
            vit_weights = self.model_config.mm_related_params.vit_weights

        weights_info: ModelDeployWeightInfo = self.get_weight_cls()(
            model_config=self.model_config,
            parallelism_config=self.engine_config.parallelism_config,
            hw_kernel_config=self.engine_config.hw_kernel_config,
            kv_cache_config=self.engine_config.kv_cache_config,
            merge_lora=self.merge_lora,
            vit_config=self.vit_config,
            vit_weights=vit_weights,
        )
        misc_weights_info = (
            self.custom_module.get_custom_weight_info() if self.custom_module else []
        )
        return get_model_loader(
            self.model_config,
            weights_info,
            misc_weights_info,
            to_torch_dtype(self.model_config.data_type),
            self.database,
        )

