import logging
from typing import Optional, Union

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
from rtp_llm.ops import MMModelConfig
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.time_util import timer_wrapper
from rtp_llm.utils.util import to_torch_dtype

class BaseModel(object):

    # Independent configuration objects
    py_model_config: ModelConfig
    mm_model_config: MMModelConfig
    engine_config: EngineConfig

    def __init__(
        self,
        py_model_config: ModelConfig,
        mm_model_config: MMModelConfig,
        engine_config: EngineConfig,
        vit_config: Optional[VitConfig] = None,
        merge_lora: bool = False,
    ) -> None:
        """Initialize BaseModel with independent configuration objects.
        Args:
            py_model_config: Model configuration (contains template_type, model_name, lora_infos)
            mm_model_config: Multimodal model configuration
            engine_config: Engine configuration
            vit_config: Optional VitConfig (needed for multimodal models)
            merge_lora: Whether to merge LoRA weights
        """
        self.py_model_config = py_model_config
        self.mm_model_config = mm_model_config
        self.engine_config = engine_config
        self.vit_config = vit_config
        self.merge_lora = merge_lora

        self.weight = None
        self.tokenizer: Optional[BaseTokenizer] = None
        self.custom_module: Optional[CustomModule] = None
        self.default_generate_config: GenerateConfig = GenerateConfig()
        self.load_tokenizer()

        self.py_model: Optional[GptModelBase] = None


    @timer_wrapper(description="load model")
    def load(self, parallel_info: ParallelInfo = g_parallel_info):
        if (
            self.engine_config.model_specific_config.load_python_model
            and self.engine_config.hw_kernel_config.enable_cuda_graph
            and self.support_cuda_graph() is False
        ):
            raise Exception("current model can't support cuda graph in py model mode")

        self.model_weights_loader = self.create_model_loader(parallel_info)
        self._load(parallel_info.device)

        if self.engine_config.model_specific_config.load_python_model:
            logging.info(
                f"Creating python model for {self.py_model_config.ckpt_path} on {parallel_info.device}"
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
        py_model_config: ModelConfig,
        mm_model_config: MMModelConfig,
        engine_config: EngineConfig,
        parallel_info: ParallelInfo = g_parallel_info,
        vit_config: Optional[VitConfig] = None,
        merge_lora: bool = False,
    ) -> "BaseModel":
        """Create model from independent configuration objects.
        
        Args:
            py_model_config: Model configuration (contains template_type, model_name, lora_infos)
            mm_model_config: Multimodal model configuration
            engine_config: Engine configuration
            parallel_info: Parallel information for loading
            vit_config: Optional VitConfig (needed for multimodal models)
            merge_lora: Whether to merge LoRA weights
        """
        # All metadata is in py_model_config
        model = cls(
            py_model_config=py_model_config,
            mm_model_config=mm_model_config,
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
    def config(self):
        """Configuration wrapper that combines py_model_config, engine_config, and vit_config.
        
        This property provides a unified interface for accessing all configuration objects
        needed by AsyncModel, RPCEngine, and other components.
        """
        # Create a simple wrapper object that combines all configs
        class ConfigWrapper:
            def __init__(self, base_model):
                self.base_model = base_model
                # Direct access to config objects
                self.py_model_config = base_model.py_model_config
                self.mm_model_config = base_model.mm_model_config
                self.engine_config = base_model.engine_config
                self.vit_config = base_model.vit_config
                
                # Expose engine_config sub-configs for compatibility
                self.parallelism_config = base_model.engine_config.parallelism_config
                self.runtime_config = base_model.engine_config.runtime_config
                self.pd_sep_config = base_model.engine_config.pd_sep_config
                self.concurrency_config = base_model.engine_config.concurrency_config
                self.fmha_config = base_model.engine_config.fmha_config
                self.kv_cache_config = base_model.engine_config.kv_cache_config
                self.profiling_debug_logging_config = base_model.engine_config.profiling_debug_logging_config
                self.hw_kernel_config = base_model.engine_config.hw_kernel_config
                self.device_resource_config = base_model.engine_config.device_resource_config
                self.moe_config = base_model.engine_config.moe_config
                self.model_specific_config = base_model.engine_config.model_specific_config
                self.sp_config = base_model.engine_config.sp_config
                self.cache_store_config = base_model.engine_config.cache_store_config
                self.misc_config = base_model.engine_config.misc_config
                self.arpc_config = base_model.engine_config.arpc_config
                self.ffn_disaggregate_config = base_model.engine_config.parallelism_config.ffn_disaggregate_config
                
                # Expose commonly accessed attributes
                self.special_tokens = base_model.py_model_config.special_tokens
                self.max_seq_len = base_model.py_model_config.max_seq_len
                self.role_type = base_model.engine_config.pd_sep_config.role_type
                self.is_multimodal = base_model.mm_model_config.is_multimodal
        
        if not hasattr(self, '_config_wrapper'):
            self._config_wrapper = ConfigWrapper(self)
        return self._config_wrapper

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        assert self.weight is not None
        return self.weight.dtype

    @timer_wrapper(description="init mutlimodal")
    def _may_init_multimodal(self):
        if self.is_multimodal():
            assert isinstance(self, MultiModalMixin)  # for syntax check
            self.mm_model_config.is_multimodal_ = True
            if self.parallel_info.tp_rank == 0:
                if self.vit_config is None:
                    raise ValueError("vit_config is required for multimodal models")
                # Only initialize multimodal if vit_separation != 2
                vit_separation = self.engine_config.runtime_config.vit_separation
                if vit_separation != 2:
                    self.init_multimodal(
                        mm_model_config=self.mm_model_config,
                        vit_config=self.vit_config,
                        device=self.parallel_info.device,
                    )
    @timer_wrapper(description="init custom_module")
    def _init_misc(self):
        self._may_init_multimodal()
        self.custom_module = self._init_custom_module()

    def _init_custom_module(self) -> Optional[CustomModule]:
        return create_custom_module(self.py_model_config.task_type, self, self.tokenizer)

    def load_tokenizer(self) -> None:
        # Get tokenizer parameters from config
        ckpt_path = self.py_model_config.ckpt_path
        tokenizer_path = self.py_model_config.tokenizer_path
        
        # Get model_type from config.json or py_model_config
        import json
        import os
        model_type = ""
        # First try to get from py_model_config.model_type
        if self.py_model_config.model_type:
            model_type = self.py_model_config.model_type.lower()
        
        # If not found, try to get from config.json
        if not model_type:
            config_json_path = os.path.join(ckpt_path, "config.json")
            if os.path.exists(config_json_path):
                with open(config_json_path, "r", encoding="utf-8") as reader:
                    config_json = json.loads(reader.read())
                    # Try to get model_type from architectures field
                    if "architectures" in config_json and config_json["architectures"]:
                        model_type = config_json["architectures"][0].lower()
                    elif "model_type" in config_json:
                        model_type = config_json["model_type"].lower()
        
        # model_type should be found in config.json or py_model_config
        if not model_type:
            raise ValueError("model_type not found in config.json and cannot be determined")

        self.tokenizer = TokenizerFactory.create(ckpt_path, tokenizer_path, model_type)
    
        # Load task prompt config and update token IDs if tokenizer is available
        # Ensure kv_cache_config is Python KVCacheConfig instance (not C++ object)
        if not isinstance(self.engine_config.kv_cache_config, KVCacheConfig):
            # Replace C++ object with Python instance
            cpp_config = self.engine_config.kv_cache_config
            python_config = KVCacheConfig()
            # Copy all attributes from C++ object to Python object
            for attr in dir(cpp_config):
                if not attr.startswith('_') and hasattr(cpp_config, attr):
                    try:
                        setattr(python_config, attr, getattr(cpp_config, attr))
                    except (AttributeError, TypeError):
                        pass
            self.engine_config.kv_cache_config = python_config
        
        if self.engine_config.kv_cache_config.multi_task_prompt or self.engine_config.kv_cache_config.multi_task_prompt_str:
            self.engine_config.kv_cache_config.load_and_update_task_prompt_config(self.tokenizer)

        if self.tokenizer.eos_token_id:
            self.py_model_config.special_tokens.eos_token_id = self.tokenizer.eos_token_id

    def is_multimodal(self) -> bool:
        return isinstance(self, MultiModalMixin)

    def _init_database(self):
        self.database = CkptDatabase(self.py_model_config.ckpt_path, self.py_model_config.ptuning_path)
        lora_infos = self.py_model_config.lora_infos
        self.static_lora: bool = len(lora_infos) == 1
        if self.static_lora:
            for name, path in lora_infos.items():
                self.database.load_lora(name, path)
            self.database.dump_lora_info()
                
    def _load_model_weights(self):
        self.weight: ModelWeights = self.model_weights_loader.load_weights(
            device=self.parallel_info.device
        )

    @timer_wrapper(description="load custom module")
    def _load_custom_module(self):
        if self.custom_module is not None:
            self.custom_module.init(self.weight)

    @timer_wrapper(description="load multimodal")
    def _load_multimodal(self):
        if self.engine_config.runtime_config.vit_separation != 2 and self.is_multimodal():
            assert isinstance(self, MultiModalMixin)  # for syntax check
            # Convert torch.dtype to string for load_mm_weight
            dtype_str = self.py_model_config.data_type
            self.load_mm_weight(
                py_model_config=self.py_model_config,
                mm_model_config=self.mm_model_config,
                ctype=dtype_str,
                tp_size=self.engine_config.parallelism_config.tp_size,
                tp_rank=self.engine_config.parallelism_config.tp_rank,
                device=self.parallel_info.device,
            )

    def create_model_loader(self, parallel_info: ParallelInfo) -> ModelLoader:
        self.parallel_info = parallel_info

        self._init_misc()
        self._init_database()

        tp_rank = self.parallel_info.tp_rank
        tp_size = self.parallel_info.tp_size

        weights_info: ModelDeployWeightInfo = self.get_weight_cls()(
            self.py_model_config, self.engine_config, self.merge_lora, tp_size, tp_rank
        )
        misc_weights_info = (
            self.custom_module.get_custom_weight_info() if self.custom_module else []
        )
        return get_model_loader(
            self.py_model_config,
            weights_info,
            misc_weights_info,
            to_torch_dtype(self.py_model_config.data_type),
            self.database,
        )

