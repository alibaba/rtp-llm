import json
import logging
import os
from typing import Any, Optional, Type, Union

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import (
    BaseTokenizer,
    TokenizerFactory,
)
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.loader import ModelLoader, get_model_loader
from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo, ModelWeights
from rtp_llm.model_loader.weight_manager import WeightManager
from rtp_llm.models.downstream_modules.custom_module import CustomModule
from rtp_llm.models.downstream_modules.utils import create_custom_module
from rtp_llm.models.multimodal.multimodal_mixin import MultiModalMixin
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import (
    DeviceResourceConfig,
    FMHAConfig,
    HWKernelConfig,
    MoeConfig,
    ParallelismConfig,
    ProfilingDebugLoggingConfig,
    VitSeparation,
)
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.time_util import timer_wrapper


class BaseModel(object):

    # Independent configuration objects
    model_config: ModelConfig
    parallelism_config: ParallelismConfig
    hw_kernel_config: HWKernelConfig
    kv_cache_config: KVCacheConfig
    fmha_config: FMHAConfig
    moe_config: MoeConfig

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        hw_kernel_config: HWKernelConfig,
        kv_cache_config: KVCacheConfig,
        fmha_config: FMHAConfig,
        moe_config: MoeConfig,
        load_python_model: bool,
        max_generate_batch_size: int,
        load_method: LoadMethod,
        vit_config: Optional[VitConfig],
        merge_lora: bool,
        device_resource_config: Optional[DeviceResourceConfig],
    ) -> None:
        """Initialize BaseModel with independent configuration objects.
        Args:
            model_config: Model configuration (contains template_type, model_name, lora_infos, mm_model_config)
            parallelism_config: Parallelism configuration
            hw_kernel_config: Hardware kernel configuration
            kv_cache_config: KV cache configuration
            fmha_config: FMHA configuration
            moe_config: MoE configuration
            load_python_model: Whether to load Python model (instead of C++ GptModel)
            max_generate_batch_size: Maximum batch size for generation
            vit_config: Optional VitConfig (needed for multimodal models)
            merge_lora: Whether to merge LoRA weights
            device_resource_config: Optional DeviceResourceConfig for device resource configuration
        """
        self.model_config = model_config
        self.parallelism_config = parallelism_config
        self.hw_kernel_config = hw_kernel_config
        self.kv_cache_config = kv_cache_config
        self.fmha_config = fmha_config
        self.moe_config = moe_config
        self.load_python_model = load_python_model
        self.max_generate_batch_size = max_generate_batch_size
        self.load_method = load_method
        self.vit_config = vit_config
        self.merge_lora = merge_lora
        self.device_resource_config = device_resource_config
        self.weight = None
        self.weight_manager = None

        self.linear_bias_slopes: Optional[torch.Tensor] = None
        self.prefix_tokens: Optional[torch.Tensor] = None
        self.py_eplb = None
        self.tokenizer: Optional[BaseTokenizer] = None
        self.custom_module: Optional[CustomModule] = None
        self.py_model: Optional[GptModelBase] = None
        self.default_generate_config: GenerateConfig = GenerateConfig()
        self.load_tokenizer()

        if (
            self.kv_cache_config.multi_task_prompt
            or self.kv_cache_config.multi_task_prompt_str
        ):
            self.kv_cache_config.load_and_update_task_prompt_config(self.tokenizer)

        if self.model_config.generate_env_config:
            self.load_default_generate_config(self.model_config.generate_env_config)

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
        return f"cuda:{self.parallelism_config.local_rank}"

    @timer_wrapper(description="load model")
    def load(self):
        if (
            self.load_python_model
            and self.hw_kernel_config.enable_cuda_graph
            and self.support_cuda_graph() is False
        ):
            raise Exception("current model can't support cuda graph in py model mode")

        self._may_init_multimodal()
        self.custom_module = self._init_custom_module()

        self.model_weights_loader = self.create_model_loader()
        self.py_eplb = self.model_weights_loader._py_eplb
        device_str = self._get_device_str()
        self._load(device_str)
        self.weight_manager = WeightManager(
            self.device, self.weight, self.model_weights_loader
        )
        if self.load_python_model:
            logging.info(
                f"Creating python model for {self.model_config.ckpt_path} on {device_str}"
            )
            remote_jit_dir = os.environ.get("REMOTE_JIT_DIR", None)
            logging.info(f"python model remote_jit_dir for deep_gemm: {remote_jit_dir}")
            if remote_jit_dir:
                os.environ["DG_JIT_REMOTE_CACHE_DIR"] = os.path.join(
                    remote_jit_dir, "deep_gemm_python"
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
        # record device string for later use (e.g., WeightManager, python model init)
        self.device = device
        self.weight: ModelWeights = self.model_weights_loader.load_weights(
            device=device
        )
        self._load_custom_module()
        self._load_multimodal()
        self.model_weights_loader.force_clean_cuda_memory()

    @classmethod
    def create_config(cls, ckpt_path: str) -> ModelConfig:
        config = cls._create_config(ckpt_path)
        if cls.is_multimodal():
            cls.init_model_weight_evaluator(config)
        return config

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        raise NotImplementedError()

    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        hw_kernel_config: HWKernelConfig,
        kv_cache_config: KVCacheConfig,
        fmha_config: FMHAConfig,
        moe_config: MoeConfig,
        load_python_model: bool,
        load_method: LoadMethod,
        max_generate_batch_size: int,
        vit_config: VitConfig,
        merge_lora: bool,
        device_resource_config: DeviceResourceConfig,
    ) -> "BaseModel":
        """Create model from independent configuration objects.

        Args:
            model_config: Model configuration (contains template_type, model_name, lora_infos, mm_model_config)
            parallelism_config: Parallelism configuration
            hw_kernel_config: Hardware kernel configuration
            kv_cache_config: KV cache configuration
            fmha_config: FMHA configuration
            moe_config: MoE configuration
            load_python_model: Whether to load Python model (instead of C++ GptModel)
            max_generate_batch_size: Maximum batch size for generation
            vit_config: VitConfig (needed for multimodal models)
            merge_lora: Whether to merge LoRA weights
            device_resource_config: DeviceResourceConfig for device resource configuration
        """
        # All metadata is in model_config
        model = cls(
            model_config=model_config,
            parallelism_config=parallelism_config,
            hw_kernel_config=hw_kernel_config,
            kv_cache_config=kv_cache_config,
            fmha_config=fmha_config,
            moe_config=moe_config,
            load_python_model=load_python_model,
            load_method=load_method,
            max_generate_batch_size=max_generate_batch_size,
            vit_config=vit_config,
            merge_lora=merge_lora,
            device_resource_config=device_resource_config,
        )
        model.load()
        return model

    @staticmethod
    def get_weight_cls() -> Type[ModelDeployWeightInfo]:
        raise NotImplementedError

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        assert self.weight is not None
        return self.weight.dtype

    @timer_wrapper(description="init mutlimodal")
    def _may_init_multimodal(self):
        if not self.is_multimodal():
            return

        assert isinstance(self, MultiModalMixin)  # for syntax check
        self.model_config.mm_model_config.is_multimodal = True
        if self.parallelism_config.tp_rank != 0:
            return

        if self.vit_config is None:
            raise ValueError("vit_config is required for multimodal models")
        # Only initialize multimodal if vit_separation != REMOTE
        vit_separation = self.vit_config.vit_separation
        if vit_separation != VitSeparation.VIT_SEPARATION_REMOTE:
            self.init_multimodal(
                device=self._get_device_str(),
            )

    def _init_custom_module(self) -> Optional[CustomModule]:
        return create_custom_module(self.model_config, self.tokenizer)

    def load_tokenizer(self) -> None:
        # Get tokenizer parameters from config
        ckpt_path = self.model_config.ckpt_path
        tokenizer_path = self.model_config.tokenizer_path
        model_type = self.model_config.model_type
        self.tokenizer = TokenizerFactory.create(ckpt_path, tokenizer_path, model_type)
        if self.tokenizer.eos_token_id:
            self.model_config.special_tokens.eos_token_id = self.tokenizer.eos_token_id

    @classmethod
    def is_multimodal(cls) -> bool:
        return issubclass(cls, MultiModalMixin)

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
        if (
            self.vit_config is not None
            and self.vit_config.vit_separation != VitSeparation.VIT_SEPARATION_REMOTE
            and self.is_multimodal()
        ):
            assert isinstance(self, MultiModalMixin)  # for syntax check
            # Convert torch.dtype to string for load_mm_weight
            dtype_str = self.model_config.data_type
            self.load_mm_weight(
                model_config=self.model_config,
                ctype=dtype_str,
                tp_size=self.parallelism_config.tp_size,
                tp_rank=self.parallelism_config.tp_rank,
                device=self._get_device_str(),
            )

    def create_model_loader(self) -> ModelLoader:
        # Create database locally, only used for model loading
        database = CkptDatabase(
            self.model_config.ckpt_path, self.model_config.ptuning_path
        )
        lora_infos = self.model_config.lora_infos
        static_lora: bool = len(lora_infos) == 1
        if static_lora:
            for name, path in lora_infos.items():
                database.load_lora(name, path)
            database.dump_lora_info()

        vit_weights = None
        if self.model_config.mm_related_params is not None:
            vit_weights = self.model_config.mm_related_params.vit_weights

        weights_info: ModelDeployWeightInfo = self.get_weight_cls()(
            model_config=self.model_config,
            parallelism_config=self.parallelism_config,
            hw_kernel_config=self.hw_kernel_config,
            kv_cache_config=self.kv_cache_config,
            merge_lora=self.merge_lora,
            vit_config=self.vit_config,
            vit_weights=vit_weights,
            load_method=self.load_method,
        )
        misc_weights_info = (
            self.custom_module.get_custom_weight_info() if self.custom_module else []
        )
        return get_model_loader(
            self.model_config,
            weights_info,
            misc_weights_info,
            database,
            load_method=self.load_method,
        )
