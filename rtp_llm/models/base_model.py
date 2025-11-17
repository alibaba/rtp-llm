import logging
import os
from typing import Any, Dict, List, NamedTuple, Optional, Union

import torch
from pydantic import BaseModel as PyBaseModel

from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.config.gpt_init_model_parameters import ConfigMode, GptInitModelParameters
from rtp_llm.config.task_type import TaskType
from rtp_llm.distribute.distributed_server import get_world_info
from rtp_llm.distribute.worker_info import ParallelInfo, g_parallel_info
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import (
    BaseTokenizer,
    TokenizerFactory,
)
from rtp_llm.model_loader.loader import ModelLoader, get_model_loader
from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo, ModelWeights
from rtp_llm.model_loader.weight_manager import WeightManager
from rtp_llm.models.downstream_modules.custom_module import CustomModule
from rtp_llm.models.downstream_modules.utils import create_custom_module
from rtp_llm.models.multimodal.multimodal_mixin import MultiModalMixin
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    EmbeddingOutput,
    GenerateContext,
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
    GenerateResponse,
    ModelConfig,
)
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.multimodal_util import MultimodalInput
from rtp_llm.utils.time_util import timer_wrapper
from rtp_llm.utils.util import to_torch_dtype

FT_DEFAULT_MAX_NEW_TOKENS = 2048


class BaseModel(object):

    config: GptInitModelParameters
    vocab_size_padded: int
    device: str

    def __init__(self, config: GptInitModelParameters) -> None:
        self.config = config
        self.weight = None
        self.weight_manager = None

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
        self.weight_manager = WeightManager(
            self.device, self.weight, self.model_weights_loader
        )
        if self.config.model_specific_config.load_python_model:
            logging.info(
                f"Creating python model for {self.config.ckpt_path} on {self.device}"
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
        config: GptInitModelParameters = cls._create_config(model_config.ckpt_path)
        if config.hidden_size == 0:
            config.hidden_size = config.size_per_head * config.head_num
        config.update_common(
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
            ref_module=model_config.ref_module,
            ref_dict=model_config.ref_dict,
            parallel_info=parallel_info,
            world_info=get_world_info(),
            config_mode=config_mode,
        )
        cls._update_config(config)
        return config

    @classmethod
    def _create_config(cls, ckpt_path: str) -> GptInitModelParameters:
        raise NotImplementedError()

    @classmethod
    def _update_config(cls, config: GptInitModelParameters):
        pass

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
        self.task_type = self.config.task_type
        self.custom_module = self._init_custom_module()
        self.compute_dtype: torch.dtype = to_torch_dtype(self.config.data_type)

    def _init_custom_module(self) -> Optional[CustomModule]:
        return create_custom_module(self.task_type, self.config, self.tokenizer)

    def load_tokenizer(self) -> None:
        if self.config:
            self.tokenizer = TokenizerFactory.create_from_config(self.config)
        else:
            self.tokenizer = TokenizerFactory.create_from_env()
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id:
            self.config.special_tokens.eos_token_id = self.tokenizer.eos_token_id
            self.config.update_task_prompt_tokens_id(self.tokenizer)

    def is_multimodal(self) -> bool:
        return isinstance(self, MultiModalMixin)

    def _init_database(self):
        self.database = CkptDatabase(self.config.ckpt_path, self.config.ptuning_path)
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

    def dup_dim0_for_beam_search(
        self, t: torch.Tensor, beam_width: int
    ) -> torch.Tensor:
        shape = list(t.shape)
        return (
            t.unsqueeze(1)
            .repeat([1, beam_width] + [1] * len(shape[1:]))
            .reshape([-1] + shape[1:])
            .contiguous()
        )

    def extend_context_combo_token_types(self, token_types: List[int]) -> List[int]:
        return []

    def extend_generate_combo_token_types(self, combo_tokens: List[int]) -> List[int]:
        return []

    def create_context_position_ids(
        self, input_lengths: Union[List[int], torch.Tensor]
    ):
        return torch.concat(
            [
                torch.arange(int(input_length), dtype=torch.int32)
                for input_length in input_lengths
            ],
            dim=0,
        )

    def create_context_decoder_mask(self, input_lengths: List[int]):
        batch_size = len(input_lengths)
        max_input_length = max(input_lengths)
        attention_mask = torch.ones(
            (max_input_length, max_input_length), dtype=torch.bool, device=self.device
        )
        if self.config.is_causal:
            attention_mask = attention_mask.tril()
        attention_mask = (
            attention_mask.unsqueeze_(0).tile(batch_size, 1, 1).to(self.dtype)
        )
        for b, input_length in enumerate(input_lengths):
            attention_mask[b, input_length:, ...] = 0
            if not self.config.is_causal:
                attention_mask[b, :, input_length:] = 0
        return attention_mask

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
        return config.eval_model_size()

    @staticmethod
    def eval_model_param_count(config: GptInitModelParameters):
        return config.model_param_count
