import json
import logging
import os
from typing import Any, List, Optional, Sequence, Union

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.grammar_tokenizer_info import build_grammar_tokenizer_info_json
from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.output_vocab_config import load_output_vocab_ids
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import (
    BaseTokenizer,
    TokenizerFactory,
)
from rtp_llm.models.downstream_modules.custom_module import CustomModule
from rtp_llm.models.downstream_modules.utils import create_custom_module
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.ops import (
    FMHAConfig,
    HWKernelConfig,
    KVCacheSpecDesc,
    KVCacheSpecType,
    MoeConfig,
    ParallelismConfig,
)
from rtp_llm.utils.model_weight import sp_0_pad8_size
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
        max_generate_batch_size: int,
        load_method: NewLoaderLoadMethod,
        vit_config: Optional[VitConfig],
        merge_lora: bool,
        force_cpu_load_weights: bool = False,
        loader_recycle_handles: bool = False,
        moe_pure_tp_preshard: bool = False,
    ) -> None:
        """Initialize BaseModel with independent configuration objects.
        Args:
            model_config: Model configuration (contains template_type, model_name, lora_infos, mm_model_config)
            parallelism_config: Parallelism configuration
            hw_kernel_config: Hardware kernel configuration
            kv_cache_config: KV cache configuration
            fmha_config: FMHA configuration
            moe_config: MoE configuration
            max_generate_batch_size: Maximum batch size for generation
            merge_lora: Whether to merge LoRA weights
        """
        self.model_config = model_config
        self.parallelism_config = parallelism_config
        self.hw_kernel_config = hw_kernel_config
        self.kv_cache_config = kv_cache_config
        self.fmha_config = fmha_config
        self.moe_config = moe_config
        self.max_generate_batch_size = max_generate_batch_size
        self.load_method = load_method
        self.vit_config = vit_config

        self.merge_lora = merge_lora
        self.force_cpu_load_weights = force_cpu_load_weights
        self.loader_recycle_handles = loader_recycle_handles
        self.moe_pure_tp_preshard = moe_pure_tp_preshard
        self.weight = None
        self.weight_manager = None
        self.rank_local_lora = None
        # Keep the owner alive for the complete lifetime of any non-owning
        # tensor aliases. The alias names are immutable construction metadata;
        # a draft must never silently become an owner during reload/update.
        self._weight_alias_owner: Optional["BaseModel"] = None
        self._weight_alias_names: tuple[str, ...] = ()

        self.linear_bias_slopes: Optional[torch.Tensor] = None
        self.prefix_tokens: Optional[torch.Tensor] = None
        self.py_eplb = None
        self.tokenizer: Optional[BaseTokenizer] = None
        self.custom_module: Optional[CustomModule] = None
        self.py_model = None
        self.checkpoint_owner = None
        self.default_generate_config: GenerateConfig = GenerateConfig()
        self.load_tokenizer()
        self._finalize_output_vocab_config()

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
            generate_config = json.load(
                open(os.path.join(generation_config_path, "generation_config.json"))
            )
            # Checkpoint null means unspecified; request updates remain strict.
            self.default_generate_config.update(
                {
                    key: value
                    for key, value in generate_config.items()
                    if key not in {"max_new_tokens", "min_new_tokens"}
                    or value is not None
                }
            )
            logging.info(
                f"load generate config:{generation_config_path}/generation_config.json: \n\
                         {json.dumps(self.default_generate_config.model_dump(), indent=4)}"
            )

    def _get_device_str(self) -> str:
        """Get device string from parallelism_config."""
        return f"cuda:{self.parallelism_config.local_rank}"

    @timer_wrapper(description="load model")
    def load(self, skip_python_model: bool = False):
        if (
            self.parallelism_config.pp_size > 1
            and not self.model_config.is_mtp
            and not self.support_pp()
        ):
            raise Exception("current model can't support pipeline parallelism")
        if (
            self.hw_kernel_config.enable_cuda_graph
            and self.support_cuda_graph() is False
        ):
            raise Exception("current model can't support cuda graph in py model mode")

        from rtp_llm.models.direct_checkpoint import load_owned_model

        self.custom_module = self._init_custom_module()
        load_owned_model(self, publish_runtime=not skip_python_model)

    def support_pp(self) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        return False

    @classmethod
    def create_config(cls, ckpt_path: str) -> ModelConfig:
        config = cls._create_config(ckpt_path)
        cls._post_build_model_config(config)
        return config

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        raise NotImplementedError()

    @classmethod
    def _apply_kv_cache_config(
        cls, model_config: ModelConfig, kv_cache_config: KVCacheConfig
    ) -> None:
        """Optionally apply parsed KV-cache settings before the legacy hook."""

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        if model_config.kv_cache_spec_descs:
            return

        desc = KVCacheSpecDesc()
        if model_config.attn_config.use_mla:
            desc.cache_type = KVCacheSpecType.MLA
        else:
            desc.cache_type = KVCacheSpecType.MHA

        desc.tag = "default"
        model_config.kv_cache_spec_descs = [
            [desc] for _ in range(model_config.num_layers)
        ]

    @classmethod
    def speculative_weight_alias_names(
        cls, target_model: "BaseModel", draft_model_config: ModelConfig
    ) -> tuple[str, ...]:
        """Declare global weights that this speculative model may borrow.

        The default is deliberately empty.  A draft model must opt in and
        validate semantic compatibility with the concrete target owner before
        the loader skips any checkpoint tensors.  ``from_config`` retains a
        strong reference to ``target_model`` for the complete alias lifetime.
        """
        return ()

    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        hw_kernel_config: HWKernelConfig,
        kv_cache_config: KVCacheConfig,
        fmha_config: FMHAConfig,
        moe_config: MoeConfig,
        load_method,
        max_generate_batch_size: int,
        vit_config: Optional[VitConfig],
        merge_lora: bool,
        force_cpu_load_weights: bool = False,
        skip_python_model: bool = False,
        loader_recycle_handles: bool = False,
        moe_pure_tp_preshard: bool = False,
        weight_alias_owner: Optional["BaseModel"] = None,
        weight_alias_names: Sequence[str] = (),
        reserve_runtime_mem_mb: int = 0,
    ) -> "BaseModel":
        """Create model from independent configuration objects.

        Args:
            model_config: Model configuration (contains template_type, model_name, lora_infos, mm_model_config)
            parallelism_config: Parallelism configuration
            hw_kernel_config: Hardware kernel configuration
            kv_cache_config: KV cache configuration
            fmha_config: FMHA configuration
            moe_config: MoE configuration
            max_generate_batch_size: Maximum batch size for generation
            merge_lora: Whether to merge LoRA weights
        """
        # All metadata is in model_config
        model = cls(
            model_config=model_config,
            parallelism_config=parallelism_config,
            hw_kernel_config=hw_kernel_config,
            kv_cache_config=kv_cache_config,
            fmha_config=fmha_config,
            moe_config=moe_config,
            load_method=load_method,
            max_generate_batch_size=max_generate_batch_size,
            vit_config=vit_config,
            merge_lora=merge_lora,
            force_cpu_load_weights=force_cpu_load_weights,
            loader_recycle_handles=loader_recycle_handles,
            moe_pure_tp_preshard=moe_pure_tp_preshard,
        )
        if weight_alias_names and weight_alias_owner is None:
            raise ValueError(
                "weight_alias_names require an explicit weight_alias_owner"
            )
        model._weight_alias_owner = weight_alias_owner
        model.reserve_runtime_mem_mb = reserve_runtime_mem_mb
        model._weight_alias_names = tuple(weight_alias_names)

        import os

        import psutil

        def get_host_memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB

        # 在加载前后分别记录内存使用
        logging.info(f"Before loading: {get_host_memory_usage():.2f} MB")
        model.load(skip_python_model=skip_python_model)
        logging.info(f"After loading: {get_host_memory_usage():.2f} MB")
        return model

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        assert self.weight is not None
        return self.weight.dtype

    def _init_custom_module(self) -> Optional[CustomModule]:
        return create_custom_module(self.model_config, self.tokenizer)

    def load_tokenizer(self) -> None:
        # Get tokenizer parameters from config
        ckpt_path = self.model_config.ckpt_path
        tokenizer_path = self.model_config.tokenizer_path
        model_type = self.model_config.model_type
        self.tokenizer = TokenizerFactory.create(ckpt_path, tokenizer_path, model_type)
        self._fill_tokenizer_special_tokens()

    def _fill_tokenizer_special_tokens(self) -> None:
        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        if eos_token_id is not None:
            self.model_config.special_tokens.eos_token_id = int(eos_token_id)

    def build_grammar_tokenizer_info(self) -> str:
        # Grammar tokenizer metadata is a startup-time compatibility contract, not
        # an optional per-request optimization.  Fail fast below if it cannot be
        # built: accepting ordinary requests with empty metadata would defer an
        # unsupported tokenizer or missing stop-token configuration until the
        # first grammar request reaches the engine.
        real_tokenizer = self.tokenizer.get_real_tokenizer()
        if real_tokenizer is None:
            return ""

        try:
            return build_grammar_tokenizer_info_json(
                real_tokenizer,
                model_vocab_size=int(self.model_config.vocab_size or 0),
                stop_token_ids=self._collect_tokenizer_info_stop_token_ids(),
            )
        except Exception as e:
            message = f"Failed to build grammar tokenizer metadata from tokenizer: {e}"
            logging.warning(message)
            raise RuntimeError(message) from e

    def _collect_tokenizer_info_stop_token_ids(self) -> List[int]:
        ids: List[int] = []

        def add_id(token_id: int) -> None:
            if token_id < 0:
                return
            if token_id not in ids:
                ids.append(token_id)

        special_tokens = self.model_config.special_tokens
        add_id(int(special_tokens.eos_token_id))
        for token_ids in special_tokens.stop_words_id_list:
            if len(token_ids) == 1:
                add_id(int(token_ids[0]))
        ids.sort()
        return ids

    def _finalize_output_vocab_config(self) -> None:
        if not self.model_config.enable_output_vocab_pruning:
            self.model_config.output_vocab_ids = []
            self.model_config.output_vocab_padded_size = 0
            return
        if not self.model_config.has_lm_head:
            raise ValueError("output vocabulary pruning requires a model LM head")

        eos_token_id = self.model_config.special_tokens.eos_token_id
        output_vocab_ids = load_output_vocab_ids(
            self.model_config.ckpt_path,
            self.model_config.vocab_size,
            self.model_config.input_vocab_size,
            tokenizer=self.tokenizer.get_real_tokenizer(),
            extra_token_ids=(eos_token_id,),
        )
        self.model_config.output_vocab_ids = output_vocab_ids

        is_distributed = (
            self.parallelism_config.tp_size > 1
            or self.parallelism_config.dp_size > 1
            or self.parallelism_config.ep_size > 1
        )
        self.model_config.output_vocab_padded_size = (
            sp_0_pad8_size(len(output_vocab_ids), self.parallelism_config.tp_size)
            if is_distributed
            else len(output_vocab_ids)
        )
        logging.info(
            "finalized output vocabulary: V=%d, K=%d, P=%d, eos_token_id=%d",
            self.model_config.vocab_size,
            len(output_vocab_ids),
            self.model_config.output_vocab_padded_size,
            eos_token_id,
        )

    def is_multimodal(self) -> bool:
        return self.model_config.mm_model_config.is_multimodal
