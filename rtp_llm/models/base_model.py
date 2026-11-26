import hashlib
import importlib.metadata
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.grammar_tokenizer_info import build_grammar_tokenizer_info_json
from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.output_vocab_config import load_output_vocab_ids
from rtp_llm.config.py_config_modules import ResidentWeightsConfig, VitConfig
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
from rtp_llm.resident_weights import ResidentWeights, checkpoint_digest, vmm
from rtp_llm.resident_weights.store import StoreSettings
from rtp_llm.resident_weights.weights import agree_resident_step
from rtp_llm.utils.model_weight import sp_0_pad8_size
from rtp_llm.utils.time_util import timer_wrapper


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


class BaseModel(object):

    resident_weights_env_vars = ("MOE_MEGA_SHARED_EXPERT", "RTP_LLM_FP4_GEMM_BACKEND")

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
        resident_weights_config: Optional[ResidentWeightsConfig] = None,
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
        self.resident_weights_config = (
            resident_weights_config or ResidentWeightsConfig()
        )
        self._resident_weights: Optional[ResidentWeights] = None
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
        self._finish_resident_weights()

    def support_pp(self) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        return False

    def _load_with_residency(self, device, load, *, group=None):
        """Agree on the source before any loader enters a checkpoint collective."""
        if not self.resident_weights_config.enable:
            return load()
        if (
            self.resident_weights_config.role == "draft"
            and self.parallelism_config.pp_size > 1
            and torch.distributed.is_initialized()
            and group is None
        ):
            raise RuntimeError("A distributed PP draft requires its checkpoint stage group")
        manager = None
        error = None
        try:
            manager = self._create_resident_weights()
        except Exception as exc:
            error = exc
        # Use exactly the loader's participants for *both* decisions. Otherwise
        # a warm rank may replay its owner while a peer enters FST cold-load collectives.
        if not agree_resident_step(int(manager is not None), error, device, group, "setup"):
            if manager is not None:
                # Every rank now loads from the checkpoint. A record this rank published
                # earlier would otherwise keep its memory alongside the new copy.
                logging.warning(
                    "resident weights %s: another rank cannot use resident weights; "
                    "dropping this rank's record and loading from the checkpoint",
                    manager.slot_key,
                )
                manager.discard_attach()
            manager = None
        self._resident_weights = manager
        restored = None if manager is None else manager.acquire_for_load(
            device, synchronize_ranks=True, group=group
        )
        owner = None
        try:
            owner = load() if manager is None else load(resident=manager, restored=restored)
        except Exception as exc:
            error = exc
        # This vote replaces the success-only barrier. Scratch/isolated or
        # resident replay can fail on one rank without synchronizing internally.
        agree_resident_step(1, error, device, group, "load")
        return owner

    @property
    def resident_initialize_gpu_growth_bytes(self) -> int:
        """GPU memory the attached record gained in its cold start's initialize(), else 0.

        This start holds it before the engine sizes the KV cache; see
        KVCacheConfig.target_initialize_gpu_growth_bytes.
        """
        manager = self._resident_weights
        return 0 if manager is None else manager.initialize_gpu_growth_bytes

    def _finish_resident_weights(self) -> None:
        if self._resident_weights is not None:
            self._resident_weights.finalize_after_initialize(self.py_model)

    def _create_resident_weights(self) -> Optional[ResidentWeights]:
        if not vmm.is_available(self.parallelism_config.local_rank):
            logging.warning(
                "resident weights disabled: CUDA VMM with POSIX FD export is unavailable"
            )
            return None
        if self.model_config.eplb_config.enable_eplb():
            # The expert balancer copies reloaded experts into the weight tensors in place,
            # which read-only resident mappings would fault on.
            logging.warning("resident weights disabled: not supported with EPLB")
            return None
        offload_after_s, idle_ttl_s = self.resident_weights_config.store_timers()
        host_reserve_ratio, host_reserve_min_gb = (
            self.resident_weights_config.store_host_reserve()
        )
        resident = ResidentWeights(
            self.resident_weights_config.role,
            self.parallelism_config.local_rank,
            self._resident_fingerprint(),
            self.resident_weights_config.verify,
            StoreSettings(
                offload_after_s=offload_after_s,
                idle_ttl_s=idle_ttl_s,
                host_reserve_ratio=host_reserve_ratio,
                host_reserve_min_gb=host_reserve_min_gb,
            ),
        )
        return resident if resident.ensure_store_running() else None

    def _resident_fingerprint(self) -> Dict[str, str]:
        """Registered configuration guards, computed before loading mutates any config.

        This is not a content or transitive-code dependency identity. Users must drop records
        after incompatible checkpoint, weight-processing code or dependency changes.
        """
        config = self.model_config
        fingerprint = {
            "model_type": config.model_type,
            "owner_layout": "newloader/native-v3",
            "load_method": str(self.load_method),
            "force_cpu_load_weights": str(self.force_cpu_load_weights),
            "fmha_config": self.fmha_config.to_string() if self.fmha_config is not None else "",
            "max_generate_batch_size": str(self.max_generate_batch_size),
            "checkpoint_path": os.path.realpath(config.ckpt_path),
            "checkpoint_files": checkpoint_digest(config.ckpt_path, follow_index=True),
            "ptuning_path": str(config.ptuning_path),
            # The loader ignores absent/non-directory optional ptuning paths.
            "ptuning_files": (
                checkpoint_digest(config.ptuning_path)
                if config.ptuning_path and os.path.isdir(config.ptuning_path)
                else ""
            ),
            "lora_infos": str(config.lora_infos),
            "lora_files": (
                repr(
                    sorted(
                        (name, checkpoint_digest(path))
                        for name, path in config.lora_infos.items()
                    )
                )
                if self.merge_lora and len(config.lora_infos) == 1
                else ""
            ),
            "enable_fp32_lm_head": str(config.enable_fp32_lm_head),
            "normalize_lm_head_weight": str(config.normalize_lm_head_weight),
            "output_vocab_ids": hashlib.sha256(
                json.dumps(list(config.output_vocab_ids)).encode()
            ).hexdigest(),
            "merge_lora": str(self.merge_lora),
            "moe_pure_tp_preshard": str(self.moe_pure_tp_preshard),
            # The requested recipe is Python-only, so to_string() below omits it, and models
            # derive storage choices (e.g. Qwen4 dense_storage) the C++ quant_algo cannot show.
            "quantization": str(config.quantization),
            # Borrowed globals are left out of this record; a different set leaves a hit
            # missing or overriding them.
            "global_weight_aliases": repr(sorted(self._weight_alias_names)),
            "model_config": config.to_string(),
            "parallelism_config": self.parallelism_config.to_string(),
            "hw_kernel_config": self.hw_kernel_config.to_string(),
            "moe_config": self.moe_config.to_string(),
            # Cheap environment guards, not a complete identity of all weight-producing code.
            "libraries": " ".join(
                f"{name}={_package_version(name)}" for name in ("torch", "deep_gemm")
            ),
        }
        if config.phy2log_path:
            # The expert placement table is a file the path alone does not pin down.
            with open(config.phy2log_path, "rb") as table:
                fingerprint["phy2log_file"] = hashlib.sha256(table.read()).hexdigest()
        for name in self.resident_weights_env_vars:
            fingerprint[f"env {name}"] = repr(os.environ.get(name))
        return fingerprint

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
        resident_weights_config: Optional[ResidentWeightsConfig] = None,
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
            resident_weights_config=resident_weights_config,
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
