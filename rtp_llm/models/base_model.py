import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Type, Union

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
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.downstream_modules.custom_module import CustomModule
from rtp_llm.models.downstream_modules.utils import create_custom_module
from rtp_llm.ops import (
    DeviceResourceConfig,
    FMHAConfig,
    HWKernelConfig,
    MoeConfig,
    ParallelismConfig,
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
        max_generate_batch_size: int,
        load_method: LoadMethod,
        vit_config: Optional[VitConfig],
        merge_lora: bool,
        device_resource_config: Optional[DeviceResourceConfig],
        force_cpu_load_weights: bool = False,
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
            device_resource_config: Optional DeviceResourceConfig for device resource configuration
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
        self.device_resource_config = device_resource_config
        self.force_cpu_load_weights = force_cpu_load_weights
        self.weight = None
        self.weight_manager = None

        self.linear_bias_slopes: Optional[torch.Tensor] = None
        self.prefix_tokens: Optional[torch.Tensor] = None
        self.py_eplb = None
        self.tokenizer: Optional[BaseTokenizer] = None
        self.custom_module: Optional[CustomModule] = None
        self.py_model = None
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
    def load(self, skip_python_model: bool = False):
        if (
            self.hw_kernel_config.enable_cuda_graph
            and self.support_cuda_graph() is False
        ):
            raise Exception("current model can't support cuda graph in py model mode")

        if self._use_new_loader():
            self._load_with_new_loader()
            return

        self.custom_module = self._init_custom_module()
        self.model_weights_loader = self.create_model_loader()
        self.py_eplb = self.model_weights_loader._py_eplb
        device_str = self._get_device_str()
        self._load(device_str)
        self.weight_manager = WeightManager(
            self.device, self.weight, self.model_weights_loader
        )
        if skip_python_model:
            return
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

    def _create_python_model(self):
        pass

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

        # 清理checkpoint加载过程中使用的临时资源，释放host内存
        self._cleanup_loader_resources()

        self.model_weights_loader.force_clean_all_memory()

    def _cleanup_loader_resources(self):
        """清理模型加载过程中使用的临时资源，释放host内存

        在模型权重加载完成后调用此方法，释放以下资源：
        1. CkptDatabase 中的 CkptFileInfo 元数据
        2. checkpoint文件列表
        3. 临时的LoRA缓存

        这可以显著减少host内存占用，为KV cache等运行时内存需求腾出空间。
        """
        self.model_weights_loader.cleanup_database()

    @classmethod
    def create_config(cls, ckpt_path: str) -> ModelConfig:
        config = cls._create_config(ckpt_path)
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
        load_method,
        max_generate_batch_size: int,
        vit_config: Optional[VitConfig],
        merge_lora: bool,
        device_resource_config: DeviceResourceConfig,
        force_cpu_load_weights: bool = False,
        skip_python_model: bool = False,
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
            load_method=load_method,
            max_generate_batch_size=max_generate_batch_size,
            vit_config=vit_config,
            merge_lora=merge_lora,
            device_resource_config=device_resource_config,
            force_cpu_load_weights=force_cpu_load_weights,
        )

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

    @staticmethod
    def get_weight_cls() -> Type[ModelDeployWeightInfo]:
        raise NotImplementedError

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
        if self.tokenizer.eos_token_id:
            self.model_config.special_tokens.eos_token_id = self.tokenizer.eos_token_id

    def is_multimodal(self) -> bool:
        return self.model_config.mm_model_config.is_multimodal

    def _load_model_weights(self):
        self.weight: ModelWeights = self.model_weights_loader.load_weights(
            device=self._get_device_str()
        )

    @timer_wrapper(description="load custom module")
    def _load_custom_module(self):
        if self.custom_module is not None:
            self.custom_module.init(self.weight)

    def _use_new_loader(self) -> bool:
        if os.environ.get("USE_NEW_LOADER", "0") == "1":
            return True
        if (
            hasattr(self.model_config, "use_new_loader")
            and self.model_config.use_new_loader
        ):
            return True
        return False

    def _load_with_new_loader(self):
        from rtp_llm.models_py.model_loader import LoadConfig, NewModelLoader

        device_str = self._get_device_str()
        logging.info(f"Using NewModelLoader (two-phase) to load model on {device_str}")
        self.custom_module = self._init_custom_module()

        load_config = LoadConfig(
            tp_size=self.parallelism_config.tp_size,
            tp_rank=self.parallelism_config.tp_rank,
            ep_size=getattr(self.parallelism_config, "ep_size", 1),
            ep_rank=getattr(self.parallelism_config, "ep_rank", 0),
            quant_type=self._get_quant_type(),
            # 走法1:把旧 config/quant_config.py 解析出的富 quant_config 对象透传给
            # 新 loader（经 LoadConfig.quant_config → QuantizationConfig.source_config），
            # 供按层/按 prefix 派发时读取 dynamic / ignore 等字段，不重复解析 ckpt。
            quant_source_config=getattr(self.model_config, "quant_config", None),
            compute_dtype=self.model_config.compute_dtype,
            device=device_str,
            parallelism_config=self.parallelism_config,
            fmha_config=getattr(self.hw_kernel_config, "fmha_config", None),
            device_resource_config=self.device_resource_config,
            moe_config=self.moe_config,
            force_cpu_load_weights=self.force_cpu_load_weights,
        )
        quant_source_config = getattr(load_config, "quant_source_config", None)
        logging.info(
            "[NewModelLoader][quant] quant_type=%s source_config=%s "
            "weight_block_size=%s",
            load_config.quant_type,
            (
                type(quant_source_config).__name__
                if quant_source_config is not None
                else None
            ),
            getattr(quant_source_config, "weight_block_size", None),
        )

        loader = NewModelLoader(
            model_config=self.model_config,
            load_config=load_config,
            model_path=self.model_config.ckpt_path,
            device=device_str,
        )

        self.device = device_str
        self.py_model = loader.load()
        self.weight = self._build_weights_from_module(self.py_model)
        self._load_custom_module()
        self.weight_manager = None
        self.model_weights_loader = loader
        # 动态 EPLB：构造一个能从 ckpt 重载并把重排权重写回 py_model.w13/w2 的 py_eplb。
        # 未开启 EPLB / 非 MoE 时返回 None（行为同原来）。
        try:
            from rtp_llm.eplb.new_loader_eplb import build_new_loader_eplb

            self.py_eplb = build_new_loader_eplb(self, self.py_model)
        except Exception as e:
            logging.warning("[EPLB][new_loader] 构造 py_eplb 失败，回退无 EPLB: %s", e)
            self.py_eplb = None
        logging.info("NewModelLoader: model loaded successfully")
        # Multimodal models: vision weights are already inside py_model; let the
        # model build a thin mm_part that reuses them (default no-op).
        self._init_multimodal_for_new_loader()

    def _init_multimodal_for_new_loader(self):
        """Hook for new-loader multimodal models to wire up their mm_part by
        reusing the already-loaded ``py_model.visual``. Default: no-op."""
        pass

    def _get_quant_type(self) -> str:
        # Each QuantizationConfig subclass (in rtp_llm/config/quant_config.py)
        # owns its dispatch identity via get_runtime_method_key(); no central
        # if-elif here. Add a new quant type by overriding that method on the
        # config and decorating the method class with @register_quant_method.
        qc = getattr(self.model_config, "quant_config", None)
        if qc is None or not hasattr(qc, "get_runtime_method_key"):
            return "none"
        return qc.get_runtime_method_key() or "none"

    def _build_weights_from_module(self, module: torch.nn.Module) -> ModelWeights:
        num_layers = self.model_config.num_layers
        dtype = self.model_config.compute_dtype or torch.float16
        weights = ModelWeights(num_layers, self.device, dtype)

        global_weights = self._extract_global_weights(module)
        source_weights = getattr(module, "weights", None)
        if source_weights is not None:
            for key, tensor in getattr(source_weights, "global_weights", {}).items():
                if key.startswith(CustomAtomicWeight.prefix):
                    global_weights[key] = tensor
        for key, tensor in global_weights.items():
            weights.set_global_weight(key, tensor)

        # MoE: expose per-layer expert weights (w13/w2) + router into ModelWeights
        # so the C++ engine can read them — required by ExpertBalancer (EPLB),
        # which reads ``ffn_weights.moe_gate_weight->kernel`` (= W.moe_w1) at
        # construction and exchanges w1/w2 across ranks at runtime. The tensors are
        # the SAME GPU buffers as py_model's experts, so in-place ops stay visible.
        moe_layer_weights = self._extract_moe_layer_weights(module)
        for layer_id, lw in moe_layer_weights.items():
            for w_key, tensor in lw.items():
                weights.set_layer_weight(layer_id, w_key, tensor)

        logging.info(
            f"Built ModelWeights shell from nn.Module: "
            f"{num_layers} layers, "
            f"global_weights={list(global_weights.keys())}, "
            f"moe_layers={len(moe_layer_weights)}"
        )
        return weights

    def _extract_moe_layer_weights(
        self, module: torch.nn.Module
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """For MoE models, map py_model expert/router params to engine W.* names.

        ``layers.{i}.mlp.experts.w13`` -> ``W.moe_w1`` ([E_local, 2*M_tp, H], [up;gate])
        ``layers.{i}.mlp.experts.w2``  -> ``W.moe_w2`` ([E_local, H, M_tp])
        ``layers.{i}.mlp.gate.weight`` -> ``W.moe_gate`` (router)
        References the same GPU tensors (no copy).
        """
        from rtp_llm.utils.model_weight import W

        layer_prefix = r"(?:(?:language_model|model)\.)?(?:model\.)?layers\.(\d+)\."
        patterns = [
            # Qwen-style MoE blocks.
            (re.compile(layer_prefix + r"mlp\.experts\.w13$"), W.moe_w1),
            (re.compile(layer_prefix + r"mlp\.experts\.w2$"), W.moe_w2),
            (re.compile(layer_prefix + r"mlp\.gate\.weight$"), W.moe_gate),
            # MiniMax-M3 uses block_sparse_moe and may be wrapped by the VL
            # top-level language_model module.
            (
                re.compile(layer_prefix + r"block_sparse_moe\.experts\.w13$"),
                W.moe_w1,
            ),
            (
                re.compile(layer_prefix + r"block_sparse_moe\.experts\.w2$"),
                W.moe_w2,
            ),
            (
                re.compile(layer_prefix + r"block_sparse_moe\.gate\.weight$"),
                W.moe_gate,
            ),
        ]
        out: Dict[int, Dict[str, torch.Tensor]] = {}
        for name, param in module.named_parameters():
            for pat, w_key in patterns:
                m = pat.search(name)
                if m:
                    out.setdefault(int(m.group(1)), {})[w_key] = param.data
                    break
        for layer_id, weights in out.items():
            missing = [key for key in (W.moe_w1, W.moe_w2) if key not in weights]
            if missing:
                logging.warning(
                    "[EPLB][new_loader] MoE layer %d missing weights for C++ "
                    "ModelWeights: %s",
                    layer_id,
                    missing,
                )
        return out

    @staticmethod
    def _in_layers(name: str) -> bool:
        return bool(re.search(r"layers\.\d+", name))

    @staticmethod
    def _is_mm_weight(name: str) -> bool:
        return any(
            part in name
            for part in (
                "vision_tower",
                "visual",
                "multi_modal_projector",
                "mm_projector",
                "patch_merge_mlp",
            )
        )

    @staticmethod
    def _is_final_norm(name: str) -> bool:
        if BaseModel._in_layers(name):
            return False
        if "lm_head" in name:
            return False
        if BaseModel._is_mm_weight(name):
            return False

        parts = name.split(".")
        if parts[-1] not in ("weight", "bias"):
            return False

        norm_name = parts[-2] if len(parts) >= 2 else ""
        if norm_name in ("norm", "ln_f", "final_layernorm", "final_norm"):
            return True
        if norm_name in ("layernorm", "layer_norm") and len(parts) <= 3:
            return True
        return False

    @staticmethod
    def _global_weight_priority(name: str) -> Tuple[int, int]:
        parts = name.split(".")
        if parts[0] == "language_model":
            return (0, len(parts))
        if parts[0] in ("model", "transformer"):
            return (1, len(parts))
        return (2, len(parts))

    def _extract_global_weights(
        self, module: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        rules: List[Tuple[Any, str]] = [
            (
                lambda n: n.endswith(("embed_tokens.weight", "embed_token.weight"))
                and not self._in_layers(n)
                and not self._is_mm_weight(n),
                "embedding",
            ),
            (
                lambda n: n.endswith("lm_head.weight")
                and not self._in_layers(n)
                and not self._is_mm_weight(n),
                "lm_head",
            ),
            (
                lambda n: self._is_final_norm(n) and n.split(".")[-1] == "weight",
                "final_layernorm.gamma",
            ),
            (
                lambda n: self._is_final_norm(n) and n.split(".")[-1] == "bias",
                "final_layernorm.beta",
            ),
            (
                lambda n: ("position" in n and "encod" in n)
                or ("wpe" in n and not self._in_layers(n)),
                "position_encoding.weight",
            ),
            (
                lambda n: "token_type" in n and "embed" in n and not self._in_layers(n),
                "token_type_embedding.weight",
            ),
        ]

        result: Dict[str, torch.Tensor] = {}
        candidates: Dict[str, List[Tuple[str, torch.Tensor]]] = {}

        for name, param in module.named_parameters():
            for matcher, w_key in rules:
                if matcher(name):
                    candidates.setdefault(w_key, []).append((name, param.data))
                    break

        for w_key, matched in candidates.items():
            name, tensor = sorted(
                matched, key=lambda item: self._global_weight_priority(item[0])
            )[0]
            result[w_key] = tensor
            if len(matched) > 1:
                logging.info(
                    "Global weight mapped: %s -> %s, skipped candidates=%s",
                    name,
                    w_key,
                    [candidate for candidate, _ in matched if candidate != name],
                )
            else:
                logging.info(f"Global weight mapped: {name} -> {w_key}")

        return result

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

        weights_info: ModelDeployWeightInfo = self.get_weight_cls()(
            model_config=self.model_config,
            parallelism_config=self.parallelism_config,
            hw_kernel_config=self.hw_kernel_config,
            kv_cache_config=self.kv_cache_config,
            merge_lora=self.merge_lora,
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
            force_cpu_load_weights=self.force_cpu_load_weights,
        )
