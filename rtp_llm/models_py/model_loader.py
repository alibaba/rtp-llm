import inspect
import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoadMethod:
    AUTO = "auto"
    SCRATCH = "scratch"
    FASTSAFETENSORS = "fastsafetensors"


class LoadConfig:

    def __init__(
        self,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        quant_type: str = "none",
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        load_method: str = LoadMethod.AUTO,
        quant_source_config: Any = None,
        force_cpu_load_weights: bool = False,
    ):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self.device = device
        self.load_method = load_method
        self.quant_source_config = quant_source_config
        self.force_cpu_load_weights = force_cpu_load_weights

    @property
    def quant_config(self):
        try:
            from rtp_llm.models_py.quant_methods.base import QuantizationConfig

            # 走法1:把旧 config/quant_config.py 解析出的富对象透传为 source_config，
            # 供新 loader 的 method 读取 dynamic / ignore / group_size 等结构化字段，
            # 而不在新 loader 重复解析 ckpt。base_model 经 quant_source_config 注入。
            return QuantizationConfig(
                quant_type=self.quant_type,
                source_config=self.quant_source_config,
            )
        except ImportError:
            return None


class _WeightsFilter:

    def __init__(self, exclude_patterns: Optional[List[str]] = None):
        self._exclude_patterns = exclude_patterns or []
        self._compiled = None
        if self._exclude_patterns:
            import re

            self._compiled = re.compile("|".join(self._exclude_patterns))

    def apply(
        self, weights_iter: Iterator[Tuple[str, torch.Tensor]]
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        if not self._exclude_patterns or self._compiled is None:
            yield from weights_iter
            return
        for name, tensor in weights_iter:
            if self.should_load(name):
                yield name, tensor

    def should_load(self, name: str) -> bool:
        if not self._exclude_patterns or self._compiled is None:
            return True
        return self._compiled.search(name) is None


def _discover_ckpt_files(model_path: str) -> List[str]:
    try:
        from rtp_llm.models_py import weight_mapper

        if hasattr(weight_mapper, "discover_ckpt_files"):
            return weight_mapper.discover_ckpt_files(model_path)
    except ImportError:
        pass
    import glob

    # Match weight_mapper.discover_ckpt_files priority: prefer safetensors,
    # then bin, then pt — and return the first non-empty group rather than
    # mixing formats. Mixing co-located safetensors and bin would double-load
    # the same weights through two paths.
    for ext in ("*.safetensors", "*.bin", "*.pt"):
        found = sorted(glob.glob(os.path.join(model_path, ext)))
        if ext == "*.bin":
            found = [f for f in found if "optimizer" not in os.path.basename(f)]
        if found:
            return found
    return []


def _evict_page_cache(ckpt_files: List[str]):
    total = 0
    for path in ckpt_files:
        try:
            size = os.path.getsize(path)
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
                total += size
            finally:
                os.close(fd)
        except (OSError, AttributeError):
            pass
    logger.info(
        f"Evicted page cache for {len(ckpt_files)} files " f"({total / 1024**3:.2f} GB)"
    )


def _get_all_weights(
    ckpt_files: List[str], device: str = "cpu", name_filter: Optional[Any] = None
) -> Iterator[Tuple[str, torch.Tensor]]:
    if name_filter is None:
        try:
            from rtp_llm.models_py import weight_mapper

            if hasattr(weight_mapper, "get_all_weights"):
                count = 0
                logger.info(
                    "Begin streaming weights via weight_mapper: files=%d device=%s",
                    len(ckpt_files),
                    device,
                )
                for name, tensor in weight_mapper.get_all_weights(
                    ckpt_files, device=device
                ):
                    count += 1
                    if count == 1 or count % 500 == 0:
                        logger.info(
                            "Streaming weight #%d: %s shape=%s dtype=%s",
                            count,
                            name,
                            tuple(tensor.shape),
                            tensor.dtype,
                        )
                    yield name, tensor
                logger.info("Finished streaming %d weights via weight_mapper", count)
                return
        except ImportError:
            pass
    else:
        logger.info(
            "Begin streaming weights with name filter: files=%d device=%s",
            len(ckpt_files),
            device,
        )

    def _should_load(name: str) -> bool:
        return name_filter is None or name_filter.should_load(name)

    def _emit_filtered_item(
        name: str, tensor: torch.Tensor
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        if name_filter is not None and hasattr(name_filter, "filter_item"):
            yield from name_filter.filter_item(name, tensor)
        else:
            yield name, tensor

    count = 0
    skipped = 0
    for ckpt_file in ckpt_files:
        logger.info("Loading checkpoint file: %s", ckpt_file)
        if ckpt_file.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(ckpt_file, framework="pt", device=device) as f:
                keys = list(f.keys())
                for name in keys:
                    if not _should_load(name):
                        skipped += 1
                        continue
                    tensor = f.get_tensor(name)
                    count += 1
                    if count == 1 or count % 500 == 0:
                        logger.info(
                            "Streaming weight #%d: %s shape=%s dtype=%s (skipped=%d)",
                            count,
                            name,
                            tuple(tensor.shape),
                            tensor.dtype,
                            skipped,
                        )
                    yield from _emit_filtered_item(name, tensor)
            logger.info(
                "Scanned checkpoint file: %s yielded=%d skipped=%d",
                ckpt_file,
                count,
                skipped,
            )
        else:
            # weights_only=True forbids arbitrary pickle payloads — torch.load
            # otherwise allows arbitrary code execution on untrusted .pt/.bin
            # files. Modern HF ckpts and the {state_dict, model} containers
            # below are pure tensor dicts and load fine under this flag.
            from rtp_llm.models_py.weight_mapper import _unwrap_pytorch_state_dict

            state_dict = _unwrap_pytorch_state_dict(
                torch.load(ckpt_file, map_location=device, weights_only=True)
            )
            logger.info(
                "Loaded checkpoint file: %s tensors=%d", ckpt_file, len(state_dict)
            )
            for name, tensor in state_dict.items():
                if not _should_load(name):
                    skipped += 1
                    continue
                count += 1
                if count == 1 or count % 500 == 0:
                    logger.info(
                        "Streaming filtered weight #%d: %s shape=%s dtype=%s "
                        "(skipped=%d)",
                        count,
                        name,
                        tuple(tensor.shape),
                        tensor.dtype,
                        skipped,
                    )
                yield from _emit_filtered_item(name, tensor)
    logger.info("Finished streaming %d weights (skipped=%d)", count, skipped)


_EP_EXPERT_RE = re.compile(r"experts\.(\d+)\.")
_STACKED_EP_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<proj>gate_up_proj|down_proj)"
    r"(?:\.(?P<param>[^.]+))?$"
)


class _ExpertRangeFilter:
    """Drop weights for experts outside [start_expert, end_expert).

    Replaces the previous N-alternation regex (one branch per excluded expert),
    which compiled and matched in O(num_experts) per weight — costly for
    256/512-expert MoE ckpts. Here we run one O(1) re.search per weight name
    against a single fixed pattern and bound-check the captured expert id.
    """

    def __init__(self, start_expert: int, end_expert: int):
        self.start_expert = start_expert
        self.end_expert = end_expert

    def apply(
        self, weights_iter: Iterator[Tuple[str, torch.Tensor]]
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, tensor in weights_iter:
            yield from self.filter_item(name, tensor)

    def filter_item(
        self, name: str, tensor: torch.Tensor
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        stacked = _STACKED_EP_RE.match(name)
        if stacked is not None:
            yield from self._split_stacked_experts(stacked, name, tensor)
            return
        if self.should_load(name):
            yield name, tensor

    def _split_stacked_experts(
        self, match: re.Match, name: str, tensor: torch.Tensor
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        if not hasattr(tensor, "dim") or tensor.dim() == 0:
            raise ValueError(
                f"EP stacked MoE tensor {name!r} must have expert dimension at dim 0"
            )
        if tensor.shape[0] < self.end_expert:
            raise ValueError(
                f"EP stacked MoE tensor {name!r} has only {tensor.shape[0]} experts, "
                f"but this rank needs [{self.start_expert}, {self.end_expert})"
            )
        prefix = match.group("prefix")
        proj = match.group("proj")
        param = match.group("param") or "weight"
        for expert_id in range(self.start_expert, self.end_expert):
            yield (
                f"{prefix}.{expert_id}.{proj}.{param}",
                tensor.select(0, expert_id).contiguous(),
            )

    def should_load(self, name: str) -> bool:
        if _STACKED_EP_RE.match(name) is not None:
            return True
        m = _EP_EXPERT_RE.search(name)
        if m is None:
            # Not an expert-scoped weight (e.g. shared attn / norm); keep.
            return True
        expert_id = int(m.group(1))
        return self.start_expert <= expert_id < self.end_expert


def _validate_ep_partition(ep_size: int, ep_rank: int, num_experts: int):
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    if ep_rank < 0 or ep_rank >= ep_size:
        raise ValueError(
            f"ep_rank must satisfy 0 <= ep_rank < ep_size, got "
            f"ep_rank={ep_rank}, ep_size={ep_size}"
        )
    if num_experts < 0:
        raise ValueError(f"num_experts must be non-negative, got {num_experts}")
    if num_experts != 0 and num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size}) "
            "to avoid silently dropping experts during EP loading."
        )


def _create_ep_filter(ep_size: int, ep_rank: int, num_experts: int):
    _validate_ep_partition(ep_size, ep_rank, num_experts)
    if num_experts == 0:
        return _WeightsFilter()
    experts_per_rank = num_experts // ep_size
    start_expert = ep_rank * experts_per_rank
    end_expert = start_expert + experts_per_rank
    return _ExpertRangeFilter(start_expert, end_expert)


def _is_quanted_config(source_config: Any) -> bool:
    if source_config is None:
        return False
    value = getattr(source_config, "is_quanted", False)
    if callable(value):
        return bool(value())
    return bool(value)


def _is_quantized_load(model_config: Any, load_config: LoadConfig) -> bool:
    if _is_quanted_config(getattr(load_config, "quant_source_config", None)):
        return True
    quant_config = getattr(model_config, "quant_config", None)
    if _is_quanted_config(quant_config):
        return True
    quantization = getattr(model_config, "quantization", None)
    if isinstance(quantization, str):
        return quantization.strip().lower() not in ("", "none", "null")
    return bool(quantization)


class NewModelLoader:

    def __init__(
        self,
        model_config: Any,
        load_config: Optional[LoadConfig] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_config = model_config
        self.load_config = load_config if load_config is not None else LoadConfig()
        self.model_path = model_path
        self.device = device if device is not None else self.load_config.device
        self._ckpt_files_cache: Optional[List[str]] = None

    def load(self) -> nn.Module:
        method, _ = self._resolve_load_method()
        logger.info(f"NewModelLoader using load method: {method}")
        if method != LoadMethod.SCRATCH:
            raise RuntimeError(f"Unsupported newloader load method after resolution: {method}")
        return self._load_via_scratch()

    def _load_via_scratch(self) -> nn.Module:
        import time

        logger.info("Starting model loading (scratch path)...")
        model = self._create_model()
        ep_filter = self._create_ep_filter()
        weights_iter = _get_all_weights(
            self._discover_ckpt_files_cached(), device="cpu", name_filter=ep_filter
        )
        t0 = time.time()
        model.load_weights(weights_iter)
        logger.info(f"model.load_weights() took {time.time() - t0:.2f}s")

        if self.load_config.force_cpu_load_weights:
            logger.warning(
                "force_cpu_load_weights is enabled; running post-load hooks on CPU "
                "before moving final weights to %s",
                self.device,
            )
            self._run_post_load_hooks(model, defer_moe_executor_build=True)
            t1 = time.time()
            logger.info(f"Moving post-load model to device: {self.device}")
            model.to(self.device)
            logger.info(f"model.to({self.device}) took {time.time() - t1:.2f}s")
            self._build_deferred_moe_executors(model)
        else:
            t1 = time.time()
            logger.info(f"Moving model to device: {self.device}")
            model.to(self.device)
            logger.info(f"model.to({self.device}) took {time.time() - t1:.2f}s")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._run_post_load_hooks(model)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._log_peak_gpu_memory()
        return model

    def _run_post_load_hooks(self, model: nn.Module, defer_moe_executor_build: bool = False):
        import time

        t0 = time.time()
        count = 0
        for module in model.modules():
            setattr(
                module,
                "_new_loader_force_cpu_load_weights",
                bool(self.load_config.force_cpu_load_weights),
            )
            setattr(module, "_new_loader_defer_moe_executor_build", defer_moe_executor_build)
            if hasattr(module, "process_weights_after_loading"):
                module.process_weights_after_loading()
                count += 1
        logger.info(
            f"Post-load hooks ran on {count} modules in {time.time() - t0:.2f}s"
        )

    def _build_deferred_moe_executors(self, model: nn.Module):
        count = 0
        for module in model.modules():
            if bool(getattr(module, "_new_loader_deferred_moe_executor", False)):
                setattr(module, "_new_loader_defer_moe_executor_build", False)
                module._maybe_build_fused_moe()
                module._new_loader_deferred_moe_executor = False
                count += 1
        if count:
            logger.info("Built %d deferred MoE executor(s) after device migration", count)

    def _log_peak_gpu_memory(self):
        if not torch.cuda.is_available():
            return
        try:
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            gib = 1024**3
            logger.info(
                f"Peak GPU memory after loading: "
                f"allocated={allocated / gib:.2f} GiB, "
                f"reserved={reserved / gib:.2f} GiB"
            )
        except Exception as e:
            logger.debug("Failed to query peak GPU memory after loading: %s", e)

    def _resolve_target_device(self) -> str:
        device = str(self.device)
        if not device.startswith("cuda"):
            return device
        if ":" in device:
            return device
        return f"cuda:{torch.cuda.current_device()}"

    def _distributed_world_size(self) -> int:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return 1
        return torch.distributed.get_world_size(torch.distributed.group.WORLD)

    def _resolve_load_method(self) -> Tuple[str, bool]:
        configured = getattr(self.load_config, "load_method", LoadMethod.AUTO)
        load_method = (configured or LoadMethod.AUTO).lower()
        explicit = load_method != LoadMethod.AUTO
        if load_method == LoadMethod.AUTO:
            env = os.environ.get("LOAD_METHOD", "").strip().lower()
            if env and env != LoadMethod.AUTO:
                if env not in (LoadMethod.SCRATCH, LoadMethod.FASTSAFETENSORS):
                    raise ValueError(
                        f"Unsupported LOAD_METHOD {env!r}; expected one of "
                        f"{LoadMethod.AUTO!r}, {LoadMethod.SCRATCH!r}, "
                        f"{LoadMethod.FASTSAFETENSORS!r}"
                    )
                load_method = env
                explicit = True
                logger.info(f"LOAD_METHOD env: {load_method}")
            else:
                # Keep the core newloader PR on the scratch path. The
                # fastsafetensors path needs a separate distributed failure/abort
                # protocol before it can be safely enabled in auto mode.
                return LoadMethod.SCRATCH, False

        if load_method not in (LoadMethod.SCRATCH, LoadMethod.FASTSAFETENSORS):
            raise ValueError(
                f"Unsupported load_method {load_method!r}; expected one of "
                f"{LoadMethod.AUTO!r}, {LoadMethod.SCRATCH!r}, "
                f"{LoadMethod.FASTSAFETENSORS!r}"
            )

        if load_method == LoadMethod.FASTSAFETENSORS:
            raise RuntimeError(
                "fastsafetensors load_method is not enabled in the newloader core PR; "
                "use scratch and submit fastsafetensors as a separate distributed-load PR."
            )
        return LoadMethod.SCRATCH, explicit

    def _discover_ckpt_files_cached(self) -> List[str]:
        if self._ckpt_files_cache is not None:
            return self._ckpt_files_cache
        model_path = self._resolve_model_path()
        if not os.path.isdir(model_path):
            raise NotADirectoryError(f"model_path is not a directory: {model_path}")
        ckpt_files = _discover_ckpt_files(model_path)
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in: {model_path}")
        logger.info(f"Found {len(ckpt_files)} checkpoint file(s) in {model_path}")
        if os.environ.get("DROP_PAGE_CACHE", "0") == "1":
            _evict_page_cache(ckpt_files)
        self._ckpt_files_cache = ckpt_files
        return ckpt_files

    def _create_model(self) -> nn.Module:
        from rtp_llm.models_py.registry import MODEL_REGISTRY

        model_type = self._get_model_type()
        if model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Model type '{model_type}' not found in registry. "
                f"Available: {list(MODEL_REGISTRY.keys())}."
            )
        model_cls = MODEL_REGISTRY[model_type]
        logger.info(
            "NewModelLoader registry hit: model_type=%s cls=%s module=%s file=%s",
            model_type,
            model_cls.__qualname__,
            model_cls.__module__,
            inspect.getfile(model_cls),
        )
        model = model_cls(self.model_config, self.load_config)
        logger.info(f"Created model: {model_cls.__name__} (type={model_type})")
        return model

    def _get_model_type(self) -> str:
        if hasattr(self.model_config, "model_type"):
            return self.model_config.model_type
        elif isinstance(self.model_config, dict):
            return self.model_config.get("model_type", "")
        raise ValueError("Cannot determine model_type from model_config.")

    def _resolve_model_path(self) -> str:
        if self.model_path:
            return self.model_path
        if hasattr(self.model_config, "model_path"):
            if self.model_config.model_path:
                return self.model_config.model_path
        if isinstance(self.model_config, dict):
            path = self.model_config.get("model_path", "")
            if path:
                return path
        raise ValueError("model_path is required for weight loading.")

    def _apply_ep_filter(self, weights_iter):
        ep_filter = self._create_ep_filter()
        if ep_filter is None:
            return weights_iter
        return ep_filter.apply(weights_iter)

    def _create_ep_filter(self):
        ep_size = getattr(self.load_config, "ep_size", 1)
        ep_rank = getattr(self.load_config, "ep_rank", 0)
        if ep_size <= 1:
            return None
        # Prefer model_config.expert_num, matching the legacy loader and MoE
        # executors. Accept num_experts as a compatibility alias for lightweight
        # configs/tests that use HF-style naming. Dense models lack both attrs
        # and short-circuit to a no-op filter.
        num_experts = getattr(
            self.model_config,
            "expert_num",
            getattr(self.model_config, "num_experts", 0),
        )
        if num_experts == 0:
            return None
        return _create_ep_filter(ep_size, ep_rank, num_experts)

    # ------------------------------------------------------------------ #
    #  LoRA support
    # ------------------------------------------------------------------ #

    def load_lora_weights(self, adapter_name: str, lora_path: str, device: str = "cpu"):
        """Load HF-format LoRA adapter weights independently of the legacy loader.

        Reads adapter_config.json for rank/alpha, then loads adapter_model
        checkpoint, maps HF module names to engine-internal W.* names, applies
        TP slicing, and returns a LoRAWeights object ready for the C++ runtime.
        """
        from rtp_llm.lora.lora_weights import LoRAWeights
        from rtp_llm.utils.model_weight import W

        lora_config = self._read_lora_config(lora_path)
        rank = lora_config["rank"]
        lora_alpha = lora_config["lora_alpha"]

        num_layers = getattr(self.model_config, "num_layers", 0)
        if num_layers == 0:
            num_layers = getattr(self.model_config, "num_hidden_layers", 0)
        lora_weights = LoRAWeights(num_layers)
        lora_weights.set_lora_rank(rank)

        tp_size = getattr(self.load_config, "tp_size", 1)
        tp_rank = getattr(self.load_config, "tp_rank", 0)
        compute_dtype = getattr(self.load_config, "compute_dtype", torch.float16)

        state_dict = self._load_lora_state_dict(lora_path, device)
        loaded_pairs = set()
        skipped = []

        for hf_name, tensor in state_dict.items():
            parsed = _parse_hf_lora_name(hf_name)
            if parsed is None:
                skipped.append((hf_name, "unrecognized name"))
                continue
            layer_id, hf_module, ab_suffix = parsed

            if layer_id >= num_layers:
                skipped.append((hf_name, f"layer {layer_id} out of range"))
                continue

            if hf_module in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            ):
                raise NotImplementedError(
                    "NewModelLoader LoRA does not yet support split q/k/v adapters. "
                    "They must be packed into the engine fused qkv LoRA tensor before "
                    "being passed to LoRAWeights; refusing to map them all to "
                    "self_attention_weights.query_weight.kernel because that would "
                    "overwrite q/k/v adapters silently."
                )

            engine_name = _HF_TO_ENGINE_LORA.get(hf_module)
            if engine_name is None:
                skipped.append((hf_name, f"unmapped module {hf_module!r}"))
                continue

            tensor = tensor.to(compute_dtype).t().contiguous()

            # TP slicing uses the engine-internal transposed layout.
            if tp_size > 1:
                split_rule = _LORA_TP_RULES.get((hf_module, ab_suffix))
                if split_rule is not None:
                    tensor = _tp_slice(tensor, tp_size, tp_rank, split_rule)

            full_name = f"{engine_name}.{ab_suffix}"
            lora_weights.set_layer_weight(False, layer_id, full_name, tensor)
            loaded_pairs.add((layer_id, engine_name, ab_suffix))

        if not loaded_pairs:
            sample = skipped[:5]
            raise RuntimeError(
                f"LoRA adapter '{adapter_name}' did not contain any mappable "
                f"newloader tensors; skipped={sample}"
            )

        missing_pairs = []
        for layer_id, engine_name, _ in loaded_pairs:
            for suffix in ("lora_A", "lora_B"):
                if (layer_id, engine_name, suffix) not in loaded_pairs:
                    missing_pairs.append((layer_id, engine_name, suffix))
        if missing_pairs:
            raise RuntimeError(
                f"LoRA adapter '{adapter_name}' has incomplete A/B tensor pairs: "
                f"missing={missing_pairs[:10]}"
            )

        lora_weights.apply_scale(lora_alpha / rank)
        logger.info(
            f"Loaded LoRA '{adapter_name}' from {lora_path}: "
            f"rank={rank}, alpha={lora_alpha}, layers={num_layers}"
        )
        return lora_weights

    @staticmethod
    def _read_lora_config(lora_path: str) -> dict:
        import json

        config_path = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"LoRA adapter_config.json not found: {config_path}"
            )
        with open(config_path) as f:
            cfg = json.load(f)
        return {
            "rank": cfg["r"],
            "lora_alpha": cfg["lora_alpha"],
        }

    @staticmethod
    def _load_lora_state_dict(lora_path: str, device: str) -> dict:
        for filename in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "adapter_model.pt",
        ):
            filepath = os.path.join(lora_path, filename)
            if os.path.exists(filepath):
                if filepath.endswith(".safetensors"):
                    from safetensors.torch import load_file

                    return load_file(filepath, device=device)
                else:
                    return torch.load(filepath, map_location=device, weights_only=True)
        raise FileNotFoundError(f"No adapter model file found in {lora_path}")


# ------------------------------------------------------------------ #
#  LoRA name mapping and TP slicing helpers
# ------------------------------------------------------------------ #

_HF_LORA_PATTERN = re.compile(
    r"base_model\.model\.model\.layers\.(\d+)\.(.*)\.(lora_A|lora_B)\.weight"
)


def _parse_hf_lora_name(name: str):
    """Parse HF LoRA tensor name -> (layer_id, hf_module, 'lora_A'|'lora_B')."""
    m = _HF_LORA_PATTERN.match(name)
    if m is None:
        return None
    return int(m.group(1)), m.group(2), m.group(3)


# HF module path -> engine internal W.* name
_HF_TO_ENGINE_LORA = {
    "self_attn.o_proj": "self_attention_weights.attention_output_weight.kernel",
    "mlp.gate_proj": "ffn_weights.intermediate_weight.kernel",
    "mlp.up_proj": "ffn_weights.intermediate_weight3.kernel",
    "mlp.down_proj": "ffn_weights.intermediate_weight2.kernel",
}


# TP slicing rules: (hf_module, ab_suffix) -> (dim, "split"|"identity")
# Column-parallel projections (q/k/v/gate/up): lora_B sliced on dim=-1
# Row-parallel projections (o/down): lora_A sliced on dim=0
_LORA_TP_RULES: Dict[tuple, tuple] = {
    # q/k/v: lora_B output-dim split
    ("self_attn.q_proj", "lora_B"): (-1, "split"),
    ("self_attn.k_proj", "lora_B"): (-1, "split"),
    ("self_attn.v_proj", "lora_B"): (-1, "split"),
    # o_proj: lora_A input-dim split
    ("self_attn.o_proj", "lora_A"): (0, "split"),
    # gate/up: lora_B output-dim split
    ("mlp.gate_proj", "lora_B"): (-1, "split"),
    ("mlp.up_proj", "lora_B"): (-1, "split"),
    # down: lora_A input-dim split
    ("mlp.down_proj", "lora_A"): (0, "split"),
}


def _tp_slice(
    tensor: torch.Tensor, tp_size: int, tp_rank: int, rule: tuple
) -> torch.Tensor:
    dim, action = rule
    if action != "split":
        return tensor
    size = tensor.shape[dim]
    if tp_size <= 0:
        raise ValueError(f"LoRA TP slice requires positive tp_size, got {tp_size}")
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(
            f"LoRA TP slice requires 0 <= tp_rank < tp_size, got "
            f"tp_rank={tp_rank}, tp_size={tp_size}"
        )
    if size % tp_size != 0:
        raise ValueError(
            f"LoRA TP slice dimension {dim} size ({size}) must be divisible "
            f"by tp_size ({tp_size}); non-divisible sharding would drop weights"
        )
    chunk_size = size // tp_size
    start = tp_rank * chunk_size
    return tensor.narrow(dim, start, chunk_size).contiguous()
