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
        **kwargs,
    ):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self.device = device
        self.load_method = load_method
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def quant_config(self):
        try:
            from rtp_llm.models_py.quant_methods.base import QuantizationConfig

            return QuantizationConfig(quant_type=self.quant_type)
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
            if not self._compiled.search(name):
                yield name, tensor


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
        if found:
            if ext == "*.bin":
                found = [f for f in found if "optimizer" not in os.path.basename(f)]
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
    ckpt_files: List[str], device: str = "cpu"
) -> Iterator[Tuple[str, torch.Tensor]]:
    try:
        from rtp_llm.models_py import weight_mapper

        if hasattr(weight_mapper, "get_all_weights"):
            yield from weight_mapper.get_all_weights(ckpt_files, device=device)
            return
    except ImportError:
        pass
    for ckpt_file in ckpt_files:
        if ckpt_file.endswith(".safetensors"):
            from safetensors.torch import load_file as safetensors_load

            state_dict = safetensors_load(ckpt_file, device=device)
        else:
            # weights_only=True forbids arbitrary pickle payloads — torch.load
            # otherwise allows arbitrary code execution on untrusted .pt/.bin
            # files. Modern HF ckpts and the {state_dict, model} containers
            # below are pure tensor dicts and load fine under this flag.
            state_dict = torch.load(
                ckpt_file, map_location=device, weights_only=True
            )
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
        for name, tensor in state_dict.items():
            yield name, tensor


def _get_fastsafetensors_weights(
    ckpt_files: List[str], device: str
) -> Iterator[Tuple[str, torch.Tensor]]:
    from fastsafetensors import ParallelLoader, SingleGroup

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        pg = torch.distributed.group.WORLD
    else:
        pg = SingleGroup()

    loader = ParallelLoader(
        pg=pg,
        hf_weights_files=sorted(ckpt_files),
        use_tqdm_on_load=True,
        device=device,
        bbuf_size_kb=1024 * 1024 * 2,
        use_shm=True,
    )
    try:
        yield from loader.iterate_weights()
    finally:
        inner = getattr(loader, "loader", None)
        if inner is not None and hasattr(inner, "close"):
            inner.close()


_EP_EXPERT_RE = re.compile(r"experts\.(\d+)\.")


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
            m = _EP_EXPERT_RE.search(name)
            if m is None:
                # Not an expert-scoped weight (e.g. shared attn / norm); keep.
                yield name, tensor
                continue
            expert_id = int(m.group(1))
            if self.start_expert <= expert_id < self.end_expert:
                yield name, tensor


def _create_ep_filter(ep_size: int, ep_rank: int, num_experts: int):
    if num_experts == 0:
        return _WeightsFilter()
    experts_per_rank = num_experts // ep_size
    start_expert = ep_rank * experts_per_rank
    end_expert = start_expert + experts_per_rank
    return _ExpertRangeFilter(start_expert, end_expert)


class NewModelLoader:

    def __init__(
        self,
        model_config: Any,
        load_config: Optional[LoadConfig] = None,
        model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.model_config = model_config
        self.load_config = load_config if load_config is not None else LoadConfig()
        self.model_path = model_path
        self.device = device
        self._ckpt_files_cache: Optional[List[str]] = None

    def load(self) -> nn.Module:
        method = self._resolve_load_method()
        logger.info(f"NewModelLoader using load method: {method}")
        if method == LoadMethod.FASTSAFETENSORS:
            return self._load_via_fastsafetensors()
        return self._load_via_scratch()

    def _load_via_scratch(self) -> nn.Module:
        import time

        logger.info("Starting model loading (scratch path)...")
        model = self._create_model()
        weights_iter = _get_all_weights(
            self._discover_ckpt_files_cached(), device="cpu"
        )
        weights_iter = self._apply_ep_filter(weights_iter)

        t0 = time.time()
        model.load_weights(weights_iter)
        logger.info(f"model.load_weights() took {time.time() - t0:.2f}s")

        t1 = time.time()
        logger.info(f"Moving model to device: {self.device}")
        model.to(self.device)
        logger.info(f"model.to({self.device}) took {time.time() - t1:.2f}s")
        return model

    def _load_via_fastsafetensors(self) -> nn.Module:
        import time

        logger.info("Starting model loading (fastsafetensors path)...")

        target_device = self._resolve_target_device()

        t0 = time.time()
        with torch.device("meta"):
            model = self._create_model()
        model.to_empty(device=target_device)
        logger.info(
            f"meta create + to_empty({target_device}) " f"took {time.time() - t0:.2f}s"
        )

        weights_iter = _get_fastsafetensors_weights(
            self._discover_ckpt_files_cached(), device=target_device
        )
        weights_iter = self._apply_ep_filter(weights_iter)

        t1 = time.time()
        model.load_weights(weights_iter)
        logger.info(f"model.load_weights() took {time.time() - t1:.2f}s")
        return model

    def _resolve_target_device(self) -> str:
        device = str(self.device)
        if not device.startswith("cuda"):
            return device
        if ":" in device:
            return device
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return f"cuda:{torch.distributed.get_rank()}"
        return f"cuda:{torch.cuda.current_device()}"

    def _resolve_load_method(self) -> str:
        method = getattr(self.load_config, "load_method", LoadMethod.AUTO)
        method = (method or LoadMethod.AUTO).lower()
        if method == LoadMethod.AUTO:
            env = os.environ.get("LOAD_METHOD", "").strip().lower()
            if env in (LoadMethod.SCRATCH, LoadMethod.FASTSAFETENSORS):
                method = env
                logger.info(f"LOAD_METHOD env: {method}")

        if method == LoadMethod.FASTSAFETENSORS:
            ok, reason = self._fastsafetensors_eligible()
            if not ok:
                raise RuntimeError(
                    f"fastsafetensors load requested but unavailable: {reason}"
                )
            return LoadMethod.FASTSAFETENSORS
        return LoadMethod.SCRATCH

    def _fastsafetensors_eligible(self) -> Tuple[bool, str]:
        try:
            import fastsafetensors  # noqa: F401
        except ImportError:
            return False, "fastsafetensors module not installed"
        if not torch.cuda.is_available():
            return False, "cuda not available"
        if not str(self.device).startswith("cuda"):
            return False, f"device {self.device} is not cuda"

        ckpt_files = self._discover_ckpt_files_cached()
        if not ckpt_files:
            return False, "no checkpoint files discovered"
        if not all(f.endswith(".safetensors") for f in ckpt_files):
            return False, "not all checkpoint files are safetensors"

        sizes = [os.path.getsize(f) for f in ckpt_files]
        max_file = max(sizes)
        total = sum(sizes)
        tp_size = max(getattr(self.load_config, "tp_size", 1), 1)
        ep_size = max(getattr(self.load_config, "ep_size", 1), 1)
        rank_share = total / max(tp_size, ep_size)

        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except Exception as e:
            return False, f"cuda mem_get_info failed: {e}"

        headroom = free_bytes - rank_share - 3 * max_file
        gib = 1024**3
        logger.info(
            f"fastsafetensors capacity: free={free_bytes/gib:.1f}GiB, "
            f"rank_share={rank_share/gib:.1f}GiB, "
            f"max_file={max_file/gib:.1f}GiB, "
            f"headroom={headroom/gib:.1f}GiB"
        )
        if headroom <= 0:
            return False, "insufficient GPU free memory"
        return True, "ok"

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
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        model_cls = MODEL_REGISTRY[model_type]
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
        ep_size = getattr(self.load_config, "ep_size", 1)
        ep_rank = getattr(self.load_config, "ep_rank", 0)
        if ep_size <= 1:
            return weights_iter
        # Single source of truth: model_config.expert_num. Same convention as
        # the legacy loader (model_loader/ffn_weight.py) and every MoE executor
        # under models_py/modules/factory/fused_moe/. dense models lack this
        # attr, getattr default 0 short-circuits _create_ep_filter to a no-op.
        num_experts = getattr(self.model_config, "expert_num", 0)
        ep_filter = _create_ep_filter(ep_size, ep_rank, num_experts)
        return ep_filter.apply(weights_iter)
