"""Offline FP4 mega-MoE weight loader.

Reads pre-quantized FP4+UE8M0 MoE weights directly from checkpoint, produced
by ``tools/convert/glm5_fp4_moe_fp8_quant.py``:

  - ``{expert_prefix}.weight``        int8 (FP4 packed)
  - ``{expert_prefix}.weight_scale``  fp32 (UE8M0 per-block scale)

Output format matches ``OnlineMegaMoeFp4Weight`` so downstream
``MegaMoeWrapper`` and ``MegaMoeFusedWrapper`` accept it identically.

Auto-detected by ``is_offline_mega_moe_fp4_ckpt(database)`` — looks for
the FP4 scale suffix on any expert weight in the ckpt index.
"""

from typing import Any, Dict, Optional

import torch

from rtp_llm.config.quant_config import QuantizationConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import (
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.database import BaseDatabase
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity

_MEGA_MOE_KERNEL_NAMES = (W.moe_w1, W.moe_w2)
_SHARED_EXPERT_KERNEL_NAMES = (W.ffn_w13, W.ffn_w2)

_FP4_W_SUFFIX = ".weight"
_FP4_S_SUFFIX = ".weight_scale"


def _mega_moe_scale_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_s1
    if name == W.moe_w2:
        return W.moe_s2
    raise ValueError(f"unsupported mega_moe kernel name: {name}")


def _shared_expert_scale_name(name: str) -> str:
    if name == W.ffn_w13:
        return W.ffn_s13
    if name == W.ffn_w2:
        return W.ffn_s2
    raise ValueError(f"unsupported shared expert kernel name: {name}")


def _concat_shared_w13(ts: list[torch.Tensor]) -> torch.Tensor:
    if len(ts) != 2:
        raise ValueError(f"shared expert w13 expects gate/up tensors, got {len(ts)}")
    return torch.cat(ts, dim=0).contiguous()


def _is_shared_expert_weight(src_weight_info: WeightModule) -> bool:
    if not isinstance(src_weight_info, FfnAtomicWeight):
        return False
    if src_weight_info.name not in _SHARED_EXPERT_KERNEL_NAMES:
        return False
    return all(".mlp.shared_experts." in w.name for w in src_weight_info.weights)


class OfflineMegaMoeFp4MoeWeight(CompositeWeight, QuantWeight):
    """Load pre-quantized FP4 MoE weights (int8 packed + fp32 UE8M0 scale)."""

    moe_weight_list = list(_MEGA_MOE_KERNEL_NAMES)

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        # Not auto-selected via QuantConfig.support(); inserted by
        # `_apply_mega_moe_fp4_wrappers` after offline-ckpt detection.
        return False

    def __init__(
        self,
        src_weight_info: MoeAtomicWeight,
        **kwargs: Any,
    ):
        if src_weight_info.name not in _MEGA_MOE_KERNEL_NAMES:
            raise ValueError(
                f"OfflineMegaMoeFp4MoeWeight only wraps {_MEGA_MOE_KERNEL_NAMES}, "
                f"got {src_weight_info.name}"
            )

        kernel = MoeAtomicWeight(
            name=src_weight_info.name,
            weights=src_weight_info.weights,
            process_fun=src_weight_info.process_fun,
            data_type=torch.int8,
            config=src_weight_info.config,
            stacked_ckpt_keys=getattr(src_weight_info, "stacked_ckpt_keys", False),
        )
        scale_weights = [
            CkptWeightInfo(
                w.name[: -len(_FP4_W_SUFFIX)] + _FP4_S_SUFFIX,
                w.merge_fun,
            )
            for w in src_weight_info.weights
        ]
        scale = MoeAtomicWeight(
            name=_mega_moe_scale_name(src_weight_info.name),
            weights=scale_weights,
            process_fun=src_weight_info.process_fun,
            data_type=torch.float32,
            config=src_weight_info.config,
        )

        sub_weights = {kernel.name: kernel, scale.name: scale}
        super().__init__(
            sub_weights,
            quant_config=None,
            name=src_weight_info.name,
            **{k: v for k, v in kwargs.items() if k != "name"},
        )
        self.kernel = kernel
        self.scale = scale

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        names = self.kernel.get_tensor_names(layer_id, load_config)
        names |= self.scale.get_tensor_names(layer_id, load_config)
        return names

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        kernel_dict = self.kernel._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        scale_dict = self.scale._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        return {
            self.kernel.name: kernel_dict[self.kernel.name],
            self.scale.name: scale_dict[self.scale.name],
        }

    def _split(self, tensor, load_config: LoadConfig):
        split_kernel = self.kernel._split(
            {self.kernel.name: tensor[self.kernel.name]}, load_config
        )
        split_scale = self.scale._split(
            {self.scale.name: tensor[self.scale.name]}, load_config
        )
        out: Dict[str, torch.Tensor] = {}
        out.update(split_kernel)
        out.update(split_scale)
        return out

    def _postprocess(self, tensor, device: str, load_config: LoadConfig):
        return {
            self.kernel.name: tensor[self.kernel.name],
            self.scale.name: tensor[self.scale.name],
        }


class OfflineMegaMoeFp4SharedExpertWeight(CompositeWeight, QuantWeight):
    """Load pre-quantized FP4 shared-expert weights for fused MegaMoE."""

    shared_weight_list = list(_SHARED_EXPERT_KERNEL_NAMES)

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        return False

    def __init__(
        self,
        src_weight_info: FfnAtomicWeight,
        **kwargs: Any,
    ):
        if not _is_shared_expert_weight(src_weight_info):
            raise ValueError(
                "OfflineMegaMoeFp4SharedExpertWeight only wraps shared_experts "
                f"{_SHARED_EXPERT_KERNEL_NAMES}, got {src_weight_info}"
            )

        process_fun = (
            _concat_shared_w13 if src_weight_info.name == W.ffn_w13 else identity
        )
        kernel = FfnAtomicWeight(
            name=src_weight_info.name,
            weights=src_weight_info.weights,
            process_fun=process_fun,
            data_type=torch.int8,
            config=src_weight_info.config,
        )
        scale_weights = [
            CkptWeightInfo(
                w.name[: -len(_FP4_W_SUFFIX)] + _FP4_S_SUFFIX,
                w.merge_fun,
            )
            for w in src_weight_info.weights
        ]
        scale = FfnAtomicWeight(
            name=_shared_expert_scale_name(src_weight_info.name),
            weights=scale_weights,
            process_fun=process_fun,
            data_type=torch.float32,
            config=src_weight_info.config,
        )

        sub_weights = {kernel.name: kernel, scale.name: scale}
        super().__init__(
            sub_weights,
            quant_config=None,
            name=src_weight_info.name,
            **{k: v for k, v in kwargs.items() if k != "name"},
        )
        self.kernel = kernel
        self.scale = scale

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        names = self.kernel.get_tensor_names(layer_id, load_config)
        names |= self.scale.get_tensor_names(layer_id, load_config)
        return names

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        kernel_dict = self.kernel._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        scale_dict = self.scale._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        return {
            self.kernel.name: kernel_dict[self.kernel.name],
            self.scale.name: scale_dict[self.scale.name],
        }

    def _split(self, tensor, load_config: LoadConfig):
        split_kernel = self.kernel._split(
            {self.kernel.name: tensor[self.kernel.name]}, load_config
        )
        split_scale = self.scale._split(
            {self.scale.name: tensor[self.scale.name]}, load_config
        )
        out: Dict[str, torch.Tensor] = {}
        out.update(split_kernel)
        out.update(split_scale)
        return out

    def _postprocess(self, tensor, device: str, load_config: LoadConfig):
        return {
            self.kernel.name: tensor[self.kernel.name],
            self.scale.name: tensor[self.scale.name],
        }


# ---------------------------------------------------------------------------
# Ckpt auto-detection
# ---------------------------------------------------------------------------

import json as _json
import os as _os
import re as _re

_OFFLINE_FP4_SCALE_RE = _re.compile(
    r"model\.layers\.\d+\.mlp\.(experts\.\d+|shared_experts)\."
    r"(gate|up|down)_proj\.weight_scale$"
)


def is_offline_mega_moe_fp4_ckpt(database: Optional[BaseDatabase]) -> bool:
    """Return True if the ckpt has pre-quantized FP4 MoE expert weights.

    Detection order:

    1. **Primary (cheap)**: read ``<ckpt>/config.json`` and check
       ``quantization_config.expert_dtype == "fp4"`` — self-describing flag
       emitted by ``tools/convert/glm5_fp4_moe_fp8_quant.py``.
    2. **Fallback (scan)**: any tensor name matches
       ``model.layers.{i}.mlp.experts.{j}.{gate|up|down}_proj.weight_scale``
       (handles older ckpts that pre-date the ``expert_dtype`` flag).
    """
    if database is None:
        return False

    # (1) config.json hint — preferred, single-file read
    path = getattr(database, "path", None)
    if path:
        cfg_path = _os.path.join(path, "config.json")
        if _os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = _json.load(f)
                qc = cfg.get("quantization_config") or {}
                if qc.get("expert_dtype") == "fp4":
                    return True
            except Exception:
                pass  # fall through to tensor scan

    # (2) fallback: scan tensor names
    try:
        names = database.get_pretrain_tensor_names()
    except Exception:
        return False
    for n in names:
        if _OFFLINE_FP4_SCALE_RE.search(n):
            return True
    return False


def wrap_moe_for_offline_fp4(weight: WeightModule) -> WeightModule:
    """Replace a MoE w1/w2 wrapper with offline FP4 loader.

    Handles both raw ``MoeAtomicWeight`` and ``PerBlockFp8Weight``-wrapped
    MoE (the latter happens when ``quant_method == "fp8"`` wraps MoE into
    PerBlockFp8Weight before we get a chance to intercept — auto-unwrap here).
    """
    from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight

    if isinstance(weight, PerBlockFp8Weight) and weight.name in _MEGA_MOE_KERNEL_NAMES:
        kernel = weight.kernel
        if kernel is None or not isinstance(kernel, MoeAtomicWeight):
            return weight
        return OfflineMegaMoeFp4MoeWeight(kernel)
    if isinstance(weight, MoeAtomicWeight) and weight.name in _MEGA_MOE_KERNEL_NAMES:
        return OfflineMegaMoeFp4MoeWeight(weight)
    return weight


def wrap_shared_expert_for_offline_fp4(weight: WeightModule) -> WeightModule:
    """Replace shared-expert FFN weights with offline FP4 loaders."""
    from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight

    if (
        isinstance(weight, PerBlockFp8Weight)
        and weight.name in _SHARED_EXPERT_KERNEL_NAMES
    ):
        kernel = weight.kernel
        if kernel is None or not isinstance(kernel, FfnAtomicWeight):
            return weight
        if _is_shared_expert_weight(kernel):
            return OfflineMegaMoeFp4SharedExpertWeight(kernel)
    if _is_shared_expert_weight(weight):
        return OfflineMegaMoeFp4SharedExpertWeight(weight)
    return weight
