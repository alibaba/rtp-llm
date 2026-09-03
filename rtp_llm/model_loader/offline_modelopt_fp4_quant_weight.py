"""Offline FP4 mega-MoE weight loader.

Reads pre-quantized FP4+UE8M0 MoE weights directly from checkpoint, produced
by ``glm5_fp4_moe_fp8_quant.py``:

  - ``{expert_prefix}.weight``  int8 (FP4 packed)
  - ``{expert_prefix}.scale``   UE8M0 (``float8_e8m0fnu``); legacy
    float32 ckpts use ``.weight_scale``

Shared-expert / dense FP8 scales use ``.scale`` (UE8M0) or legacy
``.weight_scale_inv``.

Auto-detected by ``is_offline_mega_moe_fp4_ckpt`` via
``quantization_config.expert_dtype == "fp4"`` (preferred) or legacy
``.weight_scale`` tensor names.
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
from rtp_llm.utils.model_weight import CkptWeightInfo, W, concat_0, identity

_MEGA_MOE_KERNEL_NAMES = (W.moe_w1, W.moe_w2)
_SHARED_EXPERT_KERNEL_NAMES = (W.ffn_w13, W.ffn_w2)

_FP4_W_SUFFIX = ".weight"
_FP4_S_SUFFIX_LEGACY = ".weight_scale"
_FP4_S_SUFFIX_UE8M0 = ".scale"
_FP8_S_SUFFIX_LEGACY = ".weight_scale_inv"
_FP8_S_SUFFIX_UE8M0 = ".scale"


def _fp4_scale_suffix(scale_dtype: torch.dtype) -> str:
    if scale_dtype == torch.float8_e8m0fnu:
        return _FP4_S_SUFFIX_UE8M0
    return _FP4_S_SUFFIX_LEGACY


def _fp8_scale_suffix(scale_dtype: torch.dtype) -> str:
    if scale_dtype == torch.float8_e8m0fnu:
        return _FP8_S_SUFFIX_UE8M0
    return _FP8_S_SUFFIX_LEGACY


def _fp4_scale_name(weight_name: str, scale_dtype: torch.dtype) -> str:
    # ModelOpt DeepSeek-style fused MXFP4 tensors keep ``weight`` in the
    # runtime-facing base name: w13_weight -> w13_weight_scale.
    if weight_name.endswith((".w13_weight", ".w2_weight")):
        return weight_name + "_scale"
    if weight_name.endswith(_FP4_W_SUFFIX):
        return weight_name[: -len(_FP4_W_SUFFIX)] + _fp4_scale_suffix(scale_dtype)
    return weight_name + _fp4_scale_suffix(scale_dtype)


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
        scale_dtype: torch.dtype = torch.float32,
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
        direct_modelopt_mxfp4 = any(
            w.name.endswith((".w13_weight", ".w2_weight"))
            for w in src_weight_info.weights
        )
        if direct_modelopt_mxfp4 and scale_dtype != torch.float8_e8m0fnu:
            raise ValueError(
                "ModelOpt fused MXFP4 weights require raw UE8M0 scale loading"
            )
        scale_weights = [
            CkptWeightInfo(
                _fp4_scale_name(w.name, scale_dtype),
                w.merge_fun,
            )
            for w in src_weight_info.weights
        ]
        scale = MoeAtomicWeight(
            name=_mega_moe_scale_name(src_weight_info.name),
            weights=scale_weights,
            process_fun=src_weight_info.process_fun,
            # This checkpoint serializes exponent bits as U8. Reinterpret the
            # bytes after split; a numeric U8->float8 cast changes the bits.
            data_type=torch.uint8 if direct_modelopt_mxfp4 else scale_dtype,
            config=src_weight_info.config,
            stacked_ckpt_keys=getattr(src_weight_info, "stacked_ckpt_keys", False),
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
        self._direct_modelopt_mxfp4 = direct_modelopt_mxfp4

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
        scale = tensor[self.scale.name]
        if self._direct_modelopt_mxfp4:
            if scale.dtype != torch.uint8:
                raise TypeError(
                    "ModelOpt fused MXFP4 scale must remain uint8 until "
                    f"reinterpretation, got {scale.dtype}"
                )
            scale = scale.view(torch.float8_e8m0fnu)
        return {
            self.kernel.name: tensor[self.kernel.name],
            self.scale.name: scale,
        }


class OfflineMegaMoeFp8SharedExpertWeight(CompositeWeight, QuantWeight):
    """Load pre-quantized FP8 per-block shared-expert weights for fused MegaMoE.

    Shared expert is FP8 e4m3 + 128x128 UE8M0 scale (``.scale`` / legacy
    ``.weight_scale_inv``). gate/up stacked with ``concat_0`` (TP=1 only).
    """

    shared_weight_list = list(_SHARED_EXPERT_KERNEL_NAMES)

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        return False

    def __init__(
        self,
        src_weight_info: FfnAtomicWeight,
        scale_dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        if not _is_shared_expert_weight(src_weight_info):
            raise ValueError(
                "OfflineMegaMoeFp8SharedExpertWeight only wraps shared_experts "
                f"{_SHARED_EXPERT_KERNEL_NAMES}, got {src_weight_info}"
            )

        process_fun = concat_0 if src_weight_info.name == W.ffn_w13 else identity
        kernel = FfnAtomicWeight(
            name=src_weight_info.name,
            weights=src_weight_info.weights,
            process_fun=process_fun,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        s_suffix = _fp8_scale_suffix(scale_dtype)
        scale_weights = [
            CkptWeightInfo(
                w.name[: -len(_FP4_W_SUFFIX)] + s_suffix,
                w.merge_fun,
            )
            for w in src_weight_info.weights
        ]
        scale = FfnAtomicWeight(
            name=_shared_expert_scale_name(src_weight_info.name),
            weights=scale_weights,
            process_fun=process_fun,
            data_type=scale_dtype,
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
        if load_config.tp_size != 1 or load_config.ffn_tp_size != 1:
            raise ValueError(
                "OfflineMegaMoeFp8SharedExpertWeight assumes tp_size="
                f"ffn_tp_size=1, got tp_size={load_config.tp_size}, "
                f"ffn_tp_size={load_config.ffn_tp_size}"
            )
        # The fused shared expert is replicated on every EP/DP rank.  Generic
        # FfnAtomicWeight splitting keys off ep_size/dp_size too, and would run
        # ffn_sp_neg1_w13 even though ffn_tp_size is one.  Besides needlessly
        # rebuilding an already-full tensor, that path calls torch.cat on the
        # UE8M0 scale, which CUDA does not implement.  Keep both tensors intact;
        # routed experts are still sharded by their separate MoE loader.
        return {
            self.kernel.name: tensor[self.kernel.name],
            self.scale.name: tensor[self.scale.name],
        }

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

# Legacy float32 FP4 MoE scale name. New UE8M0 ckpts use ``.scale`` for both
# FP4 MoE and FP8 linears — those must not be used for FP4 detection (all-FP8
# ckpts share the same ``.scale`` suffix); rely on ``expert_dtype`` instead.
_OFFLINE_FP4_SCALE_RE = _re.compile(
    r"model\.(?:layers|mtp_layers)\.\d+\.mlp\.experts\."
    r"(?:\d+\.(?:gate|up|down)_proj\.weight_scale|"
    r"(?:w13|w2)_weight_scale)$"
)


def is_offline_mega_moe_fp4_ckpt(database: Optional[BaseDatabase]) -> bool:
    """Return True if the ckpt has pre-quantized FP4 MoE expert weights.

    Detection order:

    1. **Primary**: ``quantization_config.expert_dtype == "fp4"``
       (emitted by ``glm5_fp4_moe_fp8_quant.py`` when not ``all_fp8``).
    2. **Fallback**: legacy ``experts.*.weight_scale`` tensor names.
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
                modelopt_quant = qc.get("quantization") or {}
                quantized_layers = modelopt_quant.get("quantized_layers") or {}
                if any(
                    isinstance(info, dict)
                    and str(info.get("quant_algo", "")).upper() == "MXFP4"
                    for info in quantized_layers.values()
                ):
                    return True
            except Exception:
                pass  # fall through to tensor scan

    # (2) fallback: legacy ``.weight_scale`` only (not ``.scale``)
    try:
        names = database.get_pretrain_tensor_names()
    except Exception:
        return False
    for n in names:
        if _OFFLINE_FP4_SCALE_RE.search(n):
            return True
    return False


def wrap_moe_for_offline_fp4(
    weight: WeightModule, scale_dtype: torch.dtype = torch.float32
) -> WeightModule:
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
        return OfflineMegaMoeFp4MoeWeight(kernel, scale_dtype=scale_dtype)
    if isinstance(weight, MoeAtomicWeight) and weight.name in _MEGA_MOE_KERNEL_NAMES:
        return OfflineMegaMoeFp4MoeWeight(weight, scale_dtype=scale_dtype)
    return weight


def wrap_shared_expert_for_offline_fp4(
    weight: WeightModule, scale_dtype: torch.dtype = torch.float32
) -> WeightModule:
    """Replace shared-expert FFN weights with the offline FP8 per-block loader.

    ``mega_moe_se`` consumes the shared expert as FP8 e4m3 per-block weights
    through the unified ``deep_gemm.fp8_fp4_mega_moe`` optional-shared API.
    The legacy ``mega_moe_fused`` strategy uses the same checkpoint contract.
    """
    from rtp_llm.model_loader.mxfp8_quant_weight import Mxfp8Weight
    from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight

    # ModelOpt mixed checkpoints already wrap shared experts with Mxfp8Weight.
    # Keep that wrapper: it knows the checkpoint's ``.weight_scale`` name and
    # converts raw UE8M0 exponent bytes into the corresponding fp32 powers of
    # two. MegaMoE-SE performs only the final DeepGEMM layout transform.
    if isinstance(weight, Mxfp8Weight):
        return weight

    if (
        isinstance(weight, PerBlockFp8Weight)
        and weight.name in _SHARED_EXPERT_KERNEL_NAMES
    ):
        kernel = weight.kernel
        if kernel is None or not isinstance(kernel, FfnAtomicWeight):
            return weight
        if _is_shared_expert_weight(kernel):
            return OfflineMegaMoeFp8SharedExpertWeight(kernel, scale_dtype=scale_dtype)
    if _is_shared_expert_weight(weight):
        return OfflineMegaMoeFp8SharedExpertWeight(weight, scale_dtype=scale_dtype)
    return weight


def wrap_for_offline_fp4(
    weight: WeightModule,
    include_shared_expert: bool = False,
    scale_dtype: torch.dtype = torch.float32,
) -> WeightModule:
    """Replace offline FP4 MoE weights, optionally including shared experts.

    ``mega_moe`` only consumes routed expert weights. ``mega_moe_se`` and
    ``mega_moe_fused`` also wrap shared-expert FP8 per-block weights.
    """
    wrapped = wrap_moe_for_offline_fp4(weight, scale_dtype=scale_dtype)
    if wrapped is not weight or not include_shared_expert:
        return wrapped
    return wrap_shared_expert_for_offline_fp4(weight, scale_dtype=scale_dtype)
