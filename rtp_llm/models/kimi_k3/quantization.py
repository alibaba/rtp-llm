"""Selective load-time FP8 quantization for Kimi K3 attention weights."""

from __future__ import annotations

import os
from typing import Optional, Union

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    LoadQuantPerBlockFp8Weight,
    per_block_cast_to_fp8,
)
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.utils.model_weight import W


_KDA_WEIGHT_NAMES = {
    W.linear_attn_qkvg_w,
    W.linear_attn_f_a_w,
    W.linear_attn_f_b_w,
    W.linear_attn_b_w,
    W.linear_attn_out_w,
}
_MLA_WEIGHT_NAMES = {
    W.mla_fusedqkrope_w,
    W.mla_q_b_w,
    W.mla_kv_b_w,
    W.attn_gate_w,
    W.attn_o_w,
}


def get_kimi_k3_load_time_fp8_config(
    *, is_kda: bool
) -> Optional[Fp8BlockWiseQuantConfig]:
    """Return the local FP8 config selected for one K3 attention type."""

    env_name = "KIMI_K3_W8A8_KDA" if is_kda else "KIMI_K3_W8A8_MLA"
    raw = os.environ.get(env_name, "0").strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{env_name} must be 0 or 1, got {raw!r}")
    if raw == "0":
        return None
    return Fp8BlockWiseQuantConfig(bits=8, group_size=128, is_quanted=False)


def quantize_rank_local_fp8(
    weight: torch.Tensor,
    *,
    group_size: int,
    pad_output_to: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a local ``[in, out]`` matrix, optionally padding output."""

    if weight.ndim != 2:
        raise ValueError(
            f"Kimi K3 load-time FP8 expects a matrix, got {tuple(weight.shape)}"
        )
    if pad_output_to is not None:
        if pad_output_to <= 0:
            raise ValueError(
                f"Kimi K3 FP8 output alignment must be positive, got {pad_output_to}"
            )
        pad_columns = (-weight.shape[1]) % pad_output_to
        if pad_columns:
            weight = torch.cat(
                (weight, weight.new_zeros((weight.shape[0], pad_columns))),
                dim=1,
            )
    quantized, scale = per_block_cast_to_fp8(weight, group_size)
    return quantized.T.contiguous(), scale.T.contiguous()


class KimiK3LoadTimeFp8Weight(LoadQuantPerBlockFp8Weight):
    """Split a BF16 K3 projection first, then quantize only the local shard."""

    w8a8_weight_list = {
        **LoadQuantPerBlockFp8Weight.w8a8_weight_list,
        W.linear_attn_qkvg_w: W.linear_attn_qkvg_s,
        W.linear_attn_f_a_w: W.linear_attn_f_a_s,
        W.linear_attn_f_b_w: W.linear_attn_f_b_s,
        W.linear_attn_b_w: W.linear_attn_b_s,
    }

    @classmethod
    def support(cls, quant_config, src_weight_info: WeightModule) -> bool:
        # K3 constructs this wrapper explicitly; never select it globally.
        return False

    def __init__(
        self,
        src_weight_info: AtomicWeight,
        quant_config: Fp8BlockWiseQuantConfig,
    ) -> None:
        self._source_weight = src_weight_info
        super().__init__(src_weight_info, quant_config, name=src_weight_info.name)

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        merged = self._source_weight._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        rank_local = self._source_weight._split(merged, load_config)
        weight = rank_local[self._source_weight.name]
        quantized, scale = quantize_rank_local_fp8(
            weight,
            group_size=self.group_size,
            pad_output_to=(
                self.group_size if self._source_weight.name == W.linear_attn_b_w else None
            ),
        )
        return {
            self.kernel.name: quantized,
            self.scale.name: scale,
        }

    def _split(self, tensor, load_config: LoadConfig):
        # The source weight was already split before quantization above.
        return tensor


def wrap_kimi_k3_load_time_fp8_weight(
    weight: AtomicWeight, *, is_kda: bool
) -> Union[AtomicWeight, KimiK3LoadTimeFp8Weight]:
    """Wrap only KDA/MLA projections selected for load-time W8A8."""

    allowed_names = _KDA_WEIGHT_NAMES if is_kda else _MLA_WEIGHT_NAMES
    if weight.name not in allowed_names:
        return weight
    config = get_kimi_k3_load_time_fp8_config(is_kda=is_kda)
    if config is None:
        return weight
    return KimiK3LoadTimeFp8Weight(weight, config)
