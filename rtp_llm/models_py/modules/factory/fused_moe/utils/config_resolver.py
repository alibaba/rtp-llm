"""Configuration resolver
Used to parse MOE configuration and extract MOE-related configuration information.
"""

from typing import Optional

import torch

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.utils.util import to_torch_dtype


class MoeConfigResolver:
    """MOE configuration resolver"""

    @staticmethod
    def get_device_type() -> DeviceType:
        """Get device type

        Returns:
            Device type
        """
        return get_device_type()

    @staticmethod
    def has_quantization(config: MoEConfigAdapter) -> bool:
        """Check if quantization is enabled for the MoE part of the model.

        Returns False even when ``model_config.quant_config`` is non-None if
        the quant config opts out of MoE quantization (``skip_moe=True``,
        e.g. ``FP8_PER_BLOCK_NO_MOE`` for the mega_moe-hybrid path), so the
        MoE strategy picker skips quantized strategies and falls back to a
        BF16 strategy that consumes the un-quantized MoE weights.
        """
        quant_config = config.model_config.quant_config
        if quant_config is None:
            return False
        if getattr(quant_config, "skip_moe", False):
            return False
        return True

    @staticmethod
    def is_bf16(config: MoEConfigAdapter) -> bool:
        """Check if data type is bf16

        Args:
            config: MOE configuration adapter

        Returns:
            Whether datatype is bf16
        """
        return to_torch_dtype(config.model_config.data_type) == torch.bfloat16

    @staticmethod
    def get_quant_method(config: MoEConfigAdapter) -> Optional[str]:
        """Get quantization method seen by the MoE strategy picker.

        Returns ``None`` for ``skip_moe=True`` quant configs so MoE strategies
        that gate on a specific FP8/FP4 method (e.g. ``CudaFp8PerBlockEpNormalStrategy``)
        do not match — the MoE weights are kept in compute dtype and need a
        BF16 strategy.
        """
        quant_config = config.model_config.quant_config
        if quant_config is None:
            return None
        if getattr(quant_config, "skip_moe", False):
            return None
        return quant_config.get_method()

    @staticmethod
    def is_ep_enabled(config: MoEConfigAdapter) -> bool:
        """Check if Expert Parallelism is enabled

        Args:
            config: MOE configuration adapter

        Returns:
            Whether EP is enabled
        """
        return config.ep_size > 1

    @staticmethod
    def use_low_latency(config: MoEConfigAdapter) -> bool:
        """Check if low latency mode is used

        Args:
            config: MOE configuration adapter

        Returns:
            Whether low latency mode is used
        """
        return config.moe_config.use_deepep_low_latency if config.moe_config else False

    @staticmethod
    def is_single_gpu(config: MoEConfigAdapter) -> bool:
        """Check if single GPU mode

        Args:
            config: MOE configuration adapter

        Returns:
            Whether single GPU
        """
        return config.ep_size == 1

    @staticmethod
    def is_tp_equal_ep(config: MoEConfigAdapter) -> bool:
        """Check if attention TP size equals EP size.

        Uses MoEConfigAdapter.tp_size, which reflects the attention/MoE-input
        view (= get_attn_tp_size(), == 1 when CP is enabled). In CP mode this
        returns False even if physical TP equals EP — use is_cp_equal_ep for
        the physical-topology check.
        """
        return config.tp_size == config.ep_size

    @staticmethod
    def is_cp_equal_ep(config: MoEConfigAdapter) -> bool:
        """Check if physical TP (= CP size when CP is enabled) equals EP size.

        Reads raw parallelism_config.tp_size, bypassing the adapter's tp_size
        which is squashed to 1 in CP mode. Used by router selectors that need
        to know the underlying TP topology (e.g. pure_cp_router).
        """
        return config.parallelism_config.tp_size == config.ep_size

    @staticmethod
    def is_pure_tp_mode(config: MoEConfigAdapter) -> bool:
        """Check if pure TP mode is applicable.

        Pure TP mode requires ep_size == 1 and dp_size == 1, meaning each
        rank holds all experts without EP/DP splitting. This covers both
        single-GPU (tp=1) and multi-GPU pure-TP (tp>1) scenarios. This
        aligns with the weight-splitting condition (moe_pure_tp_mode) in
        model_weight_info.py.

        Args:
            config: MOE configuration adapter

        Returns:
            Whether pure TP mode can be used
        """
        return config.ep_size == 1 and config.dp_size == 1 and config.tp_size >= 1

    @staticmethod
    def use_all_gather(config: MoEConfigAdapter) -> bool:
        """Check if use_all_gather is enabled

        Args:
            config: MOE configuration adapter

        Returns:
            Whether use_all_gather is enabled
        """
        return config.moe_config.use_all_gather if config.moe_config else True
