"""Configuration resolver
Used to parse MOE configuration and extract MOE-related configuration information.
"""

from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops.compute_ops import DeviceType, get_exec_ctx
from rtp_llm.utils.util import to_torch_dtype


class MoeConfigResolver:
    """MOE configuration resolver"""

    @staticmethod
    def get_device_type() -> DeviceType:
        """Get device type

        Returns:
            Device type
        """
        return get_exec_ctx().get_device_type()

    @staticmethod
    def has_quantization(config: MoEConfigAdapter) -> bool:
        """Check if quantization is enabled

        Args:
            config: MOE configuration adapter

        Returns:
            Whether quantization is enabled
        """
        return config.model_config.quant_config is not None

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
        """Get quantization method

        Args:
            config: MOE configuration adapter

        Returns:
            Quantization method name, or None if quantization is not enabled
        """
        if config.model_config.quant_config is None:
            return None
        return config.model_config.quant_config.get_method()

    @staticmethod
    def is_ep_enabled(config: MoEConfigAdapter) -> bool:
        """Check if Expert Parallelism is enabled

        Args:
            config: MOE configuration adapter

        Returns:
            Whether EP is enabled
        """
        return config.parallelism_config.ep_size > 1

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
        return config.parallelism_config.ep_size == 1

    @staticmethod
    def is_tp_equal_ep(config: MoEConfigAdapter) -> bool:
        """Check if TP size equals EP size

        Args:
            config: MOE configuration adapter

        Returns:
            Whether TP size equals EP size
        """
        return config.tp_size == config.parallelism_config.ep_size

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
        return (
            config.parallelism_config.ep_size == 1
            and config.parallelism_config.dp_size == 1
            and config.parallelism_config.tp_size >= 1
        )

    @staticmethod
    def use_all_gather(config: MoEConfigAdapter) -> bool:
        """Check if use_all_gather is enabled

        Args:
            config: MOE configuration adapter

        Returns:
            Whether use_all_gather is enabled
        """
        return config.moe_config.use_all_gather if config.moe_config else True
