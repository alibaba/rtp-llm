"""Configuration resolver
Used to parse MOE configuration and extract MOE-related configuration information.
"""

from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops.compute_ops import DeviceType, get_device
from rtp_llm.utils.util import to_torch_dtype


class MoeConfigResolver:
    """MOE configuration resolver"""

    @staticmethod
    def get_device_type() -> DeviceType:
        """Get device type

        Returns:
            Device type
        """
        return get_device().get_device_type()

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
        return config.parallelism_config.tp_size == config.parallelism_config.ep_size

    @staticmethod
    def use_all_gather(config: MoEConfigAdapter) -> bool:
        """Check if use_all_gather is enabled

        Args:
            config: MOE configuration adapter

        Returns:
            Whether use_all_gather is enabled
        """
        return config.moe_config.use_all_gather if config.moe_config else True

    @staticmethod
    def is_afd_enabled(config: GptInitModelParameters) -> bool:
        """Check if AF disaggregate is enabled

        Args:
            config: Model initialization parameters

        Returns:
            Whether AF disaggregate is enabled
        """
        return config.ffn_disaggregate_config.enable_ffn_disaggregate

    @staticmethod
    def is_afd_ffn_rank(config: GptInitModelParameters) -> bool:
        """Check if AF disaggregate FFN rank

        Args:
            config: Model initialization parameters

        Returns:
            Whether AF disaggregate FFN rank
        """
        return config.ffn_disaggregate_config.is_ffn_service()
