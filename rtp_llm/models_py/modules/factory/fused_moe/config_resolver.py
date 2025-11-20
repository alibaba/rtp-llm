"""Configuration resolver

Used to parse GptInitModelParameters configuration and extract MOE-related configuration information.
"""

from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.runtime_config import RuntimeConfig
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
    def has_quantization(config: RuntimeConfig) -> bool:
        """Check if quantization is enabled

        Args:
            config: Model initialization parameters

        Returns:
            Whether quantization is enabled
        """
        return config.model_config.quant_config is not None

    @staticmethod
    def is_bf16(config: RuntimeConfig) -> bool:
        """Check if data type is bf16

        Args:
            config: Model initialization parameters

        Returns:
            Whether datatype is bf16
        """
        return to_torch_dtype(config.model_config.data_type) == torch.bfloat16

    @staticmethod
    def get_quant_method(config: RuntimeConfig) -> Optional[str]:
        """Get quantization method

        Args:
            config: Model initialization parameters

        Returns:
            Quantization method name, or None if quantization is not enabled
        """
        if config.model_config.quant_config is None:
            return None
        return config.model_config.quant_config.get_method()

    @staticmethod
    def is_ep_enabled(config: RuntimeConfig) -> bool:
        """Check if Expert Parallelism is enabled

        Args:
            config: Model initialization parameters

        Returns:
            Whether EP is enabled
        """
        return config.model_config.ep_size > 1

    @staticmethod
    def use_low_latency(config: RuntimeConfig) -> bool:
        """Check if low latency mode is used

        Args:
            config: Model initialization parameters

        Returns:
            Whether low latency mode is used
        """
        return config.use_deepep_low_latency

    @staticmethod
    def is_single_gpu(config: RuntimeConfig) -> bool:
        """Check if single GPU mode

        Args:
            config: Model initialization parameters

        Returns:
            Whether single GPU
        """
        return config.model_config.ep_size == 1

    @staticmethod
    def is_tp_equal_ep(config: RuntimeConfig) -> bool:
        """Check if TP size equals EP size

        Args:
            config: Model initialization parameters

        Returns:
            Whether TP size equals EP size
        """
        return config.model_config.tp_size == config.model_config.ep_size
