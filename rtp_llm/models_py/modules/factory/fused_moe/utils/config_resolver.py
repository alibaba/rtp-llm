"""Configuration resolver

Used to parse GptInitModelParameters configuration and extract MOE-related configuration information.
"""

from typing import Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
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
    def has_quantization(config: GptInitModelParameters) -> bool:
        """Check if quantization is enabled

        Args:
            config: Model initialization parameters

        Returns:
            Whether quantization is enabled
        """
        return config.quant_config is not None

    @staticmethod
    def is_bf16(config: GptInitModelParameters) -> bool:
        """Check if data type is bf16

        Args:
            config: Model initialization parameters

        Returns:
            Whether datatype is bf16
        """
        return to_torch_dtype(config.data_type) == torch.bfloat16

    @staticmethod
    def get_quant_method(config: GptInitModelParameters) -> Optional[str]:
        """Get quantization method

        Args:
            config: Model initialization parameters

        Returns:
            Quantization method name, or None if quantization is not enabled
        """
        if config.quant_config is None:
            return None
        return config.quant_config.get_method()

    @staticmethod
    def is_ep_enabled(config: GptInitModelParameters) -> bool:
        """Check if Expert Parallelism is enabled

        Args:
            config: Model initialization parameters

        Returns:
            Whether EP is enabled
        """
        return config.ep_size > 1

    @staticmethod
    def use_low_latency(config: GptInitModelParameters) -> bool:
        """Check if low latency mode is used

        Args:
            config: Model initialization parameters

        Returns:
            Whether low latency mode is used
        """
        return config.moe_config.use_deepep_low_latency

    @staticmethod
    def is_single_gpu(config: GptInitModelParameters) -> bool:
        """Check if single GPU mode

        Args:
            config: Model initialization parameters

        Returns:
            Whether single GPU
        """
        return config.ep_size == 1

    @staticmethod
    def is_tp_equal_ep(config: GptInitModelParameters) -> bool:
        """Check if TP size equals EP size

        Args:
            config: Model initialization parameters

        Returns:
            Whether TP size equals EP size
        """
        return config.tp_size == config.ep_size
