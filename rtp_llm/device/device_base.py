import logging

import torch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops.compute_ops import DeviceExporter, DeviceType


class MemInfo:
    used: int = 0
    free: int = 0

    def __init__(self, used: int, free: int):
        self.used = used
        self.free = free


class DeviceBase:
    def __init__(self, exported_device: DeviceExporter):
        self.exported_device = exported_device
        from rtp_llm.server.server_args.server_args import setup_args
        self.py_env_configs = setup_args()

    def get_device_type(self) -> DeviceType:
        return self.exported_device.get_device_type()

    def get_device_id(self) -> int:
        return self.exported_device.get_device_id()

    @property
    def support_dio_load(self) -> bool:
        return False

    def _get_mem_info(self) -> MemInfo:
        raise NotImplementedError("_get_mem_info is not implemented")

    def get_mem_info(self) -> MemInfo:
        try:
            return self._get_mem_info()
        except Exception as e:
            logging.warning(f"get_mem_info failed: {e}")
            return None

    def preprocess_groupwise_weight_params(
        self,
        qweight_int32,
        qzeros_int32,
        scales_fp16,
        device: str,
        gptq: bool,
        awq: bool,
        weight_bits: int,
    ):
        raise NotImplementedError(
            "preprocess_groupwise_weight_params is not implemented"
        )

    def preprocess_moe_groupwise_weight_params(
        self,
        qweight_int32,
        qzeros_int32,
        scales_fp16,
        device: str,
        gptq: bool,
        awq: bool,
        weight_bits: int,
    ):
        raise NotImplementedError(
            "preprocess_moe_groupwise_weight_params is not implemented"
        )

    def apply_int8(self, tensor: torch.Tensor, device: str):
        raise NotImplementedError("apply_int8 is not implemented")

    def moe_apply_int8(self, tensor: torch.Tensor, device: str):
        raise NotImplementedError("moe_apply_int8 is not implemented")

    def maybe_rewrite_weight_by_key(
        self, key: str, weight: torch.Tensor
    ) -> torch.Tensor:
        return weight
