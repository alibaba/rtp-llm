import logging

import torch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops.compute_ops import DeviceType, ExecCtxExporter


class MemInfo:
    used: int = 0
    free: int = 0

    def __init__(self, used: int, free: int):
        self.used = used
        self.free = free


class DeviceBase:
    def __init__(self, exported_device: ExecCtxExporter):
        self.exported_device = exported_device
        from rtp_llm.config.server_config_setup import auto_configure_deepep
        from rtp_llm.server.server_args.server_args import setup_args

        self.py_env_configs = setup_args()
        auto_configure_deepep(
            moe_config=self.py_env_configs.moe_config,
            deep_ep_config=self.py_env_configs.deep_ep_config,
            parallelism_config=self.py_env_configs.parallelism_config,
            role_type=self.py_env_configs.role_config.role_type,
        )

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

    def maybe_prepare_static_weights_for_fp4_moe(
        self,
        kernel_name: str,
        scale_name: str,
        kernel: torch.Tensor,
        scale: torch.Tensor,
        **kwargs,
    ):
        return kernel, scale

    # ===== Attention 优先级路由 =====

    def get_prefill_mha_priorities(self) -> list:
        """返回该设备 prefill MHA 实现的优先级列表（高优先级在前）。"""
        return []

    def get_decode_mha_priorities(self) -> list:
        """返回该设备 decode MHA 实现的优先级列表（高优先级在前）。"""
        return []

    def get_prefill_mla_priorities(self) -> list:
        """返回该设备 prefill MLA 实现的优先级列表（高优先级在前）。"""
        return []

    def get_decode_mla_priorities(self) -> list:
        """返回该设备 decode MLA 实现的优先级列表（高优先级在前）。"""
        return []

    # ===== Base Ops 分派 =====

    def get_base_ops(self):
        """返回 BaseOps NamedTuple，包含该设备的基础算子类。"""
        raise NotImplementedError("get_base_ops is not implemented")

    # ===== Linear 分派 =====

    def register_linear_impl(self):
        """导入该设备的 linear 实现模块，触发策略注册。"""
        pass

    # ===== MoE 策略路由 =====

    def get_moe_strategy_candidates(self) -> list:
        """返回该设备支持的 MoE 策略候选列表。"""
        return []

    # ===== 能力查询 =====

    def is_cuda(self) -> bool:
        return self.get_device_type() == DeviceType.Cuda

    def is_rocm(self) -> bool:
        return self.get_device_type() == DeviceType.ROCm

    @property
    def supports_fp4(self) -> bool:
        return False

    @property
    def supports_fp8(self) -> bool:
        return False
