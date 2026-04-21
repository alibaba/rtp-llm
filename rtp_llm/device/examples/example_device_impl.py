"""
新设备接入完整示例 - 以虚拟的 ExampleDevice 为例。

接入步骤：
  1. 复制本文件，替换为你的设备实现
  2. 在 device/__init__.py 的 get_device_cls() 添加 1 行映射

不需要改动的文件：
  - factory/attention/__init__.py   (选择器逻辑)
  - factory/attention/attn_factory.py (选择器逻辑)
  - factory/linear/__init__.py
  - factory/fused_moe/__init__.py
  - modules/base/__init__.py
"""

from rtp_llm.device.device_base import DeviceBase, MemInfo
from rtp_llm.ops.compute_ops import ExecCtxExporter


class ExampleDeviceImpl(DeviceBase):
    """示例设备实现。

    继承 DeviceBase 并实现以下方法组：
    - Attention 优先级（必须）
    - Base Ops 路径映射（必须）
    - Linear 模块路径（必须）
    - MoE 策略候选（如果支持 MoE）
    - 权重预处理（如果设备有特殊格式）
    - 内存查询（建议实现）
    """

    def __init__(self, exported_device: ExecCtxExporter):
        super().__init__(exported_device)
        # 设备特定的初始化逻辑

    # =========================================================================
    # Attention 优先级路由
    # =========================================================================
    # 返回你的设备支持的 attention 实现类列表，按优先级排序（高优先级在前）。
    # 框架会依次尝试每个实现，调用 impl.support() 检查兼容性。
    # 在方法体内 import 你的实现类，保证懒加载。

    def get_prefill_mha_priorities(self):
        # from rtp_llm.models_py.modules.factory.attention.example_impl import ExamplePrefillImpl
        # return [ExamplePrefillImpl]
        return []

    def get_decode_mha_priorities(self):
        # from rtp_llm.models_py.modules.factory.attention.example_impl import ExampleDecodeImpl
        # return [ExampleDecodeImpl]
        return []

    def get_prefill_mla_priorities(self):
        # 如果不支持 MLA，返回空列表
        return []

    def get_decode_mla_priorities(self):
        return []

    # =========================================================================
    # Base Ops 映射
    # =========================================================================
    # 返回 BaseOps NamedTuple，包含该设备的基础算子类。
    # 在方法体内 import modules/base/your_device/ 下的实现类。
    # BaseOps 的所有字段都必须提供，缺少任何一个会在构造时报 TypeError。

    def get_base_ops(self):
        # from rtp_llm.device.base_ops import BaseOps
        # from rtp_llm.models_py.modules.base.example.activation import FusedSiluAndMul
        # from rtp_llm.models_py.modules.base.example.norm import RMSNorm, ...
        # return BaseOps(
        #     FusedSiluAndMul=FusedSiluAndMul,
        #     RMSNorm=RMSNorm,
        #     ...
        # )
        raise NotImplementedError

    # =========================================================================
    # Linear 注册
    # =========================================================================
    # 导入你的 linear 实现模块，触发策略注册。
    # 你的模块内部应调用 LinearFactory.register(YourLinearStrategy) 注册策略。

    def register_linear_impl(self):
        # import rtp_llm.models_py.modules.factory.linear.impl.example  # noqa: F401
        pass

    # =========================================================================
    # MoE 策略候选
    # =========================================================================
    # 返回你的设备支持的 MoE 策略类列表。
    # 如果不支持 MoE，返回空列表。

    def get_moe_strategy_candidates(self):
        # from rtp_llm.models_py.modules.factory.fused_moe.impl.example.strategy import ExampleStrategy
        # return [ExampleStrategy]
        return []

    # =========================================================================
    # 内存查询（建议实现）
    # =========================================================================

    def _get_mem_info(self) -> MemInfo:
        # 返回设备的内存使用情况
        # return MemInfo(used=..., free=...)
        raise NotImplementedError

    # =========================================================================
    # 权重预处理（按需 override）
    # =========================================================================
    # 如果你的设备对权重有特殊格式要求，override 以下方法：
    #
    # - preprocess_groupwise_weight_params()  — GPTQ/AWQ 权重解包
    # - apply_int8() / moe_apply_int8()       — INT8 量化
    # - maybe_rewrite_weight_by_key()         — 按 key 重写权重
    # - convert_fp8_weight_params()           — FP8 权重转换
    #
    # 默认实现会抛出 NotImplementedError，如果你的设备不需要某种量化，
    # 可以不 override 对应方法。
