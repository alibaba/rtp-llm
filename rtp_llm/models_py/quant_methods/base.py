"""新 loader 的量化抽象（vLLM 风格）。

双层抽象：
- ``QuantizationConfig``：描述「用什么量化方案」（fp8/awq/gptq...），由上游
  ``config/quant_config.py`` 的 ``load_from_ckpt`` 解析 ckpt 得到（走法1：本类只
  *携带* 那个富对象 ``source_config``，不在此重复解析）。
- ``QuantizeMethodBase``：描述「具体怎么建权重、怎么算」，三段式钩子
  ``create_weights`` / ``apply`` / ``process_weights_after_loading``。

按层类型分流（对齐 vLLM 的 ``isinstance(layer)`` 派发）：
- Linear 层 → ``LinearMethodBase`` 子类（二维权重，按 input/output 切分）；
- MoE 层   → ``FusedMoEMethodBase`` 子类（带 expert 维三维权重 + 路由信息）。
两者共享三个钩子名，但签名不同，所以是两条独立子树、两套注册表。

``get_quant_method(layer, prefix)`` 据此派发，并支持 ``prefix`` 命中 ignore 列表
（混合精度 / compressed-tensors 的 ignore / lm_head 不量化）时回退未量化。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import torch

if TYPE_CHECKING:
    from rtp_llm.models_py.layers.linear import LinearBase


class QuantizeMethodBase(ABC):
    """量化方法三段式契约的顶层基类。

    Linear 与 MoE 共享这三个钩子名;具体签名见 ``LinearMethodBase`` /
    ``FusedMoEMethodBase``。
    """

    def __init__(self, quant_config: "Any" = None):
        # 统一持有 QuantizationConfig（携带 source_config:group_size/ignore 等）。
        # 现有 fp8 方法不重写 __init__、也不读它,行为不变;AWQ 等需要 group_size 的
        # 方法从这里取（见 AWQLinearMethod）。
        self.quant_config = quant_config

    @abstractmethod
    def create_weights(
        self,
        layer: "LinearBase",
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self, layer: "LinearBase", x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: "LinearBase"):
        raise NotImplementedError


class LinearMethodBase(QuantizeMethodBase):
    """Linear（二维权重，按 input/output_partition 切分）量化方法基类。

    现有 ``Fp8*LinearMethod`` / ``UnquantizedLinearMethod`` 语义上都属于本类;
    新增的 Linear 量化方法应继承本类并用 ``@register_quant_method`` 注册。
    签名沿用 ``QuantizeMethodBase``（``create_weights(layer, input_size,
    output_size, params_dtype)`` / ``apply(layer, x, bias)``）。
    """

    pass


class FusedMoEMethodBase(QuantizeMethodBase):
    """MoE（带 expert 维三维权重 + 路由）量化方法基类。

    与 Linear 的关键区别:``create_weights`` 按 ``num_experts/hidden_size/
    intermediate_size`` 建三维（含 expert 维 + per-expert scale）权重。

    rtp-llm 结构下 MoE 的前向计算由层（``BaseMoEExperts.forward`` → ``fused_moe``）
    负责,量化在「建权重 / 加载 scale / 加载后融合·在线量化 / 把 scale 喂给
    fused_moe」这几步,因此本类的契约是:
      - ``create_weights(layer, num_experts, hidden_size, intermediate_size, params_dtype)``
        建 w13/w2 + scale buffer;
      - ``process_weights_after_loading(layer)`` 做融合 / 在线量化;
      - 可选 ``dispatch_scale(layer, local_id, proj, param_name, tensor)`` 流式加载
        期路由 scale 张量;
      - 可选 ``add_weight_tensors(layer, weights_dict)`` 把 scale 加进喂给
        FusedMoeFactory 的权重字典。
    ``apply`` 在 MoE 路径不使用（前向由层调 fused_moe）,给个默认实现避免强制重写。
    新增的 MoE 量化方法继承本类并用 ``@register_moe_quant_method`` 注册。
    """

    @abstractmethod
    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer):
        raise NotImplementedError

    # --- 可选钩子（默认 no-op），具体方法按需重写 ---

    def dispatch_scale(self, layer, local_id: int, proj: str, param_name: str, tensor):
        """流式加载期:把一个 scale/meta 张量路由到对应 buffer。默认忽略。"""
        return None

    def dispatch_weight(self, layer, local_id: int, proj: str, param_name: str, tensor):
        """流式加载期:让量化方法接管非标准权重张量。返回 True 表示已处理。"""
        return False

    def add_weight_tensors(self, layer, weights_dict: Dict[str, Any]) -> None:
        """把量化 scale 等加进喂给 FusedMoeFactory 的权重字典。默认不加。"""
        return None

    # MoE 前向由层（fused_moe）负责,本钩子不用;给默认实现避免强制重写。
    def apply(self, layer, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "FusedMoEMethodBase.apply 不应被调用:MoE 前向由 BaseMoEExperts.forward "
            "经 fused_moe 完成。"
        )


# 两套注册表:同一 quant_type 字符串可分别注册 Linear 与 MoE 两种实现，
# get_quant_method 按层类型选其一。
_LINEAR_METHOD_REGISTRY: Dict[str, Type[QuantizeMethodBase]] = {}
_MOE_METHOD_REGISTRY: Dict[str, Type[FusedMoEMethodBase]] = {}

# 向后兼容别名:历史代码里的 _RUNTIME_METHOD_REGISTRY 即 Linear 注册表。
_RUNTIME_METHOD_REGISTRY = _LINEAR_METHOD_REGISTRY


def _register(registry: Dict[str, Type], keys, cls):
    for k in keys:
        existing = registry.get(k)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"quant method key '{k}' already registered to "
                f"{existing.__name__}, cannot re-register to {cls.__name__}"
            )
        registry[k] = cls
    return cls


def register_quant_method(*keys: str):
    """注册 Linear 量化方法（名字保持向后兼容）。"""

    def deco(cls: Type[QuantizeMethodBase]) -> Type[QuantizeMethodBase]:
        return _register(_LINEAR_METHOD_REGISTRY, keys, cls)

    return deco


def register_moe_quant_method(*keys: str):
    """注册 MoE 量化方法。"""

    def deco(cls: Type[FusedMoEMethodBase]) -> Type[FusedMoEMethodBase]:
        return _register(_MOE_METHOD_REGISTRY, keys, cls)

    return deco


class QuantizationConfig:
    """量化配置载体（走法1:携带旧 loader 解析出的富对象，不重复解析 ckpt）。"""

    def __init__(
        self,
        quant_type: str = "none",
        source_config: Any = None,
        ignored_layers: Optional[List[str]] = None,
    ):
        self.quant_type = quant_type
        # 旧 config/quant_config.py 的 load_from_ckpt 已把 ckpt 解析成富对象
        # （带 dynamic / scale_suffix / group_size / ignore 等结构化字段）。
        # 这里只携带它，供各 method 读取，避免在新 loader 重复解析。
        self.source_config = source_config
        # 混合精度 / compressed-tensors 的 ignore 列表:命中的模块走未量化。
        self.ignored_layers = (
            ignored_layers
            if ignored_layers is not None
            else self._extract_ignored(source_config)
        )
        self.weight_block_size = getattr(source_config, "weight_block_size", None)
        if self.weight_block_size is None:
            self.weight_block_size = [128, 128]

    @staticmethod
    def _extract_ignored(source_config: Any) -> List[str]:
        if source_config is None:
            return []
        for attr in ("ignore_patterns", "ignored_layers", "ignore"):
            v = getattr(source_config, attr, None)
            if v:
                return list(v)
        return []

    def is_layer_ignored(self, prefix: str) -> bool:
        """模块名（prefix）是否在 ignore 列表里 → 该模块不量化。

        采用点边界（.）分段前缀匹配与 fnmatch 通配符支持，避免子串包含匹配（如 gate 误伤 gate_proj）。
        """
        if not prefix or not self.ignored_layers:
            return False

        for pat in self.ignored_layers:
            if not pat:
                continue
            # 1. 通配符模式匹配
            if "*" in pat or "?" in pat:
                import fnmatch

                if fnmatch.fnmatch(prefix, pat) or fnmatch.fnmatch(prefix, f"{pat}.*"):
                    return True
            # 2. 精确点分割路径前缀匹配
            else:
                prefix_parts = prefix.split(".")
                pat_parts = pat.split(".")
                if len(pat_parts) <= len(prefix_parts):
                    if prefix_parts[: len(pat_parts)] == pat_parts:
                        return True
        return False

    @staticmethod
    def _is_moe_layer(layer) -> bool:
        # 惰性 import 避免环依赖（moe_experts 反过来 import 本模块）。
        try:
            from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
        except Exception:
            return False
        return isinstance(layer, BaseMoEExperts)

    def get_quant_method(self, layer, prefix: str = "") -> Optional[QuantizeMethodBase]:
        """按层类型 + prefix 派发量化方法。

        - prefix 命中 ignore → 未量化;
        - MoE 层 → MoE 注册表（无对应实现则返回 None，交回 MoE 模块自身处理）;
        - 其余（Linear 层）→ Linear 注册表（无对应实现则未量化）。
        """
        from rtp_llm.models_py.quant_methods.unquantized import (
            UnquantizedFusedMoEMethod,
            UnquantizedLinearMethod,
        )

        is_moe = self._is_moe_layer(layer)

        # prefix 命中 ignore → 未量化（按层类型选对应的未量化方法）。
        if self.is_layer_ignored(prefix):
            return (
                UnquantizedFusedMoEMethod(self) if is_moe else UnquantizedLinearMethod()
            )

        if is_moe:
            cls = _MOE_METHOD_REGISTRY.get(self.quant_type)
            # 无对应 MoE 方法 → None,调用方（MoE 层）回退内置逻辑
            # （fp8/fp4/w4a8 当前仍走内置;bf16/"none" 已注册 UnquantizedFusedMoEMethod）。
            return cls(self) if cls is not None else None

        cls = _LINEAR_METHOD_REGISTRY.get(self.quant_type)
        # 传入 config（AWQ 等需要 group_size 等;fp8/unquantized 继承基类 __init__ 接收后忽略）。
        return cls(self) if cls is not None else UnquantizedLinearMethod()
