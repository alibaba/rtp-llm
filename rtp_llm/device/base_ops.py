from typing import NamedTuple, Type


class BaseOps(NamedTuple):
    """设备需要提供的基础算子集合。

    新设备实现 get_base_ops() 时返回此类型，缺少任何字段会在构造时报错。
    """

    FusedSiluAndMul: Type
    RMSNorm: Type
    RMSResNorm: Type
    AddBiasResLayerNorm: Type
    FusedQKRMSNorm: Type
    QKRMSNorm: Type
    SelectTopk: Type
    GroupTopK: Type
    FakeBalanceExpert: Type
    IndexerOp: Type
    SigmoidGateScaleAdd: Type
