"""Dcu-specific base modules"""

from rtp_llm.models_py.modules.base.not_implemented import NotImplementedOp


class RMSResNorm(NotImplementedOp):
    """RMSResNorm is not implemented for Dcu."""

    def __init__(self, *args, **kwargs):
        super().__init__(op_name="RMSResNorm", device_type="Dcu")

class IndexerOp(NotImplementedOp):
    """IndexerOp is not implemented for Dcu."""

    def __init__(self, *args, **kwargs):
        super().__init__(op_name="IndexerOp", device_type="Dcu")

class GroupTopK(NotImplementedOp):
    """GroupTopK is not implemented for Dcu."""

    def __init__(self, *args, **kwargs):
        super().__init__(op_name="GroupTopK", device_type="Dcu")

class AddBiasResLayerNorm(NotImplementedOp):
    """AddBiasResLayerNorm is not implemented for Dcu."""

    def __init__(self, *args, **kwargs):
        super().__init__(op_name="AddBiasResLayerNorm", device_type="Dcu")

class SigmoidGateScaleAdd(NotImplementedOp):
    """SigmoidGateScaleAdd is not implemented for Dcu."""

    def __init__(self, *args, **kwargs):
        super().__init__(op_name="SigmoidGateScaleAdd", device_type="Dcu")

class SelectTopk(NotImplementedOp):
    """SelectTopk is not implemented for Dcu."""

    def __init__(self, *args, **kwargs):
        super().__init__(op_name="SelectTopk", device_type="Dcu")

class FakeBalanceExpert(NotImplementedOp):
    """FakeBalanceExpert is not implemented for Dcu."""

    def __init__(self, *args, **kwargs):
        super().__init__(op_name="FakeBalanceExpert", device_type="Dcu")

