"""ROCm-specific base modules"""

from rtp_llm.models_py.modules.base.not_implemented import NotImplementedOp


class FusedAddRMSNorm(NotImplementedOp):
    """FusedAddRMSNorm is not implemented for ROCm."""

    def __init__(self, *args, **kwargs):
        super().__init__(op_name="FusedAddRMSNorm", device_type="ROCm")


class GroupTopK(NotImplementedOp):
    """GroupTopK is not implemented for ROCm."""

    def __init__(self, *args, **kwargs):
        super().__init__(op_name="GroupTopK", device_type="ROCm")
