"""XPU stubs for ops not yet needed (MoE, deep_gemm, etc.)."""

from rtp_llm.models_py.modules.base.not_implemented import NotImplementedOp


class GroupTopK(NotImplementedOp):
    """GroupTopK is not implemented for XPU."""

    def __init__(self, *args, **kwargs):
        super().__init__(op_name="GroupTopK", device_type="XPU")


class FakeBalanceExpert(NotImplementedOp):
    """FakeBalanceExpert is not implemented for XPU."""

    def __init__(self, *args, **kwargs):
        super().__init__(op_name="FakeBalanceExpert", device_type="XPU")


class IndexerOp(NotImplementedOp):
    """IndexerOp is not implemented for XPU."""

    def __init__(self, *args, **kwargs):
        super().__init__(op_name="IndexerOp", device_type="XPU")
