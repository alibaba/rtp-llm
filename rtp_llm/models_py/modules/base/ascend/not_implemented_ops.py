from rtp_llm.models_py.modules.base.not_implemented import NotImplementedOp


class GroupTopK(NotImplementedOp):
    def __init__(self, *args, **kwargs):
        super().__init__(op_name="GroupTopK", device_type="Ascend")


class FakeBalanceExpert(NotImplementedOp):
    def __init__(self, *args, **kwargs):
        super().__init__(op_name="FakeBalanceExpert", device_type="Ascend")


class IndexerOp(NotImplementedOp):
    def __init__(self, *args, **kwargs):
        super().__init__(op_name="IndexerOp", device_type="Ascend")
