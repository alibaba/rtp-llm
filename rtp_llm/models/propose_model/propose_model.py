from typing import Any, Optional


class ProposeModel:
    def __init__(
        self,
        sp_type: str,
        gen_num_per_circle: int,
        model: Optional[Any] = None,
        *,
        allocation_only_config: Optional[Any] = None,
    ):
        self.model = model
        self.allocation_only_config = allocation_only_config
        self.sp_type = sp_type
        self.gen_num_per_circle = gen_num_per_circle
