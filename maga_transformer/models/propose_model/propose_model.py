from typing import Any, Optional


class ProposeModel:
    def __init__(self, sp_type: str, model: Optional[Any] = None):
        self.model = model
        self.sp_type = sp_type