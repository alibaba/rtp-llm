from module_base import GptModelBase


class GptModelExample(GptModelBase):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden):
        return super().forward(hidden)

