class LoraException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class LoraCountException(LoraException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
