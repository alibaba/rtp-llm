from enum import Flag, auto


class OutputModality(Flag):
    TEXT = auto()
    AUDIO = auto()
    IMAGE = auto()
    VIDEO = auto()
