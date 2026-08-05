import argparse

DEFAULT_MOE_SKEW_MULT = 2.0


def str2bool(value):
    """Shared bool parser for pre-import setup and the full server argument parser."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in ("yes", "true", "t", "1", "on"):
        return True
    if normalized in ("no", "false", "f", "0", "off"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")
