import argparse
import logging
import os
import sys
from typing import Mapping, MutableMapping, Optional, Sequence


_AUTO_ENABLED_FOR_WARMUP = False


def is_start_server_entrypoint(argv: Sequence[str]) -> bool:
    # Only inspect the executable/script prefix. Server argument values must not be able to turn a
    # normal package import into an allocator configuration side effect.
    for index, arg in enumerate(argv):
        if arg == "-m":
            return (
                index + 1 < len(argv)
                and argv[index + 1] == "rtp_llm.start_server"
            )
        basename = os.path.basename(arg)
        if basename.endswith(".py"):
            return basename == "start_server.py"
    return False


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


def warmup_requested(
    argv: Sequence[str], environ: Mapping[str, str], default: bool = True
) -> bool:
    """Resolve warm_up before importing torch, matching the server parser's precedence."""
    cli_value: Optional[str] = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--warm_up":
            if index + 1 >= len(argv):
                raise ValueError("--warm_up requires a value")
            cli_value = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--warm_up="):
            cli_value = arg.split("=", 1)[1]
        index += 1

    if cli_value is not None:
        return str2bool(cli_value)
    if "WARM_UP" in environ:
        return str2bool(environ["WARM_UP"])
    return default


def warmup_requested_or_default(
    argv: Sequence[str],
    environ: Mapping[str, str],
    default: bool = True,
) -> bool:
    """Best-effort pre-import parsing; the full server parser remains authoritative."""
    try:
        return warmup_requested(argv, environ, default)
    except (argparse.ArgumentTypeError, ValueError) as error:
        logging.getLogger(__name__).warning(
            "ignoring invalid pre-import warm_up value and using default=%s: %s",
            str(default).lower(),
            error,
        )
        return default


def configure_expandable_segments_for_warmup(
    argv: Optional[Sequence[str]] = None,
    environ: Optional[MutableMapping[str, str]] = None,
    is_rocm: Optional[bool] = None,
) -> bool:
    """Set the allocator default for CUDA warmup without overriding user configuration."""
    global _AUTO_ENABLED_FOR_WARMUP
    args = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if environ is None else environ
    if not warmup_requested_or_default(args, env):
        return False

    if is_rocm is None:
        is_rocm = os.path.exists("/opt/rocm") or bool(env.get("ROCM_PATH"))
    if is_rocm:
        return False

    before = env.get("PYTORCH_CUDA_ALLOC_CONF")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    auto_enabled = before is None
    if environ is None and auto_enabled:
        _AUTO_ENABLED_FOR_WARMUP = True
    return auto_enabled


def expandable_segments_auto_enabled_for_warmup() -> bool:
    return _AUTO_ENABLED_FOR_WARMUP
