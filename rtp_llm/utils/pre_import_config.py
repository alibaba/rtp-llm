import os
import sys
from typing import Mapping, MutableMapping, Optional, Sequence


_TRUE_VALUES = {"yes", "true", "t", "1", "on"}
_FALSE_VALUES = {"no", "false", "f", "0", "off"}


def is_start_server_entrypoint(argv: Sequence[str]) -> bool:
    for index, arg in enumerate(argv[:-1]):
        if arg == "-m" and argv[index + 1] == "rtp_llm.start_server":
            return True
    return any(os.path.basename(arg) == "start_server.py" for arg in argv)


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"invalid boolean value for warm_up: {value}")


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
        return _parse_bool(cli_value)
    if "WARM_UP" in environ:
        return _parse_bool(environ["WARM_UP"])
    return default


def configure_expandable_segments_for_warmup(
    argv: Optional[Sequence[str]] = None,
    environ: Optional[MutableMapping[str, str]] = None,
    is_rocm: Optional[bool] = None,
) -> bool:
    """Set the allocator default for CUDA warmup without overriding user configuration."""
    args = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if environ is None else environ
    if not warmup_requested(args, env):
        return False

    if is_rocm is None:
        is_rocm = os.path.exists("/opt/rocm") or bool(env.get("ROCM_PATH"))
    if is_rocm:
        return False

    before = env.get("PYTORCH_CUDA_ALLOC_CONF")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return before is None
