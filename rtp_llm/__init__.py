import importlib
from typing import Any

from rtp_llm.utils.import_util import has_internal_source
from rtp_llm.utils.triton_compile_patch import maybe_enable_compile_monitor

# Opt-in and side-effect free unless RTP_LLM_TRITON_COMPILE_MONITOR is set, so it
# stays outside the lazy __getattr__ below: the monitor has to be installed before
# anything triggers a Triton compile, and importing this module is the only point
# guaranteed to run first. It patches nothing when the variable is unset.
maybe_enable_compile_monitor()

# The internal entrypoint registers models and extends the server argument parser,
# so it has to run before anything builds that parser -- start_server imports
# setup_args at module level, which leaves importing this package as the only
# guaranteed-earlier point. It stays outside the lazy __getattr__ below for the
# same reason, and is a no-op in open-source builds.
if has_internal_source():
    import internal_source.rtp_llm.models_py  # noqa: F401


def __getattr__(name: str) -> Any:
    """Preserve old top-level access without importing C++ ops eagerly."""
    if name == "_ft_pickler":
        module = importlib.import_module("rtp_llm._ft_pickler")
        globals()[name] = module
        return module

    if name == "enable_compile_monitor":
        from rtp_llm.utils.triton_compile_patch import enable_compile_monitor

        globals()[name] = enable_compile_monitor
        return enable_compile_monitor

    ops = importlib.import_module("rtp_llm.ops")
    try:
        value = getattr(ops, name)
    except AttributeError as e:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from e
    globals()[name] = value
    return value
