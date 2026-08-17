import importlib
import logging
from typing import Any

from rtp_llm.utils.import_util import (
    has_internal_source,
    import_optional_internal_source_entrypoint,
)
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
#
# Go through the shared entrypoint helper rather than a bare import guarded by a
# directory check: it tolerates a missing submodule the same way the three register
# modules do, so a present-but-unimportable internal_source cannot make
# `import rtp_llm` itself fail. The import still happens here, at module level, because
# the registration it performs must precede argparse construction.
#
# Warn when the entrypoint is skipped despite internal_source being present: that is the
# state where models go unregistered and the failure resurfaces much later as an
# `--moe_strategy invalid choice` argparse error, which is expensive to trace back here.
if not import_optional_internal_source_entrypoint("models_py") and has_internal_source():
    logging.getLogger(__name__).warning(
        "internal_source is present but internal_source.rtp_llm.models_py was not imported; "
        "internal models and their server argument extensions are NOT registered"
    )


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
