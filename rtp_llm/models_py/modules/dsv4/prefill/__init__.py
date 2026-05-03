"""DeepSeek-V4 prefill-only runtime helpers.

Symmetric to :mod:`rtp_llm.models_py.modules.dsv4.decode`. Hosts the
pieces used only by the prefill forward path (per-request KV pool
registration, Context-Parallel metadata binding, qwen3-style per-layer
loop, PD-disagg cache_store registration).

Nothing here is imported during decode; nothing in ``decode/`` is
imported from here — the two packages sit on a clean cleavage line so
PD-disagg can later split prefill/decode services without tangled
internal refs.
"""

from rtp_llm.models_py.modules.dsv4.prefill.forward import (
    DSv4WriteCacheStoreOp,
    create_dsv4_write_cache_store_impl,
    forward_layers,
    forward_prefill,
    set_cp_info,
)

__all__ = [
    "DSv4WriteCacheStoreOp",
    "create_dsv4_write_cache_store_impl",
    "forward_layers",
    "forward_prefill",
    "set_cp_info",
]
