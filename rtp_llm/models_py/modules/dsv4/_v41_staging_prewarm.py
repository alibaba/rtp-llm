"""R30/B: DSV41_STAGING_PREWARM (default "0") — caching-allocator pre-warm +
pinned-staging pre-creation.

KDA-IDLE-20261030 item 5: the first-chunk allocator expansions and the
pinned-buffer creations are paid inside the measured window's early
chunks. The 4 warmup requests already cover most of it, so the honest
expectation is small; the flag makes it measurable. Under the flag, at the
first native PP prefill of each worker: (1) pre-create the per-device
pinned staging buffers used by the adopted counts-overlap path, (2) expand
the caching allocator with a reserve of the steady-state per-chunk block
sizes (allocate + free), (3) one device synchronize. Byte-exact by
construction (host/allocator only; no kernel content changes).
"""

import os
import sys

import torch

STAGING_PREWARM_FLAG = "DSV41_STAGING_PREWARM"
_DONE = set()


def staging_prewarm_enabled() -> bool:
    # R30: adopted as default (pairs -9.5 median, 2/3, byte-exact).
    # Escape: DSV41_STAGING_PREWARM=0.
    return os.environ.get(STAGING_PREWARM_FLAG, "1") == "1"


def prewarm_device(device, expert_count=256):
    """Idempotent per-device pre-warm: pinned staging buffers + allocator
    pool expansion. Called once per worker (first native PP prefill)."""
    key = (device.type, device.index)
    if key in _DONE:
        return
    _DONE.add(key)
    if not staging_prewarm_enabled() or device.type != "cuda":
        return
    # (1) the pinned staging buffers used by the adopted counts path
    from rtp_llm.models_py.modules.dsv4.v41_moe import (
        _COUNTS_READ_CACHE,
        _H2D_STAGING_CACHE,
    )

    for experts in (expert_count,):
        _COUNTS_READ_CACHE.setdefault(
            (device.type, device.index, experts),
            (
                torch.empty((experts,), dtype=torch.int64, pin_memory=True),
                torch.cuda.Event(),
            ),
        )
    _H2D_STAGING_CACHE.setdefault(
        (device.type, device.index, expert_count),
        torch.empty((expert_count,), dtype=torch.int64, pin_memory=True),
    )
    # (2) allocator pool expansion: the steady-state per-chunk blocks seen
    # in the migration traces (the Q workspace, the routed payloads, the
    # residual streams). Allocate the set once and free: the pools stay
    # expanded for the caching allocator's reuse.
    with torch.cuda.device(device):
        blocks = [
            torch.empty((4096, 4, 5120), dtype=torch.bfloat16, device=device),
            torch.empty((4096, 5120), dtype=torch.bfloat16, device=device),
            torch.empty((16384, 5120), dtype=torch.uint8, device=device),
            torch.empty((16384, 5120 // 32), dtype=torch.uint8, device=device),
            torch.empty((4096, 4, 5120), dtype=torch.float32, device=device),
        ]
        del blocks
        torch.cuda.synchronize(device)
    print(
        f"[STAGINGPREWARM] engaged ({STAGING_PREWARM_FLAG}=1) dev={device.index}",
        file=sys.stderr,
        flush=True,
    )
