"""Qualified M890P decode tiles for the DSV4 attention projections."""

from functools import lru_cache
import os

# (groups, N, K) -> M -> (block M, block N, warp M, warp N, stages).
# Keep unmeasured shapes on the library's normal dispatch path.
_TILES = {
    (1, 32768, 1024): {
        8: (16, 128, 16, 32, 3),
        16: (32, 64, 16, 32, 4),
        24: (32, 64, 16, 32, 4),
        32: (32, 64, 16, 32, 4),
    },
    (8, 1024, 4096): {
        8: (16, 128, 16, 32, 4),
        16: (16, 128, 16, 32, 4),
        24: (32, 64, 16, 32, 4),
        32: (32, 64, 16, 32, 4),
    },
    (1, 4096, 8192): {
        8: (16, 64, 16, 16, 4),
        16: (16, 64, 16, 16, 4),
        24: (16, 64, 16, 16, 4),
        32: (16, 64, 16, 16, 4),
    },
}

_LARGE_EP_TILES = {
    (1, 32768, 1024): {m: (16, 128, 16, 32, 3) for m in (2, 4, 6, 12)},
    (1, 4096, 8192): {m: (16, 64, 16, 16, 4) for m in (2, 4, 6, 12)},
}


@lru_cache(maxsize=None)
def _config(tile, k, sms):
    from deep_gemm.jit_kernels.gemm_fp8 import get_smem_config

    bm, bn, wm, wn, stages = tile
    return sms, bm, bn, 128, wm, wn, stages, get_smem_config(stages, k, bm, bn)


def decode_attention_gemm_config(m, n, k, *, groups=1):
    """Return None for unqualified geometry or a different SM allocation."""
    tile = _TILES.get((groups, n, k), {}).get(m)
    if tile is None and os.environ.get("EP_SIZE") in ("16", "32"):
        tile = _LARGE_EP_TILES.get((groups, n, k), {}).get(m)
    if (
        tile is None
        and os.environ.get("EP_SIZE") in ("16", "32")
        and os.environ.get("TP_SIZE") == "1"
        and os.environ.get("DSV4_PPU_WOA_SMALL_TILE", "0") == "1"
        and (groups, n, k) == (8, 1024, 4096)
        and (m in (2, 6, 12) or (m == 4 and os.environ.get("EP_SIZE") == "16"))
    ):
        tile = (16, 64, 16, 16, 4)
    if tile is None:
        return None
    from deep_gemm.jit_kernels.utils import get_num_sms

    sms = get_num_sms()
    if sms != 39:
        return None
    return _config(tile, k, sms)
