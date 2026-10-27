"""Qualified M890P decode tiles for the DSV4 attention projections."""

from functools import lru_cache

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

# Separate opt-in for QKV, shared-expert and MTP projections. These tiles keep
# the existing quantizer, K traversal and BF16 output boundary unchanged.
_PROJECTION_TILES = {
    (1536, 4096): {
        8: (16, 64, 16, 16, 4),
        24: (16, 64, 16, 16, 4),
        32: (16, 64, 16, 16, 4),
    },
    (4096, 4096): {
        8: (16, 64, 16, 16, 4),
        24: (16, 64, 16, 16, 4),
        32: (16, 64, 16, 16, 4),
    },
    (4096, 2048): {
        8: (16, 128, 16, 32, 4),
        24: (32, 64, 16, 32, 4),
        32: (32, 64, 16, 32, 4),
    },
    (8192, 1024): {
        8: (16, 128, 16, 32, 4),
        24: (32, 64, 16, 32, 4),
        32: (32, 64, 16, 32, 4),
    },
}


@lru_cache(maxsize=None)
def _config(tile, k, sms):
    from deep_gemm.jit_kernels.gemm_fp8 import get_smem_config

    bm, bn, wm, wn, stages = tile
    return sms, bm, bn, 128, wm, wn, stages, get_smem_config(stages, k, bm, bn)


def decode_attention_gemm_config(m, n, k, *, groups=1):
    """Return None for unqualified geometry or a different SM allocation."""
    tile = _TILES.get((groups, n, k), {}).get(m)
    if tile is None:
        return None
    from deep_gemm.jit_kernels.utils import get_num_sms

    sms = get_num_sms()
    if sms != 39:
        return None
    return _config(tile, k, sms)


def decode_projection_gemm_config(m, n, k):
    """Opt-in dense projection tiles, with library fallback outside qualification."""
    tile = _PROJECTION_TILES.get((n, k), {}).get(m)
    if tile is None:
        return None
    from deep_gemm.jit_kernels.utils import get_num_sms

    sms = get_num_sms()
    return _config(tile, k, sms) if sms == 39 else None
