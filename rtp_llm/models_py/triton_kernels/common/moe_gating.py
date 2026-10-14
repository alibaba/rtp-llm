"""BLOCK_H heuristics shared by the MoE gating Triton kernels.

The fused single-kernel gate-scale-add that used to live here was retired:
its in-register fp32 -> bf16 -> fp32 materialization rounds are folded away
by the Triton/LLVM pipeline, so it was not bit-identical to the per-op aten
chain.  modules/base/cuda/moe_gating.py now composes the bit-exact
sigmoid_mul_triton / topk_sum_triton kernels (common/moe_combine.py) with a
trailing aten add_.
"""

import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

_MIN_TOTAL_PROGRAMS = 512  # Target minimum number of Triton programs to launch
# (aims to keep enough SMs busy; A100 has 108 SMs)
_MIN_BLOCK_H = 128  # Minimum BLOCK_H: 128 bf16 = 256 bytes = 2 cache lines,
# ensuring coalesced memory access efficiency
_MAX_BLOCK_H = 4096  # Maximum BLOCK_H: prevents register spilling


# ---------------------------------------------------------------------------
# Retired: the fused single-kernel gate-scale-add.  See
# modules/base/cuda/moe_gating.py for the bit-exact decomposition that
# replaced it.

# ---------------------------------------------------------------------------
# BLOCK_H selector
# ---------------------------------------------------------------------------


def _select_block_h(T: int, H: int) -> int:
    """Choose BLOCK_H to balance SM utilisation and memory access efficiency.

    Strategy:
    - Target at least _MIN_TOTAL_PROGRAMS Triton programs (total = T * ceil(H/BLOCK_H)).
    - For small T (decode: T=1~32), this means using a small BLOCK_H so the
      H dimension is sliced into many blocks, keeping more SMs busy.
    - For large T (prefill), BLOCK_H naturally grows, reducing launch overhead.
    - Hard lower bound _MIN_BLOCK_H=128 ensures coalesced memory access:
        128 bf16 elements = 256 bytes = 2 L2 cache lines (128 bytes each).
    - Hard upper bound _MAX_BLOCK_H=4096 prevents register spilling.

    Example grid sizes:
      T=1,  H=4096 → BLOCK_H=128, grid=(1,32)  → 32  programs
      T=8,  H=4096 → BLOCK_H=128, grid=(8,32)  → 256 programs
      T=32, H=2048 → BLOCK_H=128, grid=(32,16) → 512 programs ✅
      T=32, H=4096 → BLOCK_H=256, grid=(32,16) → 512 programs ✅
      T=1024,H=7168→ BLOCK_H=4096,grid=(1024,2)→2048 programs ✅
    """
    target_h_blocks = max(1, _MIN_TOTAL_PROGRAMS // max(T, 1))
    ideal = triton.next_power_of_2(max(1, H // target_h_blocks))
    return max(_MIN_BLOCK_H, min(_MAX_BLOCK_H, ideal))
