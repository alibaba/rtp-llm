"""MoE gating fusion kernels.

Provides fused elementwise operations for MoE shared-expert gating patterns,
avoiding intermediate tensor allocations and redundant memory round-trips.
"""

import torch
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
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def _SigmoidGateScaleAdd_kernel(
    gate_ptr,  # [T, 1]  — scalar gate output from shared_expert_gate linear
    shared_ptr,  # [T, H]  — shared expert MLP output
    output_ptr,  # [T, H]  — routed experts output (modified in-place)
    T,
    H,
    stride_gate_t,  # gate stride along token dim (elements)
    stride_shared_t,  # shared stride along token dim (elements)
    stride_out_t,  # output stride along token dim (elements)
    BLOCK_H: tl.constexpr,
):
    """Fused: output[t, :] = sigmoid(gate[t, 0]) * shared[t, :] + output[t, :]

    Grid: (T, ceil(H / BLOCK_H))
      - axis-0: token index
      - axis-1: hidden-dimension block index
    """
    tid = tl.program_id(axis=0)  # token index
    hid = tl.program_id(axis=1)  # hidden block index

    # ------------------------------------------------------------------
    # 1. Load scalar gate for this token; compute sigmoid in fp32
    # ------------------------------------------------------------------
    gate_val = tl.load(gate_ptr + tid * stride_gate_t).to(tl.float32)
    gate_val = tl.sigmoid(gate_val)  # 1 / (1 + exp(-x)), numerically stable

    # ------------------------------------------------------------------
    # 2. Compute the hidden-dim slice this program is responsible for
    # ------------------------------------------------------------------
    h_offsets = hid * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_offsets < H

    # ------------------------------------------------------------------
    # 3. Load shared-expert output and experts output
    # ------------------------------------------------------------------
    shared_base = shared_ptr + tid * stride_shared_t
    out_base = output_ptr + tid * stride_out_t

    shared_vec = tl.load(shared_base + h_offsets, mask=mask, other=0.0)
    experts_vec = tl.load(out_base + h_offsets, mask=mask, other=0.0)

    # ------------------------------------------------------------------
    # 4. Fused computation in fp32, then cast back to original dtype
    # ------------------------------------------------------------------
    result = gate_val * shared_vec.to(tl.float32) + experts_vec.to(tl.float32)
    tl.store(out_base + h_offsets, result.to(shared_vec.dtype), mask=mask)


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


# ---------------------------------------------------------------------------
# Low-level functional entry point (used by the nn.Module wrapper)
# ---------------------------------------------------------------------------


def sigmoid_gate_scale_add_triton(
    gate: torch.Tensor,
    shared: torch.Tensor,
    experts: torch.Tensor,
) -> torch.Tensor:
    """Launch the fused Triton kernel.

    Computes in-place on *experts*:
        experts[t, :] = sigmoid(gate[t, 0]) * shared[t, :] + experts[t, :]

    Args:
        gate:    [T, 1]  — scalar gate; dtype fp16 / bf16 / fp32.
        shared:  [T, H]  — shared expert MLP output.
        experts: [T, H]  — routed experts output (modified in-place).

    Returns:
        experts tensor (same object, modified in-place).
    """
    T, H = shared.shape
    BLOCK_H = _select_block_h(T, H)
    grid = (T, triton.cdiv(H, BLOCK_H))
    _SigmoidGateScaleAdd_kernel[grid](
        gate,
        shared,
        experts,
        T,
        H,
        gate.stride(0),
        shared.stride(0),
        experts.stride(0),
        BLOCK_H=BLOCK_H,
    )
    return experts
