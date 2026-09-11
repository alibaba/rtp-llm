"""SGLang PPU paired FP8/scale permute; see dsv4_ppu/NOTICE."""

from pathlib import Path

import torch

# --- Fused permute(1,0,2) kernel: includes & template (Python JIT pattern) ---
permute_includes = (
    '"' + str(Path(__file__).with_name("dsv4_ppu") / "fused_permute.cuh") + '"',
)
permute_template = """
using namespace deep_gemm;

// Compile-time constants from Python JIT
constexpr int VEC_SIZE = {VEC_SIZE};
constexpr int THREADS_PER_BLOCK = {THREADS_PER_BLOCK};
constexpr int ROWS_PER_ITER = {ROWS_PER_ITER};
constexpr int DIM1 = {DIM1};
constexpr int NUM_D2_VECS_A = {NUM_D2_VECS_A};
constexpr int TAIL_BYTES_A = {TAIL_BYTES_A};
constexpr int DIM2_SFA_BYTES = {DIM2_SFA_BYTES};
constexpr int D1_TILE = {D1_TILE};
constexpr bool USE_FAST_PATH = {USE_FAST_PATH};

// Grid: fast path uses (DIM1 * D2_BLOCKS, grid_y), SMEM path uses (D1_GROUPS, grid_y)
constexpr int D2_BLOCKS = (NUM_D2_VECS_A + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
constexpr int D1_GROUPS = DIM1 / D1_TILE;
int grid_x = USE_FAST_PATH ? (DIM1 * D2_BLOCKS) : D1_GROUPS;
int grid_y = (dim0 + ROWS_PER_ITER - 1) / ROWS_PER_ITER;
dim3 grid(grid_x, grid_y, 1);
dim3 block(THREADS_PER_BLOCK, 1, 1);
// SMEM: 0 for fast path, full tile for SMEM transpose path
constexpr int SMEM_SIZE = USE_FAST_PATH ? 0 : (ROWS_PER_ITER * D1_TILE * NUM_D2_VECS_A * VEC_SIZE);

fused_permute_kernel<VEC_SIZE, THREADS_PER_BLOCK, ROWS_PER_ITER, DIM1, NUM_D2_VECS_A, TAIL_BYTES_A, DIM2_SFA_BYTES, D1_TILE, USE_FAST_PATH><<<grid, block, SMEM_SIZE, stream>>>(
    (const char*)src_a, (char*)dst_a,
    (const char*)src_sfa, (char*)dst_sfa,
    dim0, dim2_a);
__return_code = 0;
"""


def fused_permute(a: torch.Tensor, sfa: torch.Tensor):
    from deep_gemm.jit_kernels.tuner import jit_tuner

    if (
        a.device != sfa.device
        or a.dtype != torch.float8_e4m3fn
        or sfa.dtype != torch.float32
    ):
        raise ValueError(
            "DSV4 PPU fused permute requires FP8 payload and FP32 scales on one device"
        )
    if torch.cuda.get_device_name(a.device) != "ZW-M890P":
        raise RuntimeError("DSV4 PPU fused permute requires ZW-M890P")
    """
    Fused JIT kernel: permute(1,0,2).contiguous() for two 3-D tensors in one launch.

    Both tensors must share the same (dim0, dim1) but may have different dim2 and dtypes.
    Returns (out_a, out_sfa) with shape (dim1, dim0, dim2_x) for each.
    """
    assert (
        a.dim() == 3 and a.is_contiguous()
    ), "fused_permute: tensor a must be a contiguous 3-D tensor"
    assert (
        sfa.dim() == 3 and sfa.is_contiguous()
    ), "fused_permute: tensor sfa must be a contiguous 3-D tensor"

    dim0, dim1, dim2_a = a.shape
    dim0_s, dim1_s, dim2_sfa = sfa.shape
    assert (
        dim0 == dim0_s and dim1 == dim1_s
    ), "fused_permute: dim0 and dim1 must match between a and sfa"

    out_a = torch.empty((dim1, dim0, dim2_a), dtype=a.dtype, device=a.device)
    out_sfa = torch.empty((dim1, dim0, dim2_sfa), dtype=sfa.dtype, device=sfa.device)

    if dim0 == 0 or dim1 == 0:
        return out_a, out_sfa

    # Convert to byte-level dimensions
    elem_size_a = a.element_size()
    dim2_a_bytes = dim2_a * elem_size_a
    elem_size_sfa = sfa.element_size()
    dim2_sfa_bytes = dim2_sfa * elem_size_sfa

    num_d2_vecs_a = dim2_a_bytes // 16
    tail_bytes_a = dim2_a_bytes % 16
    max_smem = 65536  # 64KB target for good occupancy

    # Determine fast path early — it affects parameter selection.
    # Fast path: no SMEM, direct register load/store. Grid = (DIM1*D2_BLOCKS, grid_y).
    # For tiny data, we want MAXIMUM blocks to saturate 39 SMs.
    total_a_bytes = dim0 * dim1 * dim2_a_bytes
    use_fast_path = total_a_bytes <= 524288  # 512KB

    if use_fast_path:
        # Fast path: no SMEM constraint, maximize parallelism.
        # Dynamically choose THREADS_PER_BLOCK to ensure enough grid blocks for GPU saturation.
        # With k=2048, NUM_D2_VECS_A=128 → D2_BLOCKS=2 with THREADS_PER_BLOCK=64 (only 2 warps).
        # THREADS_PER_BLOCK=32 (1 warp) doubles D2_BLOCKS to 4, recovering block count.
        rows_per_iter = 1
        d1_tile = 1  # Not used for fast path, but needed for cache key
        d2_blocks_64 = (num_d2_vecs_a + 63) // 64
        grid_x_64 = dim1 * d2_blocks_64
        grid_y_fast = dim0  # rows_per_iter=1
        total_blocks_64 = grid_x_64 * grid_y_fast
        if total_blocks_64 <= 256 and grid_y_fast >= 8:
            threads_per_block = 32  # 1 warp: doubles D2_BLOCKS for more parallelism
        else:
            threads_per_block = 64  # 2 warps: sufficient parallelism
    else:
        # SMEM path: balance parallelism with SMEM constraints.
        # Choose THREADS_PER_BLOCK: larger blocks have lower dispatch overhead per thread
        if dim0 <= 64:
            threads_per_block = 128
        elif dim0 <= 256:
            threads_per_block = 256
        else:
            threads_per_block = 512

        # Choose rows_per_iter based on dim0 (m) size:
        # Large m → more blocks is fine (good parallelism), minimize per-block work
        # Small m → need high rows_per_iter to reduce block count (launch overhead)
        if dim0 >= 256:
            rows_per_iter = 4
        elif dim0 >= 64:
            rows_per_iter = 2
        else:
            max_rows_for_smem = max_smem // (dim1 * num_d2_vecs_a * 16)
            rows_per_iter = min(dim0, max(1, max_rows_for_smem))
            rows_per_iter = 1 << (rows_per_iter.bit_length() - 1)
            rows_per_iter = min(rows_per_iter, 16)

        # Compute D1_TILE: balance parallelism vs per-block overhead.
        # SMEM path uses grid_x = D1_GROUPS = DIM1 / D1_TILE.
        grid_y = (dim0 + rows_per_iter - 1) // rows_per_iter
        target_blocks = 128
        min_d1_groups = max(1, (target_blocks + grid_y - 1) // grid_y)
        max_d1_tile_by_parallelism = max(1, dim1 // min_d1_groups)
        max_d1_tile_by_smem = max(1, max_smem // (rows_per_iter * num_d2_vecs_a * 16))
        d1_tile = min(dim1, max_d1_tile_by_smem, max_d1_tile_by_parallelism)
        d1_tile = max(1, d1_tile)
        while dim1 % d1_tile != 0:
            d1_tile -= 1

    global permute_includes, permute_template
    args = (a, out_a, sfa, out_sfa, dim0, dim2_a_bytes, torch.cuda.current_stream())
    ElementA = "cutlass::float_e4m3_t" if a.dtype == torch.float8_e4m3fn else "int8_t"
    runtime = jit_tuner.compile_and_tune(
        name="rtp_sglang_fused_permute_v1_" + ElementA,
        keys={
            "VEC_SIZE": 16,
            "THREADS_PER_BLOCK": threads_per_block,
            "ROWS_PER_ITER": rows_per_iter,
            "DIM1": dim1,
            "NUM_D2_VECS_A": num_d2_vecs_a,
            "TAIL_BYTES_A": tail_bytes_a,
            "DIM2_SFA_BYTES": dim2_sfa_bytes,
            "D1_TILE": d1_tile,
            "USE_FAST_PATH": "true" if use_fast_path else "false",
        },
        space=(),
        includes=permute_includes,
        arg_defs=(
            ("src_a", a.dtype),
            ("dst_a", a.dtype),
            ("src_sfa", sfa.dtype),
            ("dst_sfa", sfa.dtype),
            ("dim0", int),
            ("dim2_a", int),
            ("stream", torch.cuda.Stream),
        ),
        template=permute_template,
        args=args,
        jit_include_dir="cutlass3",
    )
    runtime(*args)

    return out_a, out_sfa
