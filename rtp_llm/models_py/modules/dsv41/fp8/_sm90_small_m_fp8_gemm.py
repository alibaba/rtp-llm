"""SM90 Hopper Zero-TMA-OOB Warp-Specialized TMA+WGMMA Split-K G32 FP8 Dense GEMM.

Accelerates DeepSeek-V4.1 native group-32 FP8 dense projections on SM90 (e.g.
L20Z 132 SMs / H100) across decode & speculative verification batch sizes
``1 <= M <= 128`` (BS=1..16 with verify/draft widths 4/5 -> M=4, 5, 16, 20,
32, 40, 64, 80):

1. Zero-TMA-OOB descriptor sizing: ``boxDim_A = {128, M}`` and
   ``boxDim_SFA = {align4(M), 4}`` eliminate Hopper's TMA out-of-bounds
   serialization penalty at non-power-of-two ``M`` (4, 5, 20, 40, 80).
2. Deterministic single-kernel Split-K (``atom.global.acq_rel.gpu.add.s32`` +
   ascending-order reduction + counter reset) saturates all 132 SMs on narrow-N
   shapes like ``qkv_a (1792x5120)`` with zero extra kernel launch and zero
   ``cudaMemset`` (100% CUDA Graph capture & replay safe).
3. Three Hopper WGMMA modes selected statically by ``(M, N, K)``:
   - ``Swap64`` (``M <= 32``): transposed WGMMA ``m64n{8,16,24,32}k32``,
     4-bank accumulators, ``float2`` LDS.64 SFA loads, preloaded SMEM SFB.
   - ``NS32`` (``M <= 64``, ``N <= 8192``): non-swapped WGMMA ``m64n32k32``
     with ``BLOCK_N=32`` (doubles N-tiles to 160 on ``o_proj 5120x8192``,
     144 on ``shared_w13``, 128 on ``index_wq``, 1 SFB row, 2 FMUL/sub_k).
   - ``NS64`` (``33 <= M <= 64`` wide-N, or Dual-WG ``BLOCK_M=128`` for
     ``65 <= M <= 128`` reading weight B only once at ``M=80``).

Set ``DSV41_SM90_SMALL_M_FP8_GEMM=0`` to disable and retain stock DeepGEMM.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_MAX_DYNAMIC_SMEM_BYTES = 224 * 1024
_DEFAULT_WS_FLOATS = 4 * 1024 * 1024  # 16 MB reusable FP32 Split-K workspace
_DEFAULT_COUNTER_ELEMS = 4096         # 16 KB (supports N up to 131,072)

_MAX_CACHED_STREAMS = 32
_LIB_LOCK = threading.Lock()
_CACHED_LIB: Optional[ctypes.CDLL] = None
_LIB_LOAD_FAILED = False
_INITIALIZED_DEVICES: set[int] = set()
_SM90_CAPABLE_DEVICES: dict[int, bool] = {}
_STREAM_BUFFERS: dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
_CAPTURE_BUFFERS: dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
_DEVICE_BUFFERS_LOCK = threading.Lock()


def is_enabled() -> bool:
    return os.environ.get("DSV41_SM90_SMALL_M_FP8_GEMM", "1").strip() != "0"


def _gen_wgmma_fn(n_val: int) -> str:
    n_acc = n_val // 2
    regs_out = ", ".join(f'"+f"(d[{i}])' for i in range(n_acc))
    regs_ptx = ", ".join(f"%{i}" for i in range(n_acc))
    da = n_acc
    db = n_acc + 1
    return f"""
template <>
__device__ __forceinline__ void wgmma_m64nNk32_e4m3_zero<{n_val}>(uint64_t desc_a, uint64_t desc_b, float* d) {{
    asm volatile(
        "{{\\n"
        ".reg .pred p;\\n"
        "setp.ne.b32 p, 0, 0;\\n"
        "wgmma.mma_async.sync.aligned.m64n{n_val}k32.f32.e4m3.e4m3 "
        "{{{regs_ptx}}}, %{da}, %{db}, p, 1, 1;\\n"
        "}}\\n"
        : {regs_out}
        : "l"(desc_a), "l"(desc_b)
    );
}}
"""


_CUDA_SOURCE = (
    r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

__host__ __device__ constexpr int align_up(int v, int a) { return (v + a - 1) / a * a; }
__device__ __forceinline__ uint32_t cvta_smem(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
__device__ __forceinline__ uint64_t make_wgmma_desc(const void* smem_ptr) {
    uint32_t addr = cvta_smem(smem_ptr) >> 4;
    return static_cast<uint64_t>(addr & 0x3FFFu)
         | (static_cast<uint64_t>(64u) << 32)
         | (static_cast<uint64_t>(1u) << 62);
}
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(cvta_smem(bar)), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(cvta_smem(bar)), "r"(tx_bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(cvta_smem(bar)) : "memory");
}
__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar, uint32_t parity) {
    uint32_t s = cvta_smem(bar);
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@!P1 bra.uni LAB_WAIT;\n"
        "}\n"
        :: "r"(s), "r"(parity) : "memory"
    );
}
__device__ __forceinline__ void tma_load_2d(
    const CUtensorMap* desc, uint64_t* bar, void* smem_ptr, int32_t c0, int32_t c1
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(cvta_smem(smem_ptr)), "l"(desc), "r"(c0), "r"(c1), "r"(cvta_smem(bar)) : "memory"
    );
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}
__device__ __forceinline__ void wgmma_fence_operand(float& reg) { asm volatile("" : "+f"(reg) :: "memory"); }
template <int N> __device__ __forceinline__ void setmaxnreg_dec() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N));
}
template <int N> __device__ __forceinline__ void setmaxnreg_inc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N));
}
template <int BAR_ID, int COUNT> __device__ __forceinline__ void named_bar_sync() {
    asm volatile("bar.sync %0, %1;\n" :: "n"(BAR_ID), "n"(COUNT) : "memory");
}

template <int WGMMA_N>
__device__ __forceinline__ void wgmma_m64nNk32_e4m3_zero(uint64_t desc_a, uint64_t desc_b, float* d);
"""
    + "\n".join(_gen_wgmma_fn(n) for n in (8, 16, 24, 32, 64))
    + r"""

// Non-swapped Zero-TMA-OOB TMA+WGMMA (BLOCK_N in {32, 64}, NUM_MATH_WGS in {1, 2})
template <int BLOCK_N, int NUM_MATH_WGS, int STAGES, int SPLIT_K, int MIN_BLOCKS>
__global__ void __launch_bounds__(NUM_MATH_WGS * 128 + 128, MIN_BLOCKS)
sm90_fp8_g32_nonswap_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_sfa,
    const float* __restrict__ SFB,
    __nv_bfloat16* __restrict__ D,
    float* __restrict__ workspace,
    int* __restrict__ counters,
    int M, int N, int K,
    int tx_bytes, int sfa_box_m
) {
    constexpr int BLOCK_M = NUM_MATH_WGS * 64;
    constexpr int BLOCK_K = 128;
    constexpr int NUM_MATH_THREADS = NUM_MATH_WGS * 128;
    constexpr int kNumAccum = BLOCK_N / 2;
    constexpr int SFB_ROWS = BLOCK_N / 32;
    constexpr int SMEM_B_STAGE = BLOCK_N * BLOCK_K;
    constexpr int SMEM_A_STAGE = BLOCK_M * BLOCK_K;
    constexpr int SMEM_SFA_STAGE = align_up(4 * BLOCK_M * static_cast<int>(sizeof(float)), 128);
    constexpr int TMA_REGS = (MIN_BLOCKS == 2) ? 24 : 40;
    constexpr int MATH_REGS = (NUM_MATH_WGS == 2) ? 232 : ((MIN_BLOCKS == 2) ? 208 : 248);

    const int n_tile = blockIdx.x;
    const int k_split = blockIdx.y;
    const int n_base = n_tile * BLOCK_N;

    const int total_k_blocks = K >> 7;
    const int k_per = total_k_blocks / SPLIT_K;
    const int k_rem = total_k_blocks % SPLIT_K;
    const int k_block_start = k_split * k_per + (k_split < k_rem ? k_split : k_rem);
    const int num_k_iters = k_per + (k_split < k_rem ? 1 : 0);
    if (num_k_iters <= 0) return;

    const int k_scales = K >> 5;
    const int split_k_scales = num_k_iters * 4;

    const int tid = threadIdx.x;
    const int warp_id = __shfl_sync(0xffffffff, tid >> 5, 0);
    const int lane_id = tid & 31;

    extern __shared__ __align__(1024) uint8_t smem_buf[];
    uint8_t* smem_b = smem_buf;
    uint8_t* smem_a = smem_b + STAGES * SMEM_B_STAGE;
    uint8_t* smem_sfa = smem_a + STAGES * SMEM_A_STAGE;
    uint64_t* full_bars = reinterpret_cast<uint64_t*>(smem_sfa + STAGES * SMEM_SFA_STAGE);
    uint64_t* empty_bars = full_bars + STAGES;
    float* smem_sfb = reinterpret_cast<float*>(empty_bars + STAGES);

    constexpr int TMA_WARP_BASE = NUM_MATH_WGS * 4;
    if (warp_id == TMA_WARP_BASE && lane_id == 0) {
        asm volatile("prefetch.tensormap [%0];\n" :: "l"(&tma_a));
        asm volatile("prefetch.tensormap [%0];\n" :: "l"(&tma_b));
        asm volatile("prefetch.tensormap [%0];\n" :: "l"(&tma_sfa));
    }
    if (warp_id == TMA_WARP_BASE + 1 && lane_id == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbarrier_init(&full_bars[s], 1);
            mbarrier_init(&empty_bars[s], NUM_MATH_WGS * 4);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (warp_id >= TMA_WARP_BASE) {
        setmaxnreg_dec<TMA_REGS>();
        if (warp_id == TMA_WARP_BASE && lane_id == 0) {
            int stage = 0;
            uint32_t phase = 0;
            for (int iter = 0; iter < num_k_iters; ++iter) {
                mbarrier_wait_parity(&empty_bars[stage], phase ^ 1u);
                const int kb = k_block_start + iter;
                const int k_coord = kb * BLOCK_K;
                tma_load_2d(&tma_b, &full_bars[stage], smem_b + stage * SMEM_B_STAGE, k_coord, n_base);
                tma_load_2d(&tma_a, &full_bars[stage], smem_a + stage * SMEM_A_STAGE, k_coord, 0);
                tma_load_2d(&tma_sfa, &full_bars[stage], smem_sfa + stage * SMEM_SFA_STAGE, 0, kb * 4);
                mbarrier_arrive_expect_tx(&full_bars[stage], tx_bytes);
                stage++;
                if (stage == STAGES) { stage = 0; phase ^= 1u; }
            }
        }
        return;
    }

    setmaxnreg_inc<MATH_REGS>();

    {
        const int sfb_row0 = n_base >> 5;
        const int sf_k_start = k_block_start * 4;
        const int total_sfb = SFB_ROWS * split_k_scales;
        for (int idx = tid; idx < total_sfb; idx += NUM_MATH_THREADS) {
            int r = (SFB_ROWS == 2 && idx >= split_k_scales) ? 1 : 0;
            int c = idx - r * split_k_scales;
            smem_sfb[idx] = __ldg(SFB + (size_t)(sfb_row0 + r) * k_scales + sf_k_start + c);
        }
        named_bar_sync<1, NUM_MATH_THREADS>();
    }

    const int math_wg = __shfl_sync(0xffffffff, tid >> 7, 0);
    const int local_warp = warp_id & 3;
    const int r_0 = math_wg * 64 + local_warp * 16 + (lane_id >> 2);
    const int r_1 = r_0 + 8;
    const int col_idx = lane_id & 3;
    const int n_0_base = col_idx * 2;
    const bool r0_valid = (r_0 < M);
    const bool r1_valid = (r_1 < M);

    const uint64_t desc_a_base = make_wgmma_desc(smem_a + math_wg * 64 * BLOCK_K);
    const uint64_t desc_b_base = make_wgmma_desc(smem_b);

    float final_accum[kNumAccum];
    #pragma unroll
    for (int i = 0; i < kNumAccum; ++i) final_accum[i] = 0.0f;

    constexpr bool kFourBanks = (NUM_MATH_WGS == 1 || BLOCK_N == 32);
    int stage = 0;
    uint32_t phase = 0;

    if constexpr (kFourBanks) {
        float accum[4][kNumAccum];
        #pragma unroll 4
        for (int iter = 0; iter < num_k_iters; ++iter) {
            mbarrier_wait_parity(&full_bars[stage], phase);

            const uint64_t da_s = desc_a_base + static_cast<uint64_t>((stage * SMEM_A_STAGE) >> 4);
            const uint64_t db_s = desc_b_base + static_cast<uint64_t>((stage * SMEM_B_STAGE) >> 4);
            const float* sfa_s = reinterpret_cast<const float*>(smem_sfa + stage * SMEM_SFA_STAGE);

            #pragma unroll
            for (int sk = 0; sk < 4; ++sk)
                #pragma unroll
                for (int i = 0; i < kNumAccum; ++i)
                    wgmma_fence_operand(accum[sk][i]);
            wgmma_fence();
            #pragma unroll
            for (int sk = 0; sk < 4; ++sk) {
                wgmma_m64nNk32_e4m3_zero<BLOCK_N>(da_s + sk * 2, db_s + sk * 2, accum[sk]);
            }
            wgmma_commit();
            #pragma unroll
            for (int sk = 0; sk < 4; ++sk)
                #pragma unroll
                for (int i = 0; i < kNumAccum; ++i)
                    wgmma_fence_operand(accum[sk][i]);

            float4 sb0_4 = *reinterpret_cast<const float4*>(smem_sfb + iter * 4);
            float sb0_arr[4] = {sb0_4.x, sb0_4.y, sb0_4.z, sb0_4.w};
            float sb1_arr[4];
            if constexpr (SFB_ROWS == 2) {
                float4 sb1_4 = *reinterpret_cast<const float4*>(smem_sfb + split_k_scales + iter * 4);
                sb1_arr[0] = sb1_4.x; sb1_arr[1] = sb1_4.y; sb1_arr[2] = sb1_4.z; sb1_arr[3] = sb1_4.w;
            }
            float sa0_arr[4], sa1_arr[4];
            #pragma unroll
            for (int sk = 0; sk < 4; ++sk) {
                sa0_arr[sk] = r0_valid ? sfa_s[sk * sfa_box_m + r_0] : 0.0f;
                sa1_arr[sk] = r1_valid ? sfa_s[sk * sfa_box_m + r_1] : 0.0f;
            }

            wgmma_wait<0>();
            if (lane_id == 0) mbarrier_arrive(&empty_bars[stage]);

            #pragma unroll
            for (int sk = 0; sk < 4; ++sk) {
                const float sc00 = sa0_arr[sk] * sb0_arr[sk];
                const float sc10 = sa1_arr[sk] * sb0_arr[sk];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    final_accum[i * 4 + 0] = fmaf(sc00, accum[sk][i * 4 + 0], final_accum[i * 4 + 0]);
                    final_accum[i * 4 + 1] = fmaf(sc00, accum[sk][i * 4 + 1], final_accum[i * 4 + 1]);
                    final_accum[i * 4 + 2] = fmaf(sc10, accum[sk][i * 4 + 2], final_accum[i * 4 + 2]);
                    final_accum[i * 4 + 3] = fmaf(sc10, accum[sk][i * 4 + 3], final_accum[i * 4 + 3]);
                }
                if constexpr (SFB_ROWS == 2) {
                    const float sc01 = sa0_arr[sk] * sb1_arr[sk];
                    const float sc11 = sa1_arr[sk] * sb1_arr[sk];
                    #pragma unroll
                    for (int i = 4; i < 8; ++i) {
                        final_accum[i * 4 + 0] = fmaf(sc01, accum[sk][i * 4 + 0], final_accum[i * 4 + 0]);
                        final_accum[i * 4 + 1] = fmaf(sc01, accum[sk][i * 4 + 1], final_accum[i * 4 + 1]);
                        final_accum[i * 4 + 2] = fmaf(sc11, accum[sk][i * 4 + 2], final_accum[i * 4 + 2]);
                        final_accum[i * 4 + 3] = fmaf(sc11, accum[sk][i * 4 + 3], final_accum[i * 4 + 3]);
                    }
                }
            }
            stage++;
            if (stage == STAGES) { stage = 0; phase ^= 1u; }
        }
    } else {
        float accum[2][kNumAccum];
        #pragma unroll 4
        for (int iter = 0; iter < num_k_iters; ++iter) {
            mbarrier_wait_parity(&full_bars[stage], phase);

            const uint64_t da_s = desc_a_base + static_cast<uint64_t>((stage * SMEM_A_STAGE) >> 4);
            const uint64_t db_s = desc_b_base + static_cast<uint64_t>((stage * SMEM_B_STAGE) >> 4);
            const float* sfa_s = reinterpret_cast<const float*>(smem_sfa + stage * SMEM_SFA_STAGE);

            float4 sb0_4 = *reinterpret_cast<const float4*>(smem_sfb + iter * 4);
            float4 sb1_4 = *reinterpret_cast<const float4*>(smem_sfb + split_k_scales + iter * 4);
            float sb0_arr[4] = {sb0_4.x, sb0_4.y, sb0_4.z, sb0_4.w};
            float sb1_arr[4] = {sb1_4.x, sb1_4.y, sb1_4.z, sb1_4.w};
            float sa0_arr[4], sa1_arr[4];
            #pragma unroll
            for (int sk = 0; sk < 4; ++sk) {
                sa0_arr[sk] = r0_valid ? sfa_s[sk * sfa_box_m + r_0] : 0.0f;
                sa1_arr[sk] = r1_valid ? sfa_s[sk * sfa_box_m + r_1] : 0.0f;
            }

            auto promote_bank = [&](int bank, int sk) {
                const float sc00 = sa0_arr[sk] * sb0_arr[sk];
                const float sc10 = sa1_arr[sk] * sb0_arr[sk];
                const float sc01 = sa0_arr[sk] * sb1_arr[sk];
                const float sc11 = sa1_arr[sk] * sb1_arr[sk];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    final_accum[i * 4 + 0] = fmaf(sc00, accum[bank][i * 4 + 0], final_accum[i * 4 + 0]);
                    final_accum[i * 4 + 1] = fmaf(sc00, accum[bank][i * 4 + 1], final_accum[i * 4 + 1]);
                    final_accum[i * 4 + 2] = fmaf(sc10, accum[bank][i * 4 + 2], final_accum[i * 4 + 2]);
                    final_accum[i * 4 + 3] = fmaf(sc10, accum[bank][i * 4 + 3], final_accum[i * 4 + 3]);
                }
                #pragma unroll
                for (int i = 4; i < 8; ++i) {
                    final_accum[i * 4 + 0] = fmaf(sc01, accum[bank][i * 4 + 0], final_accum[i * 4 + 0]);
                    final_accum[i * 4 + 1] = fmaf(sc01, accum[bank][i * 4 + 1], final_accum[i * 4 + 1]);
                    final_accum[i * 4 + 2] = fmaf(sc11, accum[bank][i * 4 + 2], final_accum[i * 4 + 2]);
                    final_accum[i * 4 + 3] = fmaf(sc11, accum[bank][i * 4 + 3], final_accum[i * 4 + 3]);
                }
            };

            #pragma unroll
            for (int sk = 0; sk < 4; ++sk) {
                int bank = sk & 1;
                #pragma unroll
                for (int i = 0; i < kNumAccum; ++i) wgmma_fence_operand(accum[bank][i]);
                wgmma_fence();
                wgmma_m64nNk32_e4m3_zero<64>(da_s + sk * 2, db_s + sk * 2, accum[bank]);
                wgmma_commit();
                #pragma unroll
                for (int i = 0; i < kNumAccum; ++i) wgmma_fence_operand(accum[bank][i]);
                if (sk > 0) {
                    wgmma_wait<1>();
                    promote_bank((sk - 1) & 1, sk - 1);
                }
            }
            wgmma_wait<0>();
            if (lane_id == 0) mbarrier_arrive(&empty_bars[stage]);
            promote_bank(1, 3);
            stage++;
            if (stage == STAGES) { stage = 0; phase ^= 1u; }
        }
    }

    named_bar_sync<1, NUM_MATH_THREADS>();
    constexpr int ITERS_N = BLOCK_N / 8;
    if constexpr (SPLIT_K == 1) {
        __nv_bfloat16* smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buf);
        if (r0_valid) {
            #pragma unroll
            for (int i = 0; i < ITERS_N; ++i) {
                int c0 = i * 8 + n_0_base;
                smem_d[r_0 * BLOCK_N + c0 + 0] = __float2bfloat16_rn(final_accum[i * 4 + 0]);
                smem_d[r_0 * BLOCK_N + c0 + 1] = __float2bfloat16_rn(final_accum[i * 4 + 1]);
            }
        }
        if (r1_valid) {
            #pragma unroll
            for (int i = 0; i < ITERS_N; ++i) {
                int c0 = i * 8 + n_0_base;
                smem_d[r_1 * BLOCK_N + c0 + 0] = __float2bfloat16_rn(final_accum[i * 4 + 2]);
                smem_d[r_1 * BLOCK_N + c0 + 1] = __float2bfloat16_rn(final_accum[i * 4 + 3]);
            }
        }
        named_bar_sync<1, NUM_MATH_THREADS>();
        constexpr int VEC_PER_ROW = BLOCK_N / 8;
        const int num_vec = M * VEC_PER_ROW;
        const uint4* src = reinterpret_cast<const uint4*>(smem_d);
        for (int idx = tid; idx < num_vec; idx += NUM_MATH_THREADS) {
            int rm = idx / VEC_PER_ROW;
            int vec_c = idx - rm * VEC_PER_ROW;
            reinterpret_cast<uint4*>(D + (size_t)rm * N + n_base)[vec_c] = src[idx];
        }
    } else {
        const int out_elems = M * BLOCK_N;
        float* smem_f = reinterpret_cast<float*>(smem_buf);
        if (r0_valid) {
            #pragma unroll
            for (int i = 0; i < ITERS_N; ++i) {
                int c0 = i * 8 + n_0_base;
                smem_f[r_0 * BLOCK_N + c0 + 0] = final_accum[i * 4 + 0];
                smem_f[r_0 * BLOCK_N + c0 + 1] = final_accum[i * 4 + 1];
            }
        }
        if (r1_valid) {
            #pragma unroll
            for (int i = 0; i < ITERS_N; ++i) {
                int c0 = i * 8 + n_0_base;
                smem_f[r_1 * BLOCK_N + c0 + 0] = final_accum[i * 4 + 2];
                smem_f[r_1 * BLOCK_N + c0 + 1] = final_accum[i * 4 + 3];
            }
        }
        named_bar_sync<1, NUM_MATH_THREADS>();

        float* tile_ws = workspace + (size_t)n_tile * SPLIT_K * out_elems;
        float* my_ws = tile_ws + (size_t)k_split * out_elems;
        const int num_f4 = out_elems >> 2;
        const float4* src4 = reinterpret_cast<const float4*>(smem_f);
        float4* dst4 = reinterpret_cast<float4*>(my_ws);
        for (int idx = tid; idx < num_f4; idx += NUM_MATH_THREADS) dst4[idx] = src4[idx];
        named_bar_sync<1, NUM_MATH_THREADS>();

        __shared__ int is_last;
        if (tid == 0) {
            int prev;
            asm volatile("atom.global.acq_rel.gpu.add.s32 %0, [%1], 1;"
                         : "=r"(prev) : "l"(counters + n_tile) : "memory");
            is_last = (prev == SPLIT_K - 1) ? 1 : 0;
        }
        named_bar_sync<1, NUM_MATH_THREADS>();

        if (is_last) {
            constexpr int F4_PER_ROW = BLOCK_N / 4;
            for (int idx = tid; idx < num_f4; idx += NUM_MATH_THREADS) {
                int rm = idx / F4_PER_ROW;
                int c4 = idx - rm * F4_PER_ROW;
                float4 acc4 = make_float4(0.f, 0.f, 0.f, 0.f);
                #pragma unroll
                for (int s = 0; s < SPLIT_K; ++s) {
                    float4 v = __ldcg(reinterpret_cast<const float4*>(tile_ws + (size_t)s * out_elems) + idx);
                    acc4.x += v.x; acc4.y += v.y; acc4.z += v.z; acc4.w += v.w;
                }
                __nv_bfloat162 b01 = __floats2bfloat162_rn(acc4.x, acc4.y);
                __nv_bfloat162 b23 = __floats2bfloat162_rn(acc4.z, acc4.w);
                uint2 packed;
                packed.x = *reinterpret_cast<uint32_t*>(&b01);
                packed.y = *reinterpret_cast<uint32_t*>(&b23);
                reinterpret_cast<uint2*>(D + (size_t)rm * N + n_base)[c4] = packed;
            }
            named_bar_sync<1, NUM_MATH_THREADS>();
            if (tid == 0) counters[n_tile] = 0;
        }
    }
}

// SwapAB Zero-TMA-OOB TMA+WGMMA (M <= 32, M_TILE in {8, 16, 24, 32})
template <int M_TILE, int STAGES, int SPLIT_K, int MIN_BLOCKS>
__global__ void __launch_bounds__(256, MIN_BLOCKS)
sm90_fp8_g32_swap_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_sfa,
    const float* __restrict__ SFB,
    __nv_bfloat16* __restrict__ D,
    float* __restrict__ workspace,
    int* __restrict__ counters,
    int M, int N, int K,
    int tx_bytes, int sfa_box_m
) {
    constexpr int BLOCK_N = 64, BLOCK_K = 128;
    constexpr int kNumAccum = M_TILE / 2;
    constexpr int SMEM_B_STAGE = BLOCK_N * BLOCK_K;
    constexpr int SMEM_A_STAGE = M_TILE * BLOCK_K;
    constexpr int SMEM_SFA_STAGE = align_up(4 * M_TILE * static_cast<int>(sizeof(float)), 128);
    constexpr int TMA_REGS = (MIN_BLOCKS == 2) ? 24 : 40;
    constexpr int MATH_REGS = (MIN_BLOCKS == 2) ? 208 : 248;

    const int n_tile = blockIdx.x;
    const int k_split = blockIdx.y;
    const int n_base = n_tile * BLOCK_N;

    const int total_k_blocks = K >> 7;
    const int k_per = total_k_blocks / SPLIT_K;
    const int k_rem = total_k_blocks % SPLIT_K;
    const int k_block_start = k_split * k_per + (k_split < k_rem ? k_split : k_rem);
    const int num_k_iters = k_per + (k_split < k_rem ? 1 : 0);
    if (num_k_iters <= 0) return;

    const int k_scales = K >> 5;
    const int split_k_scales = num_k_iters * 4;

    const int tid = threadIdx.x;
    const int warp_id = __shfl_sync(0xffffffff, tid >> 5, 0);
    const int lane_id = tid & 31;

    extern __shared__ __align__(1024) uint8_t smem_buf[];
    uint8_t* smem_b = smem_buf;
    uint8_t* smem_a = smem_b + STAGES * SMEM_B_STAGE;
    uint8_t* smem_sfa = smem_a + STAGES * SMEM_A_STAGE;
    uint64_t* full_bars = reinterpret_cast<uint64_t*>(smem_sfa + STAGES * SMEM_SFA_STAGE);
    uint64_t* empty_bars = full_bars + STAGES;
    float* smem_sfb = reinterpret_cast<float*>(empty_bars + STAGES);

    if (warp_id == 4 && lane_id == 0) {
        asm volatile("prefetch.tensormap [%0];\n" :: "l"(&tma_a));
        asm volatile("prefetch.tensormap [%0];\n" :: "l"(&tma_b));
        asm volatile("prefetch.tensormap [%0];\n" :: "l"(&tma_sfa));
    }
    if (warp_id == 5 && lane_id == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbarrier_init(&full_bars[s], 1);
            mbarrier_init(&empty_bars[s], 4);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (warp_id >= 4) {
        setmaxnreg_dec<TMA_REGS>();
        if (warp_id == 4 && lane_id == 0) {
            int stage = 0;
            uint32_t phase = 0;
            for (int iter = 0; iter < num_k_iters; ++iter) {
                mbarrier_wait_parity(&empty_bars[stage], phase ^ 1u);
                const int kb = k_block_start + iter;
                const int k_coord = kb * BLOCK_K;
                tma_load_2d(&tma_b, &full_bars[stage], smem_b + stage * SMEM_B_STAGE, k_coord, n_base);
                tma_load_2d(&tma_a, &full_bars[stage], smem_a + stage * SMEM_A_STAGE, k_coord, 0);
                tma_load_2d(&tma_sfa, &full_bars[stage], smem_sfa + stage * SMEM_SFA_STAGE, 0, kb * 4);
                mbarrier_arrive_expect_tx(&full_bars[stage], tx_bytes);
                stage++;
                if (stage == STAGES) { stage = 0; phase ^= 1u; }
            }
        }
        return;
    }

    setmaxnreg_inc<MATH_REGS>();

    {
        const int sfb_row0 = n_base >> 5;
        const int sf_k_start = k_block_start * 4;
        const int total_sfb = 2 * split_k_scales;
        for (int idx = tid; idx < total_sfb; idx += 128) {
            int r = (idx >= split_k_scales) ? 1 : 0;
            int c = idx - r * split_k_scales;
            smem_sfb[idx] = __ldg(SFB + (size_t)(sfb_row0 + r) * k_scales + sf_k_start + c);
        }
        named_bar_sync<1, 128>();
    }

    const int r_0 = warp_id * 16 + (lane_id >> 2);
    const int r_1 = r_0 + 8;
    const int col_idx = lane_id & 3;
    const int token_0_base = col_idx * 2;
    const float* my_sfb = smem_sfb + (r_0 >> 5) * split_k_scales;

    const uint64_t desc_a_base = make_wgmma_desc(smem_b);
    const uint64_t desc_b_base = make_wgmma_desc(smem_a);

    float final_accum[kNumAccum];
    #pragma unroll
    for (int i = 0; i < kNumAccum; ++i) final_accum[i] = 0.0f;

    float accum[4][kNumAccum];
    int stage = 0;
    uint32_t phase = 0;

    #pragma unroll 4
    for (int iter = 0; iter < num_k_iters; ++iter) {
        mbarrier_wait_parity(&full_bars[stage], phase);

        const uint64_t da_s = desc_a_base + static_cast<uint64_t>((stage * SMEM_B_STAGE) >> 4);
        const uint64_t db_s = desc_b_base + static_cast<uint64_t>((stage * SMEM_A_STAGE) >> 4);
        const float* sfa_s = reinterpret_cast<const float*>(smem_sfa + stage * SMEM_SFA_STAGE);

        #pragma unroll
        for (int sk = 0; sk < 4; ++sk)
            #pragma unroll
            for (int i = 0; i < kNumAccum; ++i)
                wgmma_fence_operand(accum[sk][i]);
        wgmma_fence();
        #pragma unroll
        for (int sk = 0; sk < 4; ++sk) {
            wgmma_m64nNk32_e4m3_zero<M_TILE>(da_s + sk * 2, db_s + sk * 2, accum[sk]);
        }
        wgmma_commit();
        #pragma unroll
        for (int sk = 0; sk < 4; ++sk)
            #pragma unroll
            for (int i = 0; i < kNumAccum; ++i)
                wgmma_fence_operand(accum[sk][i]);

        float4 sb4 = *reinterpret_cast<const float4*>(my_sfb + iter * 4);
        float sb_arr[4] = {sb4.x, sb4.y, sb4.z, sb4.w};
        float2 sa_reg[4][kNumAccum / 4];
        #pragma unroll
        for (int sk = 0; sk < 4; ++sk) {
            const float* sfa_sub = sfa_s + sk * sfa_box_m;
            #pragma unroll
            for (int i = 0; i < kNumAccum / 4; ++i) {
                int tm0 = i * 8 + token_0_base;
                if (tm0 + 1 < sfa_box_m) {
                    float2 pair = *reinterpret_cast<const float2*>(sfa_sub + tm0);
                    sa_reg[sk][i] = make_float2(
                        (tm0 < M) ? pair.x : 0.0f,
                        (tm0 + 1 < M) ? pair.y : 0.0f
                    );
                } else {
                    sa_reg[sk][i] = make_float2(0.0f, 0.0f);
                }
            }
        }

        wgmma_wait<0>();
        if (lane_id == 0) mbarrier_arrive(&empty_bars[stage]);

        #pragma unroll
        for (int sk = 0; sk < 4; ++sk) {
            const float sb = sb_arr[sk];
            #pragma unroll
            for (int i = 0; i < kNumAccum / 4; ++i) {
                const float sc0 = sa_reg[sk][i].x * sb;
                const float sc1 = sa_reg[sk][i].y * sb;
                final_accum[i * 4 + 0] = fmaf(sc0, accum[sk][i * 4 + 0], final_accum[i * 4 + 0]);
                final_accum[i * 4 + 1] = fmaf(sc1, accum[sk][i * 4 + 1], final_accum[i * 4 + 1]);
                final_accum[i * 4 + 2] = fmaf(sc0, accum[sk][i * 4 + 2], final_accum[i * 4 + 2]);
                final_accum[i * 4 + 3] = fmaf(sc1, accum[sk][i * 4 + 3], final_accum[i * 4 + 3]);
            }
        }
        stage++;
        if (stage == STAGES) { stage = 0; phase ^= 1u; }
    }

    named_bar_sync<1, 128>();
    if constexpr (SPLIT_K == 1) {
        __nv_bfloat16* smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buf);
        #pragma unroll
        for (int i = 0; i < kNumAccum / 4; ++i) {
            int rm0 = i * 8 + token_0_base;
            int rm1 = rm0 + 1;
            if (rm0 < M) {
                smem_d[rm0 * BLOCK_N + r_0] = __float2bfloat16_rn(final_accum[i * 4 + 0]);
                smem_d[rm0 * BLOCK_N + r_1] = __float2bfloat16_rn(final_accum[i * 4 + 2]);
            }
            if (rm1 < M) {
                smem_d[rm1 * BLOCK_N + r_0] = __float2bfloat16_rn(final_accum[i * 4 + 1]);
                smem_d[rm1 * BLOCK_N + r_1] = __float2bfloat16_rn(final_accum[i * 4 + 3]);
            }
        }
        named_bar_sync<1, 128>();
        const int num_vec = M * 8;
        const uint4* src = reinterpret_cast<const uint4*>(smem_d);
        for (int idx = tid; idx < num_vec; idx += 128) {
            int rm = idx >> 3;
            int vec_c = idx & 7;
            reinterpret_cast<uint4*>(D + (size_t)rm * N + n_base)[vec_c] = src[idx];
        }
    } else {
        const int out_elems = M * BLOCK_N;
        float* smem_f = reinterpret_cast<float*>(smem_buf);
        #pragma unroll
        for (int i = 0; i < kNumAccum / 4; ++i) {
            int rm0 = i * 8 + token_0_base;
            int rm1 = rm0 + 1;
            if (rm0 < M) {
                smem_f[rm0 * BLOCK_N + r_0] = final_accum[i * 4 + 0];
                smem_f[rm0 * BLOCK_N + r_1] = final_accum[i * 4 + 2];
            }
            if (rm1 < M) {
                smem_f[rm1 * BLOCK_N + r_0] = final_accum[i * 4 + 1];
                smem_f[rm1 * BLOCK_N + r_1] = final_accum[i * 4 + 3];
            }
        }
        named_bar_sync<1, 128>();

        float* tile_ws = workspace + (size_t)n_tile * SPLIT_K * out_elems;
        float* my_ws = tile_ws + (size_t)k_split * out_elems;
        const int num_f4 = out_elems >> 2;
        const float4* src4 = reinterpret_cast<const float4*>(smem_f);
        float4* dst4 = reinterpret_cast<float4*>(my_ws);
        for (int idx = tid; idx < num_f4; idx += 128) dst4[idx] = src4[idx];
        named_bar_sync<1, 128>();

        __shared__ int is_last;
        if (tid == 0) {
            int prev;
            asm volatile("atom.global.acq_rel.gpu.add.s32 %0, [%1], 1;"
                         : "=r"(prev) : "l"(counters + n_tile) : "memory");
            is_last = (prev == SPLIT_K - 1) ? 1 : 0;
        }
        named_bar_sync<1, 128>();

        if (is_last) {
            for (int idx = tid; idx < num_f4; idx += 128) {
                int rm = idx >> 4;
                int c4 = idx & 15;
                float4 acc4 = make_float4(0.f, 0.f, 0.f, 0.f);
                #pragma unroll
                for (int s = 0; s < SPLIT_K; ++s) {
                    float4 v = __ldcg(reinterpret_cast<const float4*>(tile_ws + (size_t)s * out_elems) + idx);
                    acc4.x += v.x; acc4.y += v.y; acc4.z += v.z; acc4.w += v.w;
                }
                __nv_bfloat162 b01 = __floats2bfloat162_rn(acc4.x, acc4.y);
                __nv_bfloat162 b23 = __floats2bfloat162_rn(acc4.z, acc4.w);
                uint2 packed;
                packed.x = *reinterpret_cast<uint32_t*>(&b01);
                packed.y = *reinterpret_cast<uint32_t*>(&b23);
                reinterpret_cast<uint2*>(D + (size_t)rm * N + n_base)[c4] = packed;
            }
            named_bar_sync<1, 128>();
            if (tid == 0) counters[n_tile] = 0;
        }
    }
}

static inline CUresult make_tma_2d_fp8(
    CUtensorMap* desc, const void* ptr, int rows, int cols, int box_rows, int box_cols
) {
    uint64_t globalDim[2] = { static_cast<uint64_t>(cols), static_cast<uint64_t>(rows) };
    uint64_t globalStrides[1] = { static_cast<uint64_t>(cols) };
    uint32_t boxDim[2] = { static_cast<uint32_t>(box_cols), static_cast<uint32_t>(box_rows) };
    uint32_t elemStrides[2] = { 1, 1 };
    return cuTensorMapEncodeTiled(
        desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr),
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

static inline CUresult make_tma_2d_sfa(
    CUtensorMap* desc, const void* ptr, int sfa_global_m, int sfa_stride_k, int k_scales, int sfa_box_m
) {
    uint64_t globalDim[2] = { static_cast<uint64_t>(sfa_global_m), static_cast<uint64_t>(k_scales) };
    uint64_t globalStrides[1] = { static_cast<uint64_t>(sfa_stride_k) * sizeof(float) };
    uint32_t boxDim[2] = { static_cast<uint32_t>(sfa_box_m), 4u };
    uint32_t elemStrides[2] = { 1, 1 };
    return cuTensorMapEncodeTiled(
        desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<void*>(ptr),
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

constexpr int kMaxDynamicSmem = 224 * 1024;

template <int BLOCK_N, int NUM_MATH_WGS, int STAGES, int SPLIT_K, int MIN_BLOCKS>
int launch_ns(
    const void* A, const void* SFA, const void* B, const void* SFB,
    void* D, void* ws, void* counters,
    int M, int N, int K, int sfa_global_m, int sfa_stride_k, cudaStream_t stream
) {
    constexpr int BLOCK_M = NUM_MATH_WGS * 64;
    constexpr int BLOCK_K = 128;
    constexpr int SFB_ROWS = BLOCK_N / 32;
    constexpr int SMEM_SFA_STAGE = align_up(4 * BLOCK_M * (int)sizeof(float), 128);
    int k_blocks_per_split = ((K >> 7) + SPLIT_K - 1) / SPLIT_K;
    int sfb_bytes = SFB_ROWS * k_blocks_per_split * 4 * (int)sizeof(float);
    int smem_pipe = STAGES * (BLOCK_N * BLOCK_K + BLOCK_M * BLOCK_K + SMEM_SFA_STAGE)
                  + STAGES * 16 + sfb_bytes + 256;
    int smem_epi = (SPLIT_K == 1) ? (M * BLOCK_N * 2) : (M * BLOCK_N * 4);
    int smem_bytes = smem_pipe > smem_epi ? smem_pipe : smem_epi;
    if (smem_bytes > kMaxDynamicSmem) return -2;

    int sfa_box_m = align_up(M, 4);
    int tx_bytes = BLOCK_N * BLOCK_K + M * BLOCK_K + 4 * sfa_box_m * (int)sizeof(float);

    CUtensorMap tma_a, tma_b, tma_sfa;
    if (make_tma_2d_fp8(&tma_a, A, M, K, M, BLOCK_K) != CUDA_SUCCESS) return -3;
    if (make_tma_2d_fp8(&tma_b, B, N, K, BLOCK_N, BLOCK_K) != CUDA_SUCCESS) return -4;
    if (make_tma_2d_sfa(&tma_sfa, SFA, sfa_global_m, sfa_stride_k, K >> 5, sfa_box_m) != CUDA_SUCCESS) return -5;

    sm90_fp8_g32_nonswap_kernel<BLOCK_N, NUM_MATH_WGS, STAGES, SPLIT_K, MIN_BLOCKS>
        <<<dim3(N / BLOCK_N, SPLIT_K), dim3(NUM_MATH_WGS * 128 + 128), smem_bytes, stream>>>(
            tma_a, tma_b, tma_sfa, (const float*)SFB, (__nv_bfloat16*)D, (float*)ws, (int*)counters,
            M, N, K, tx_bytes, sfa_box_m
        );
    if (cudaGetLastError() != cudaSuccess) return -6;
    return 0;
}

template <int M_TILE, int STAGES, int SPLIT_K, int MIN_BLOCKS>
int launch_sw(
    const void* A, const void* SFA, const void* B, const void* SFB,
    void* D, void* ws, void* counters,
    int M, int N, int K, int sfa_global_m, int sfa_stride_k, cudaStream_t stream
) {
    constexpr int BLOCK_N = 64, BLOCK_K = 128;
    constexpr int SMEM_SFA_STAGE = align_up(4 * M_TILE * (int)sizeof(float), 128);
    int k_blocks_per_split = ((K >> 7) + SPLIT_K - 1) / SPLIT_K;
    int sfb_bytes = 2 * k_blocks_per_split * 4 * (int)sizeof(float);
    int smem_pipe = STAGES * (BLOCK_N * BLOCK_K + M_TILE * BLOCK_K + SMEM_SFA_STAGE)
                  + STAGES * 16 + sfb_bytes + 256;
    int smem_epi = (SPLIT_K == 1) ? (M * BLOCK_N * 2) : (M * BLOCK_N * 4);
    int smem_bytes = smem_pipe > smem_epi ? smem_pipe : smem_epi;
    if (smem_bytes > kMaxDynamicSmem) return -2;

    int sfa_box_m = align_up(M, 4);
    int tx_bytes = BLOCK_N * BLOCK_K + M * BLOCK_K + 4 * sfa_box_m * (int)sizeof(float);

    CUtensorMap tma_a, tma_b, tma_sfa;
    if (make_tma_2d_fp8(&tma_a, A, M, K, M, BLOCK_K) != CUDA_SUCCESS) return -3;
    if (make_tma_2d_fp8(&tma_b, B, N, K, BLOCK_N, BLOCK_K) != CUDA_SUCCESS) return -4;
    if (make_tma_2d_sfa(&tma_sfa, SFA, sfa_global_m, sfa_stride_k, K >> 5, sfa_box_m) != CUDA_SUCCESS) return -5;

    sm90_fp8_g32_swap_kernel<M_TILE, STAGES, SPLIT_K, MIN_BLOCKS>
        <<<dim3(N / BLOCK_N, SPLIT_K), dim3(256), smem_bytes, stream>>>(
            tma_a, tma_b, tma_sfa, (const float*)SFB, (__nv_bfloat16*)D, (float*)ws, (int*)counters,
            M, N, K, tx_bytes, sfa_box_m
        );
    if (cudaGetLastError() != cudaSuccess) return -6;
    return 0;
}

template <typename KernelFn>
static inline bool configure_smem(KernelFn fn) {
    return cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSmem) == cudaSuccess;
}

} // namespace

// Must be called once per CUDA device outside CUDA Graph capture to configure dynamic SMEM limits.
extern "C" int init_sm90_small_m_fp8_gemm() {
    bool ok = true;
    #define INIT_SW(MT, ST, SK, MB) ok = ok && configure_smem(sm90_fp8_g32_swap_kernel<MT, ST, SK, MB>);
    #define INIT_SW_SET(MT, ST1, ST2) \
        INIT_SW(MT, ST1, 1, 1) INIT_SW(MT, ST1, 3, 1) INIT_SW(MT, ST1, 4, 1) \
        INIT_SW(MT, ST2, 1, 2) INIT_SW(MT, ST2, 4, 2)
    INIT_SW_SET(8, 8, 6)
    INIT_SW_SET(16, 8, 6)
    INIT_SW_SET(24, 8, 5)
    INIT_SW_SET(32, 8, 5)

    #define INIT_NS(BN, WGS, ST, SK, MB) ok = ok && configure_smem(sm90_fp8_g32_nonswap_kernel<BN, WGS, ST, SK, MB>);
    // NS32 (M <= 64)
    INIT_NS(32, 1, 8, 1, 1) INIT_NS(32, 1, 8, 2, 1)
    INIT_NS(32, 1, 6, 1, 2) INIT_NS(32, 1, 6, 2, 2)
    // NS64 (33 <= M <= 64 wide-N, and Dual-WG 65 <= M <= 128)
    INIT_NS(64, 1, 5, 1, 2)
    INIT_NS(64, 2, 6, 1, 1) INIT_NS(64, 2, 6, 3, 1) INIT_NS(64, 2, 6, 4, 1)
    if (!ok) {
        (void)cudaGetLastError();
        return -1;
    }
    return 0;
}

// mode: 0 = Swap64, 1 = NS32, 2 = NS64
extern "C" int run_sm90_small_m_fp8_gemm(
    const void* A, const void* SFA, const void* B, const void* SFB,
    void* D, void* ws, void* counters,
    int M, int N, int K, int sfa_global_m, int sfa_stride_k,
    int mode, int split_k, int min_blocks,
    void* stream_ptr
) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    #define D_SW(MT, ST, SK, MB) \
        if (split_k == SK && min_blocks == MB) \
            return launch_sw<MT, ST, SK, MB>(A, SFA, B, SFB, D, ws, counters, M, N, K, sfa_global_m, sfa_stride_k, stream);
    #define D_SW_SET(MT, ST1, ST2) \
        D_SW(MT, ST1, 1, 1) D_SW(MT, ST1, 3, 1) D_SW(MT, ST1, 4, 1) \
        D_SW(MT, ST2, 1, 2) D_SW(MT, ST2, 4, 2)
    #define D_NS(BN, WGS, ST, SK, MB) \
        if (split_k == SK && min_blocks == MB) \
            return launch_ns<BN, WGS, ST, SK, MB>(A, SFA, B, SFB, D, ws, counters, M, N, K, sfa_global_m, sfa_stride_k, stream);

    if (mode == 0) {
        if (M <= 8)       { D_SW_SET(8, 8, 6) }
        else if (M <= 16) { D_SW_SET(16, 8, 6) }
        else if (M <= 24) { D_SW_SET(24, 8, 5) }
        else if (M <= 32) { D_SW_SET(32, 8, 5) }
    } else if (mode == 1) {
        if (M <= 64) {
            D_NS(32, 1, 8, 1, 1) D_NS(32, 1, 8, 2, 1)
            D_NS(32, 1, 6, 1, 2) D_NS(32, 1, 6, 2, 2)
        }
    } else if (mode == 2) {
        if (M <= 64) {
            D_NS(64, 1, 5, 1, 2)
        } else if (M <= 128) {
            D_NS(64, 2, 6, 1, 1) D_NS(64, 2, 6, 3, 1) D_NS(64, 2, 6, 4, 1)
        }
    }
    return -1;
}
"""
)


def _find_nvcc() -> Optional[str]:
    for candidate in (
        os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nvcc"),
        "/usr/local/cuda/bin/nvcc",
        shutil.which("nvcc") or "",
    ):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _cache_dir() -> str:
    base = os.environ.get("RTP_SM90_FP8_GEMM_CACHE_DIR")
    if base:
        return base
    home = os.path.expanduser("~")
    if home and os.path.isdir(home) and os.access(home, os.W_OK):
        return os.path.join(home, ".cache", "rtp_llm", "sm90_small_m_fp8_gemm")
    return os.path.join(tempfile.gettempdir(), "rtp_llm_sm90_small_m_fp8_gemm")


def _load_library() -> Optional[ctypes.CDLL]:
    global _CACHED_LIB, _LIB_LOAD_FAILED
    if _CACHED_LIB is not None:
        return _CACHED_LIB
    if _LIB_LOAD_FAILED:
        return None
    with _LIB_LOCK:
        if _CACHED_LIB is not None:
            return _CACHED_LIB
        if _LIB_LOAD_FAILED:
            return None
        nvcc = _find_nvcc()
        if nvcc is None:
            logger.warning("SM90 small-M FP8 GEMM disabled: nvcc not found")
            _LIB_LOAD_FAILED = True
            return None
        try:
            ver = subprocess.check_output([nvcc, "--version"], text=True, stderr=subprocess.STDOUT)
            digest = hashlib.sha256((_CUDA_SOURCE + "\n" + ver).encode("utf-8")).hexdigest()[:16]
            cache = _cache_dir()
            os.makedirs(cache, exist_ok=True)
            so_path = os.path.join(cache, f"libsm90_small_m_fp8_gemm_{digest}.so")
            if not os.path.isfile(so_path):
                pid = os.getpid()
                cu_tmp = os.path.join(cache, f"sm90_small_m_fp8_gemm_{digest}_{pid}.cu")
                so_tmp = os.path.join(cache, f"libsm90_small_m_fp8_gemm_{digest}_{pid}.so.tmp")
                with open(cu_tmp, "w", encoding="utf-8") as fh:
                    fh.write(_CUDA_SOURCE)
                try:
                    subprocess.check_call(
                        [
                            nvcc,
                            "-O3",
                            "-gencode=arch=compute_90a,code=sm_90a",
                            "--ptxas-options=--register-usage-level=10",
                            "--shared",
                            "-Xcompiler",
                            "-fPIC",
                            "-std=c++17",
                            "-lcuda",
                            "-diag-suppress=177",
                            cu_tmp,
                            "-o",
                            so_tmp,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    os.replace(so_tmp, so_path)
                finally:
                    for path in (cu_tmp, so_tmp):
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                            except OSError:
                                pass
            lib = ctypes.CDLL(so_path)
            lib.init_sm90_small_m_fp8_gemm.argtypes = []
            lib.init_sm90_small_m_fp8_gemm.restype = ctypes.c_int
            lib.run_sm90_small_m_fp8_gemm.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            lib.run_sm90_small_m_fp8_gemm.restype = ctypes.c_int
            _CACHED_LIB = lib
            return lib
        except Exception as exc:
            logger.warning("SM90 small-M FP8 GEMM JIT compile/load failed; falling back to DeepGEMM: %s", exc)
            _LIB_LOAD_FAILED = True
            return None


def _is_sm90_device(dev_idx: int) -> bool:
    capable = _SM90_CAPABLE_DEVICES.get(dev_idx)
    if capable is not None:
        return capable
    capable = torch.cuda.is_available() and torch.cuda.get_device_capability(dev_idx) == (9, 0)
    _SM90_CAPABLE_DEVICES[dev_idx] = capable
    return capable


def _ensure_device_initialized(lib: ctypes.CDLL, dev_idx: int, *, capturing: bool) -> bool:
    if dev_idx in _INITIALIZED_DEVICES:
        return True
    if capturing:
        return False
    with _DEVICE_BUFFERS_LOCK:
        if dev_idx in _INITIALIZED_DEVICES:
            return True
        with torch.cuda.device(dev_idx):
            rc = lib.init_sm90_small_m_fp8_gemm()
            if rc != 0:
                logger.warning("SM90 small-M FP8 GEMM cudaFuncSetAttribute failed on cuda:%d (rc=%d)", dev_idx, rc)
                return False
            if dev_idx not in _CAPTURE_BUFFERS:
                dev = torch.device("cuda", dev_idx)
                _CAPTURE_BUFFERS[dev_idx] = (
                    torch.empty(_DEFAULT_WS_FLOATS, dtype=torch.float32, device=dev),
                    torch.zeros(_DEFAULT_COUNTER_ELEMS, dtype=torch.int32, device=dev),
                )
        _INITIALIZED_DEVICES.add(dev_idx)
        return True


def _get_stream_buffers(
    dev_idx: int, stream_ptr: int, *, capturing: bool
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    key = (dev_idx, stream_ptr)
    bufs = _STREAM_BUFFERS.get(key)
    if bufs is not None:
        return bufs
    if capturing:
        return _CAPTURE_BUFFERS.get(dev_idx)
    with _DEVICE_BUFFERS_LOCK:
        bufs = _STREAM_BUFFERS.get(key)
        if bufs is not None:
            return bufs
        dev = torch.device("cuda", dev_idx)
        ws = torch.empty(_DEFAULT_WS_FLOATS, dtype=torch.float32, device=dev)
        counters = torch.zeros(_DEFAULT_COUNTER_ELEMS, dtype=torch.int32, device=dev)
        bufs = (ws, counters)
        if len(_STREAM_BUFFERS) < _MAX_CACHED_STREAMS:
            _STREAM_BUFFERS[key] = bufs
        if dev_idx not in _CAPTURE_BUFFERS:
            _CAPTURE_BUFFERS[dev_idx] = (
                torch.empty(_DEFAULT_WS_FLOATS, dtype=torch.float32, device=dev),
                torch.zeros(_DEFAULT_COUNTER_ELEMS, dtype=torch.int32, device=dev),
            )
        return bufs


def select_config(m: int, n: int, k: int) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(mode, split_k, min_blocks, max_sfa_box_m)`` or ``None`` for DeepGEMM fallback.

    Modes: ``0 = Swap64``, ``1 = NS32``, ``2 = NS64`` (Dual-WG for ``M > 64``).
    Empirically calibrated on L20Z (SM90, 132 SMs, 3.01 TB/s HBM) under cold-L2
    CUDA-Graph replay across ``M in [1..128]``.
    """
    if m <= 0 or m > 128 or n < 64 or (n % 64) != 0 or k < 128 or (k % 128) != 0:
        return None
    k_blocks = k >> 7

    if m <= 32:
        m_tile = 8 if m <= 8 else (16 if m <= 16 else (24 if m <= 24 else 32))
        if n <= 2048 and k_blocks >= 8:
            if m <= 24:
                # Narrow-N (e.g. qkv_a 1792x5120): Split-K saturates 132 SMs!
                if m <= 4 and k_blocks >= 12:
                    return (0, 3, 1, m_tile)
                return (0, 4, 2, m_tile)
            # 25 <= M <= 32, narrow-N: NS32 SK=2 beats Swap64!
            return (1, 2, 1, 64)
        # Wider N (o_proj, kv_b/q_b, engram, fc, shared_w13, shared_w2, index_wq)
        mb = 2 if n >= 16384 else 1
        return (0, 1, mb, m_tile)

    if m <= 64:
        if n <= 2048 and k_blocks >= 8:
            return (1, 2, 1, 64)
        if n <= 8192:
            # NS32 doubles N-tiles (160 on o_proj, 144 on shared_w13, 128 on index_wq)
            return (1, 1, 2, 64)
        # Very wide N (kv_b/q_b 32768x1280, engram 25600x6144)
        return (2, 1, 2, 64)

    # 65 <= M <= 128 (e.g. M=80 BS=16 draft): Dual-WG BM=128 reads weight B only once!
    if n <= 5120 and k >= 2304:
        if n <= 2048 and k_blocks >= 8:
            return (2, 4, 1, 128)
        if k >= 8192:
            return (2, 3, 1, 128)
        return (2, 1, 1, 128)
    # Wide-N (kv_b/q_b, engram) or short-K (index_wq) at M > 64: defer to DeepGEMM
    return None


def warmup_sm90_small_m_fp8_gemm(device: Optional[torch.device] = None) -> bool:
    """Eagerly compile/load the kernel library, configure dynamic SMEM, and allocate buffers."""
    if not is_enabled() or not torch.cuda.is_available():
        return False
    dev = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if dev.type != "cuda":
        return False
    dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if not _is_sm90_device(dev_idx):
        return False
    lib = _load_library()
    if lib is None:
        return False
    with torch.cuda.device(dev_idx):
        if not _ensure_device_initialized(lib, dev_idx, capturing=False):
            return False
        stream_ptr = torch.cuda.current_stream(dev_idx).cuda_stream
        bufs = _get_stream_buffers(dev_idx, stream_ptr, capturing=False)
    return bufs is not None


def sm90_small_m_fp8_gemm_nt(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    *,
    recipe: Tuple[int, int, int] = (1, 32, 32),
    compiled_dims: str = "",
    fallback=None,
) -> bool:
    """Execute SM90 small-M G32 FP8 GEMM if supported, else call ``fallback``.

    Returns ``True`` if handled by the optimized SM90 kernel, ``False`` if
    delegated to ``fallback``.
    """
    x_q, x_s = lhs
    weight, weight_scale = rhs

    if is_enabled() and recipe == (1, 32, 32) and x_q.ndim == 2 and weight.ndim == 2 and out.ndim == 2:
        m, k = x_q.shape
        n, k_w = weight.shape
        cfg = select_config(m, n, k) if k == k_w else None
        sfa_box_m = (m + 3) & ~3
        k_scales = k >> 5
        dev = x_q.device
        dev_idx = dev.index if dev.index is not None else (
            torch.cuda.current_device() if dev.type == "cuda" else -1
        )
        if (
            cfg is not None
            and x_q.is_cuda
            and _is_sm90_device(dev_idx)
            and x_s.device == dev
            and weight.device == dev
            and weight_scale.device == dev
            and out.device == dev
            and x_q.dtype == torch.float8_e4m3fn
            and weight.dtype == torch.float8_e4m3fn
            and x_s.dtype == torch.float32
            and weight_scale.dtype == torch.float32
            and out.dtype == torch.bfloat16
            and x_q.is_contiguous()
            and weight.is_contiguous()
            and weight_scale.is_contiguous()
            and out.is_contiguous()
            and out.shape == (m, n)
            and x_s.shape == (m, k_scales)
            and weight_scale.shape == (n >> 5, k_scales)
            and x_s.stride(0) == 1
            and x_s.stride(1) >= sfa_box_m
            and (x_s.stride(1) & 3) == 0
            and (x_q.data_ptr() & 15) == 0
            and (weight.data_ptr() & 15) == 0
            and (x_s.data_ptr() & 15) == 0
            and (weight_scale.data_ptr() & 3) == 0
            and (out.data_ptr() & 15) == 0
        ):
            sfa_storage_elems = x_s.untyped_storage().nbytes() >> 2
            sfa_tail_base = x_s.storage_offset() + (k_scales - 1) * x_s.stride(1)
            sfa_global_m = (
                sfa_box_m
                if (sfa_tail_base + sfa_box_m <= sfa_storage_elems)
                else (m if (sfa_tail_base + m <= sfa_storage_elems) else 0)
            )
            mode, split_k, min_blocks, _ = cfg
            capturing = torch.cuda.is_current_stream_capturing()
            lib = (
                None
                if sfa_global_m == 0
                else (
                    _CACHED_LIB
                    if _CACHED_LIB is not None
                    else (None if capturing else _load_library())
                )
            )
            if lib is not None and _ensure_device_initialized(lib, dev_idx, capturing=capturing):
                stream_ptr = torch.cuda.current_stream(dev_idx).cuda_stream
                ws_ptr = 0
                counters_ptr = 0
                ready = split_k == 1
                if not ready:
                    bufs = _get_stream_buffers(dev_idx, stream_ptr, capturing=capturing)
                    if bufs is not None:
                        ws, counters = bufs
                        if n * m * split_k <= ws.numel() and (n >> 5) <= counters.numel():
                            ws_ptr = ws.data_ptr()
                            counters_ptr = counters.data_ptr()
                            ready = True
                if ready:
                    if torch.cuda.current_device() != dev_idx:
                        with torch.cuda.device(dev_idx):
                            rc = lib.run_sm90_small_m_fp8_gemm(
                                x_q.data_ptr(),
                                x_s.data_ptr(),
                                weight.data_ptr(),
                                weight_scale.data_ptr(),
                                out.data_ptr(),
                                ws_ptr,
                                counters_ptr,
                                m,
                                n,
                                k,
                                sfa_global_m,
                                x_s.stride(1),
                                mode,
                                split_k,
                                min_blocks,
                                stream_ptr,
                            )
                    else:
                        rc = lib.run_sm90_small_m_fp8_gemm(
                            x_q.data_ptr(),
                            x_s.data_ptr(),
                            weight.data_ptr(),
                            weight_scale.data_ptr(),
                            out.data_ptr(),
                            ws_ptr,
                            counters_ptr,
                            m,
                            n,
                            k,
                            sfa_global_m,
                            x_s.stride(1),
                            mode,
                            split_k,
                            min_blocks,
                            stream_ptr,
                        )
                    if rc == 0:
                        return True

    if fallback is None:
        import deep_gemm

        fallback = deep_gemm.fp8_gemm_nt
    fallback(lhs, rhs, out, recipe=recipe, compiled_dims=compiled_dims)
    return False


def attach_sm90_small_m_fp8_gemm(linear) -> bool:
    """Attach the SM90 small-M G32 FP8 GEMM fast path to a ``V41MXFP8Linear``.

    No-op on non-SM90 devices, meta tensors, or when disabled via
    ``DSV41_SM90_SMALL_M_FP8_GEMM=0``. Preserves ``linear._sm90_linear``,
    checkpoint weight storage, and DeepGEMM fallback for unsupported shapes.
    """
    if linear is None or not is_enabled():
        return False
    adapter = getattr(linear, "_sm90_linear", None)
    if adapter is None:
        return False
    weight = getattr(adapter, "weight", None)
    if (
        not isinstance(weight, torch.Tensor)
        or not weight.is_cuda
        or torch.cuda.get_device_capability(weight.device) != (9, 0)
    ):
        return False
    if getattr(adapter, "_sm90_small_m_attached", False):
        return True
    if not warmup_sm90_small_m_fp8_gemm(weight.device):
        return False
    orig_forward = adapter._forward

    def _dispatched_forward(lhs, rhs, out, *, recipe=(1, 32, 32), compiled_dims=""):
        sm90_small_m_fp8_gemm_nt(
            lhs,
            rhs,
            out,
            recipe=recipe,
            compiled_dims=compiled_dims,
            fallback=orig_forward,
        )

    adapter._forward = _dispatched_forward
    adapter._sm90_small_m_orig_forward = orig_forward
    adapter._sm90_small_m_attached = True
    return True
