// SPDX-License-Identifier: Apache-2.0
// Fused TP2 P2P AllReduce + 4-branch Hyper-Connection Residual Injection (+ Grouped RMSNorm + Fused FP8 Scale).
//
// Architecture:
//   1. Decode & medium batches (rows <= max_lamport_rows, default 1024):
//      - 3-slot Lamport one-way NVLink push (zero flag write, zero remote read RTT;
//        only peer payload is pushed over NVLink while local operand stays in registers).
//      - Non-spinning last-CTA slot advance via Hopper scoped atom.global.acq_rel.gpu.add.u32,
//        overlapped with one-way NVLink payload flight.
//      - MoE packed_dual (routed + shared) is pre-summed in registers via SM90 native
//        add.rn.bf16x2 BEFORE the NVLink push, cutting MoE NVLink payload in half.
//      - Branch-parallel CTA grid (4 * rows CTAs, 1 branch per CTA, up to 1 full resident wave:
//        78 rows / 312 CTAs on H20, 132 rows / 528 CTAs on H200) with 0 stack spill;
//        row-parallel Lamport grid for larger batches with __launch_bounds__(320, 3) (0 stack spill).
//      - Previous Lamport slot clearing in local L2 overlaps with one-way NVLink flight.
//   2. Large prefill batches (rows > max_lamport_rows):
//      - 2-slot one-way push + release/acquire flag (eliminates remote NVLink read RTT),
//        capped at <= min(kMaxSignalBlocks, max_resident_ctas) (234 CTAs on H20, 396 CTAs on H200)
//        for single-wave co-residency + full HBM3/HBM3e saturation, 0 stack spill, and pre-summed packed_dual.
//   3. Fused FP8 per-tensor scale (collect_amax=true):
//      - Accumulates BF16 absmax in registers via SM90 max.bf16x2 (2 SIMD ops per pair, NaN-safe),
//        reduces once per CTA (zero per-row barriers), and writes scale[0] = __fdiv_rn(amax, 448.0f)
//        directly from the last arriving CTA — eliminating the second _norm_fp8_scale kernel launch.
#include "rtp_llm/models_py/bindings/cuda/kernels/tp2_ar_hc_fused.h"
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace rtp_llm {
namespace tp2_hc_ipc {

constexpr int    kMaxBlocks            = 128;   // Alignment unit for tp2ArHcBufferSize & AllGather
constexpr int    kMaxSignalBlocks      = 512;   // Max CTAs for 2-slot Push+Flag prefill kernels (234 H20 / 396 H200)
constexpr int    kMaxSingleRows        = 12288;
constexpr int    kHidden               = 2560;
constexpr int    kBranches             = 4;
constexpr int    kThreads              = 320;   // 320 * 8 BF16 = 2560 columns
constexpr int    kWarps                = kThreads / 32;  // 10
constexpr int    kBranchPushChunk      = kThreads / kBranches;  // 80 uint4s per branch CTA
constexpr size_t kRowU4                = kHidden / 8;    // 320 uint4s
constexpr int    kLamportSlots         = 3;
constexpr int    kGatherThreads        = 256;

struct Tp2HcSignal {
    alignas(128) uint32_t cta_gen[kMaxSignalBlocks][32];
    alignas(128) uint32_t peer_ready[2][kMaxSignalBlocks][32];
    // Lamport 3-slot one-way push state (each on its own 128B cache line to avoid false sharing).
    alignas(128) uint32_t lamport_flag;
    alignas(128) uint32_t lamport_clear_u4;
    alignas(128) uint32_t lamport_counter;
    // Fused per-tensor FP8 scale reduction state (zeroed on init and by last CTA of each collect call).
    alignas(128) uint32_t amax_bits;
    alignas(128) uint32_t amax_done;
};

#define DINLINE __device__ __forceinline__

static DINLINE uint32_t atom_add_acq_rel_gpu(uint32_t* addr, uint32_t val) {
    uint32_t old;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("atom.global.acq_rel.gpu.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(addr), "r"(val) : "memory");
#else
    __threadfence();
    old = atomicAdd(addr, val);
    __threadfence();
#endif
    return old;
}

static DINLINE void atom_max_relaxed_gpu(uint32_t* addr, uint32_t val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    uint32_t old;
    asm volatile("atom.global.relaxed.gpu.max.u32 %0, [%1], %2;" : "=r"(old) : "l"(addr), "r"(val) : "memory");
    (void)old;
#else
    atomicMax(addr, val);
#endif
}

static DINLINE uint32_t atom_exch_relaxed_gpu(uint32_t* addr, uint32_t val) {
    uint32_t old;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("atom.global.relaxed.gpu.exch.b32 %0, [%1], %2;" : "=r"(old) : "l"(addr), "r"(val) : "memory");
#else
    old = atomicExch(addr, val);
#endif
    return old;
}

static DINLINE void st_flag_release(uint32_t* addr, uint32_t val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(val), "l"(addr) : "memory");
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(val), "l"(addr) : "memory");
#endif
}

static DINLINE uint32_t ld_flag_acquire(uint32_t* addr) {
    uint32_t val;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(val) : "l"(addr) : "memory");
#endif
    return val;
}

static DINLINE uint32_t ld_u32_volatile(const uint32_t* addr) {
    uint32_t val;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    return val;
}

static DINLINE void st_u32_volatile(uint32_t* addr, uint32_t val) {
    asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(val), "l"(addr) : "memory");
}

static DINLINE void st_u4_wt(uint4* addr, uint4 val) {
    asm volatile("st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};"
                 ::"l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                 : "memory");
}

static DINLINE void st_u4_volatile(uint4* addr, uint4 val) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};"
                 ::"l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                 : "memory");
}

static DINLINE uint4 ld_u4_volatile(const uint4* addr) {
    uint4 val;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                 : "l"(addr)
                 : "memory");
    return val;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 900
static DINLINE uint16_t f32_to_bf16_bits(float x) {
    uint16_t out;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(out) : "f"(x));
#else
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7fffffffU) > 0x7f800000U) {
        out = static_cast<uint16_t>((u >> 16) | 0x0040U);
    } else {
        out = static_cast<uint16_t>((u + (0x7fffU + ((u >> 16) & 1U))) >> 16);
    }
#endif
    return out;
}
#endif

static DINLINE uint32_t pack_bf16x2(float lo, float hi) {
    uint32_t out;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(lo), "f"(hi));
#else
    out = static_cast<uint32_t>(f32_to_bf16_bits(lo)) | (static_cast<uint32_t>(f32_to_bf16_bits(hi)) << 16);
#endif
    return out;
}

static DINLINE uint32_t add_bf16x2(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t out;
    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
#else
    float a0 = __uint_as_float(a << 16);
    float a1 = __uint_as_float(a & 0xffff0000U);
    float b0 = __uint_as_float(b << 16);
    float b1 = __uint_as_float(b & 0xffff0000U);
    return pack_bf16x2(__fadd_rn(a0, b0), __fadd_rn(a1, b1));
#endif
}

static DINLINE uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t out;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
#else
    float a0 = __uint_as_float(a << 16);
    float a1 = __uint_as_float(a & 0xffff0000U);
    float b0 = __uint_as_float(b << 16);
    float b1 = __uint_as_float(b & 0xffff0000U);
    return pack_bf16x2(__fmul_rn(a0, b0), __fmul_rn(a1, b1));
#endif
}

// Updates non-NaN non-negative BF16x2 running max with |val_pair| (NaN lanes are ignored).
static DINLINE uint32_t max_abs_bf16x2(uint32_t acc_abs_pair, uint32_t val_pair) {
    const uint32_t abs_pair = val_pair & 0x7fff7fffU;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t out;
    asm("max.bf16x2 %0, %1, %2;" : "=r"(out) : "r"(acc_abs_pair), "r"(abs_pair));
    return out;
#else
    float a0 = __uint_as_float(acc_abs_pair << 16);
    float a1 = __uint_as_float(acc_abs_pair & 0xffff0000U);
    float b0 = __uint_as_float(abs_pair << 16);
    float b1 = __uint_as_float(abs_pair & 0xffff0000U);
    b0       = (b0 == b0) ? b0 : 0.0f;
    b1       = (b1 == b1) ? b1 : 0.0f;
    return pack_bf16x2(fmaxf(a0, b0), fmaxf(a1, b1));
#endif
}

static DINLINE uint4 add_bf16x2_u4(uint4 a, uint4 b) {
    return make_uint4(
        add_bf16x2(a.x, b.x), add_bf16x2(a.y, b.y), add_bf16x2(a.z, b.z), add_bf16x2(a.w, b.w));
}

static DINLINE uint4 mul_bf16x2_u4_scalar(uint4 a, uint32_t scalar_pair) {
    return make_uint4(mul_bf16x2(a.x, scalar_pair),
                      mul_bf16x2(a.y, scalar_pair),
                      mul_bf16x2(a.z, scalar_pair),
                      mul_bf16x2(a.w, scalar_pair));
}

// Carry-free SWAR check for BF16 -0.0 (0x8000) in each 16-bit lane of w.
// Returns bit 15 = 1 iff low 16-bit lane == 0x8000, bit 31 = 1 iff high 16-bit lane == 0x8000.
static DINLINE uint32_t neg_zero_lane_mask(uint32_t w) {
    const uint32_t d  = w ^ 0x80008000U;
    const uint32_t nz = ((d & 0x7fff7fffU) + 0x7fff7fffU) | d;
    return (~nz) & 0x80008000U;
}

static DINLINE uint4 sanitize_neg_zero_u4(uint4 v) {
    return make_uint4(v.x ^ neg_zero_lane_mask(v.x),
                      v.y ^ neg_zero_lane_mask(v.y),
                      v.z ^ neg_zero_lane_mask(v.z),
                      v.w ^ neg_zero_lane_mask(v.w));
}

static DINLINE bool has_neg_zero_u4(uint4 v) {
    return (neg_zero_lane_mask(v.x) | neg_zero_lane_mask(v.y) | neg_zero_lane_mask(v.z)
            | neg_zero_lane_mask(v.w))
           != 0U;
}

static DINLINE uint32_t dup_bf16_bits(uint16_t b) {
    return static_cast<uint32_t>(b) | (static_cast<uint32_t>(b) << 16);
}

static DINLINE float sum_sq_u4(uint4 v) {
    float sq = 0.0f;
    float f0 = __uint_as_float(v.x << 16);
    float f1 = __uint_as_float(v.x & 0xffff0000U);
    sq       = __fmaf_rn(f0, f0, sq);
    sq       = __fmaf_rn(f1, f1, sq);
    f0       = __uint_as_float(v.y << 16);
    f1       = __uint_as_float(v.y & 0xffff0000U);
    sq       = __fmaf_rn(f0, f0, sq);
    sq       = __fmaf_rn(f1, f1, sq);
    f0       = __uint_as_float(v.z << 16);
    f1       = __uint_as_float(v.z & 0xffff0000U);
    sq       = __fmaf_rn(f0, f0, sq);
    sq       = __fmaf_rn(f1, f1, sq);
    f0       = __uint_as_float(v.w << 16);
    f1       = __uint_as_float(v.w & 0xffff0000U);
    sq       = __fmaf_rn(f0, f0, sq);
    sq       = __fmaf_rn(f1, f1, sq);
    return sq;
}

template<bool kCollectAmax>
static DINLINE uint32_t norm_pair_bf16x2(uint32_t r_pair, uint32_t g_pair, float inv_rms, uint32_t& amax_pair) {
    const float    r0     = __uint_as_float(r_pair << 16);
    const float    r1     = __uint_as_float(r_pair & 0xffff0000U);
    const float    g0     = __uint_as_float(g_pair << 16);
    const float    g1     = __uint_as_float(g_pair & 0xffff0000U);
    const float    n0     = (r0 * inv_rms) * (1.0f + g0);
    const float    n1     = (r1 * inv_rms) * (1.0f + g1);
    const uint32_t packed = pack_bf16x2(n0, n1);
    if constexpr (kCollectAmax) {
        amax_pair = max_abs_bf16x2(amax_pair, packed);
    }
    return packed;
}

template<bool kCollectAmax>
static DINLINE uint4 norm_u4_bf16(uint4 res_u4, uint4 gamma_u4, float inv_rms, uint32_t& amax_pair) {
    return make_uint4(norm_pair_bf16x2<kCollectAmax>(res_u4.x, gamma_u4.x, inv_rms, amax_pair),
                      norm_pair_bf16x2<kCollectAmax>(res_u4.y, gamma_u4.y, inv_rms, amax_pair),
                      norm_pair_bf16x2<kCollectAmax>(res_u4.z, gamma_u4.z, inv_rms, amax_pair),
                      norm_pair_bf16x2<kCollectAmax>(res_u4.w, gamma_u4.w, inv_rms, amax_pair));
}

static DINLINE float block_reduce_sum_10w(float val, float* sh_warp, int warp, int lane) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffU, val, offset);
    }
    if (lane == 0) {
        sh_warp[warp] = val;
    }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < kWarps) ? sh_warp[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffffU, v, offset);
        }
        if (lane == 0) {
            sh_warp[0] = v;
        }
    }
    __syncthreads();
    return sh_warp[0];
}

static DINLINE void finalize_grid_fp8_scale(uint32_t     amax_pair,
                                            float*       sh_amax,
                                            Tp2HcSignal* self_sig,
                                            float*       scale_out,
                                            int          warp,
                                            int          lane,
                                            int          tid) {
    float val = fmaxf(__uint_as_float(amax_pair << 16), __uint_as_float(amax_pair & 0xffff0000U));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffffU, val, offset));
    }
    if (lane == 0) {
        sh_amax[warp] = val;
    }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < kWarps) ? sh_amax[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffffU, v, offset));
        }
        if (tid == 0) {
            atom_max_relaxed_gpu(&self_sig->amax_bits, __float_as_uint(v));
            if (atom_add_acq_rel_gpu(&self_sig->amax_done, 1U) + 1U == static_cast<uint32_t>(gridDim.x)) {
                const uint32_t final_bits = atom_exch_relaxed_gpu(&self_sig->amax_bits, 0U);
                st_u32_volatile(&self_sig->amax_done, 0U);
                scale_out[0] = __fdiv_rn(__uint_as_float(final_bits), 448.0f);
            }
        }
    }
}

// Phase 1: Read Lamport slot metadata with zero fence/atomic on critical path before NVLink push.
static DINLINE void lamport_read_slot(Tp2HcSignal* self_sig,
                                      uint32_t&    sh_flag,
                                      uint32_t&    sh_clear_u4,
                                      int          tid) {
    if (tid == 0) {
        sh_flag     = ld_u32_volatile(&self_sig->lamport_flag);
        sh_clear_u4 = ld_u32_volatile(&self_sig->lamport_clear_u4);
    }
    __syncthreads();
}

// Phase 2: Advance Lamport slot from the last CTA (called during NVLink flight, hidden latency).
static DINLINE void lamport_advance_slot(Tp2HcSignal* self_sig,
                                         uint32_t     slot,
                                         size_t       total_u4,
                                         int          tid) {
    if (tid == 0) {
        if (atom_add_acq_rel_gpu(&self_sig->lamport_counter, 1U) + 1U == static_cast<uint32_t>(gridDim.x)) {
            st_u32_volatile(&self_sig->lamport_flag, (slot == 2U) ? 0U : (slot + 1U));
            st_u32_volatile(&self_sig->lamport_clear_u4, static_cast<uint32_t>(total_u4));
            st_u32_volatile(&self_sig->lamport_counter, 0U);
        }
    }
}

template<bool kEnabled>
static DINLINE void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (kEnabled) {
        cudaGridDependencySynchronize();
    }
#endif
}

template<bool kEnabled>
static DINLINE void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (kEnabled) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

__global__ void tp2ArHcInitLamportKernel(uint4* buf, size_t num_u4) {
    const uint4  neg_zero = make_uint4(0x80008000U, 0x80008000U, 0x80008000U, 0x80008000U);
    const size_t stride   = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < num_u4; i += stride) {
        buf[i] = neg_zero;
    }
}

// -----------------------------------------------------------------------------
// Lamport Kernel A: Branch-Parallel (grid = 4 * rows, up to 1 full resident wave)
// Each CTA handles 1 (row, branch), zero stack spill (REG <= 42, STACK = 0).
// The 4 branch CTAs of row r split the NVLink push (80 uint4s each = 10 cache lines)
// and share the peer data from local L2.
// -----------------------------------------------------------------------------
template<bool kPackedDual, bool kCollectAmax, bool kEnablePdl>
__global__ void __launch_bounds__(kThreads, 4)
tp2ArHcLamportBranchKernel(const uint4* __restrict__    hyper_x,
                           const uint4* __restrict__    local_sub,
                           const uint64_t* __restrict__ gates_packed,
                           const uint4* __restrict__    gamma,
                           uint4* __restrict__          residual_out,
                           uint4* __restrict__          norm_out,
                           float* __restrict__          scale_out,
                           uint4* __restrict__          self_lamp_base,
                           uint4* __restrict__          peer_lamp_base,
                           Tp2HcSignal* __restrict__    self_sig,
                           int                          rows,
                           float                        eps,
                           size_t                       slot_u4) {
    __shared__ uint32_t sh_flag;
    __shared__ uint32_t sh_clear_u4;
    __shared__ float    sh_warp[kWarps];
    __shared__ float    sh_amax[kWarps];

    const int    bid        = blockIdx.x;
    const int    tid        = threadIdx.x;
    const int    lane       = tid & 31;
    const int    warp       = tid >> 5;
    const int    r          = bid >> 2;
    const int    g          = bid & 3;
    const size_t total_u4   = static_cast<size_t>(rows) * kRowU4;
    const size_t branch_idx = (static_cast<size_t>(r) * kBranches + g) * kRowU4 + tid;

    // Preload inputs independent of the preceding sublayer GEMM during PDL window.
    const uint4    gamma_u4 = gamma[static_cast<size_t>(g) * kRowU4 + tid];
    const uint4    x_u4     = hyper_x[branch_idx];
    const uint16_t gate_u16 = static_cast<uint16_t>((gates_packed[r] >> (g * 16)) & 0xffffU);

    // Wait for preceding GEMM (and any prior back-to-back Lamport metadata/scale update).
    pdl_wait<kEnablePdl>();

    lamport_read_slot(self_sig, sh_flag, sh_clear_u4, tid);

    const uint32_t slot       = sh_flag;
    const uint32_t clear_slot = (slot == 0U) ? 2U : (slot - 1U);
    const size_t   clear_u4   = static_cast<size_t>(sh_clear_u4);

    const uint4* self_data  = self_lamp_base + static_cast<size_t>(slot) * slot_u4;
    uint4*       peer_data  = peer_lamp_base + static_cast<size_t>(slot) * slot_u4;
    uint4*       self_clear = self_lamp_base + static_cast<size_t>(clear_slot) * slot_u4;

    // Load local sublayer output (pre-summing routed + shared for MoE packed_dual).
    const size_t row_off   = static_cast<size_t>(r) * kRowU4 + tid;
    uint4        local_sum = local_sub[row_off];
    if constexpr (kPackedDual) {
        local_sum = add_bf16x2_u4(local_sum, local_sub[total_u4 + row_off]);
    }

    // Split 1-way NVLink push of row r across its 4 branch CTAs (80 uint4s each, contiguous).
    const int push_lo = g * kBranchPushChunk;
    if (tid >= push_lo && tid < push_lo + kBranchPushChunk) {
        st_u4_wt(peer_data + row_off, sanitize_neg_zero_u4(local_sum));
    }

    // Overlap slot advance and previous-slot L2 clear with one-way NVLink flight.
    lamport_advance_slot(self_sig, slot, total_u4, tid);

    // Overlap previous-slot L2 clear with one-way NVLink flight.
    if (clear_u4 > 0) {
        const uint4  neg_zero = make_uint4(0x80008000U, 0x80008000U, 0x80008000U, 0x80008000U);
        const size_t gstride  = static_cast<size_t>(gridDim.x) * kThreads;
        for (size_t i = static_cast<size_t>(bid) * kThreads + tid; i < clear_u4; i += gstride) {
            st_u4_wt(self_clear + i, neg_zero);
        }
    }

    // Poll local L2 until peer's one-way NVLink write arrives.
    uint4 pr;
    do {
        pr = ld_u4_volatile(self_data + row_off);
    } while (has_neg_zero_u4(pr));

    // Native SM90 bf16x2 AllReduce sum + HC residual injection.
    const uint4 sub_u4 = add_bf16x2_u4(local_sum, pr);
    const uint4 inj_u4 = mul_bf16x2_u4_scalar(sub_u4, dup_bf16_bits(gate_u16));
    const uint4 res_u4 = add_bf16x2_u4(x_u4, inj_u4);
    residual_out[branch_idx] = res_u4;

    // Single-branch RMSNorm + optional fused BF16x2 absmax.
    const float sum_sq  = block_reduce_sum_10w(sum_sq_u4(res_u4), sh_warp, warp, lane);
    const float inv_rms = rsqrtf(sum_sq * (1.0f / 2560.0f) + eps);

    uint32_t    amax_pair = 0U;
    const uint4 norm_u4   = norm_u4_bf16<kCollectAmax>(res_u4, gamma_u4, inv_rms, amax_pair);

    pdl_trigger<kEnablePdl>();
    norm_out[branch_idx] = norm_u4;

    if constexpr (kCollectAmax) {
        finalize_grid_fp8_scale(amax_pair, sh_amax, self_sig, scale_out, warp, lane, tid);
    }
}

// -----------------------------------------------------------------------------
// Lamport Kernel B: Row-Parallel (grid = min(rows, max_res_row), 1 row/iter)
// Uses __launch_bounds__(320, 3) -> up to 64 regs/thread -> 0 stack spill for res_u4[4],
// 234 resident CTAs on H20, and fused once-per-CTA FP8 scale reduction.
// -----------------------------------------------------------------------------
template<bool kPackedDual, bool kCollectAmax, bool kEnablePdl>
__global__ void __launch_bounds__(kThreads, 3)
tp2ArHcLamportRowKernel(const uint4* __restrict__    hyper_x,
                        const uint4* __restrict__    local_sub,
                        const uint64_t* __restrict__ gates_packed,
                        const uint4* __restrict__    gamma,
                        uint4* __restrict__          residual_out,
                        uint4* __restrict__          norm_out,
                        float* __restrict__          scale_out,
                        uint4* __restrict__          self_lamp_base,
                        uint4* __restrict__          peer_lamp_base,
                        Tp2HcSignal* __restrict__    self_sig,
                        int                          rows,
                        float                        eps,
                        size_t                       slot_u4) {
    __shared__ uint32_t sh_flag;
    __shared__ uint32_t sh_clear_u4;
    __shared__ float    sh_warp[kBranches][kWarps];
    __shared__ float    sh_inv_rms[kBranches];

    const int    bid         = blockIdx.x;
    const int    tid         = threadIdx.x;
    const int    lane        = tid & 31;
    const int    warp        = tid >> 5;
    const size_t total_u4    = static_cast<size_t>(rows) * kRowU4;
    const bool   single_iter = (rows <= gridDim.x);

    pdl_wait<kEnablePdl>();

    lamport_read_slot(self_sig, sh_flag, sh_clear_u4, tid);

    const uint32_t slot       = sh_flag;
    const uint32_t clear_slot = (slot == 0U) ? 2U : (slot - 1U);
    const size_t   clear_u4   = static_cast<size_t>(sh_clear_u4);

    const uint4* self_data  = self_lamp_base + static_cast<size_t>(slot) * slot_u4;
    uint4*       peer_data  = peer_lamp_base + static_cast<size_t>(slot) * slot_u4;
    uint4*       self_clear = self_lamp_base + static_cast<size_t>(clear_slot) * slot_u4;

    // Phase 1: Push all assigned rows over NVLink (pre-summing MoE packed_dual in registers).
    uint4 local_sum0{};
    for (int r = bid; r < rows; r += gridDim.x) {
        const size_t row_off = static_cast<size_t>(r) * kRowU4 + tid;
        uint4        val     = local_sub[row_off];
        if constexpr (kPackedDual) {
            val = add_bf16x2_u4(val, local_sub[total_u4 + row_off]);
        }
        st_u4_wt(peer_data + row_off, sanitize_neg_zero_u4(val));
        if (r == bid) {
            local_sum0 = val;
        }
    }

    // Phase 2: Advance slot and cooperatively clear previous Lamport slot in local L2 while NVLink write flies.
    lamport_advance_slot(self_sig, slot, total_u4, tid);
    if (clear_u4 > 0) {
        const uint4  neg_zero = make_uint4(0x80008000U, 0x80008000U, 0x80008000U, 0x80008000U);
        const size_t gstride  = static_cast<size_t>(gridDim.x) * kThreads;
        for (size_t i = static_cast<size_t>(bid) * kThreads + tid; i < clear_u4; i += gstride) {
            st_u4_wt(self_clear + i, neg_zero);
        }
    }

    // Phase 3: Poll local L2 once per row; only 2 __syncthreads() per row even with kCollectAmax.
    uint32_t running_amax_pair = 0U;
    for (int r = bid; r < rows; r += gridDim.x) {
        const bool   last_iter = (r + gridDim.x >= rows);
        const size_t row_off   = static_cast<size_t>(r) * kRowU4 + tid;

        uint4 local_sum;
        if (single_iter || r == bid) {
            local_sum = local_sum0;
        } else {
            local_sum = local_sub[row_off];
            if constexpr (kPackedDual) {
                local_sum = add_bf16x2_u4(local_sum, local_sub[total_u4 + row_off]);
            }
        }

        uint4 pr;
        do {
            pr = ld_u4_volatile(self_data + row_off);
        } while (has_neg_zero_u4(pr));

        const uint4    sub_u4   = add_bf16x2_u4(local_sum, pr);
        const uint64_t gate_cur = gates_packed[r];

        uint4 res_u4[kBranches];
#pragma unroll
        for (int g = 0; g < kBranches; ++g) {
            const size_t   out_idx  = (static_cast<size_t>(r) * kBranches + g) * kRowU4 + tid;
            const uint16_t gate_u16 = static_cast<uint16_t>((gate_cur >> (g * 16)) & 0xffffU);
            const uint4    x_u4     = hyper_x[out_idx];
            const uint4    inj_u4   = mul_bf16x2_u4_scalar(sub_u4, dup_bf16_bits(gate_u16));
            res_u4[g]               = add_bf16x2_u4(x_u4, inj_u4);
            residual_out[out_idx]   = res_u4[g];

            float sq = sum_sq_u4(res_u4[g]);
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sq += __shfl_xor_sync(0xffffffffU, sq, offset);
            }
            if (lane == 0) {
                sh_warp[g][warp] = sq;
            }
        }

        __syncthreads();
        if (warp < kBranches) {
            float v = (lane < kWarps) ? sh_warp[warp][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xffffffffU, v, offset);
            }
            if (lane == 0) {
                sh_inv_rms[warp] = rsqrtf(v * (1.0f / 2560.0f) + eps);
            }
        }
        __syncthreads();

#pragma unroll
        for (int g = 0; g < kBranches; ++g) {
            const size_t out_idx  = (static_cast<size_t>(r) * kBranches + g) * kRowU4 + tid;
            const uint4  gamma_u4 = gamma[static_cast<size_t>(g) * kRowU4 + tid];
            const uint4  norm_u4  = norm_u4_bf16<kCollectAmax>(res_u4[g], gamma_u4, sh_inv_rms[g], running_amax_pair);
            if (last_iter && g == kBranches - 1) {
                pdl_trigger<kEnablePdl>();
            }
            norm_out[out_idx] = norm_u4;
        }
    }

    if constexpr (kCollectAmax) {
        finalize_grid_fp8_scale(running_amax_pair, &sh_warp[0][0], self_sig, scale_out, warp, lane, tid);
    }
}

// -----------------------------------------------------------------------------
// Lamport InjectOnly (final decoder layer): Branch-Parallel & Row-Parallel
// -----------------------------------------------------------------------------
template<bool kPackedDual, bool kEnablePdl>
__global__ void __launch_bounds__(kThreads, 4)
tp2ArHcLamportInjectOnlyBranchKernel(const uint4* __restrict__    hyper_x,
                                     const uint4* __restrict__    local_sub,
                                     const uint64_t* __restrict__ gates_packed,
                                     uint4* __restrict__          residual_out,
                                     uint4* __restrict__          self_lamp_base,
                                     uint4* __restrict__          peer_lamp_base,
                                     Tp2HcSignal* __restrict__    self_sig,
                                     int                          rows,
                                     size_t                       slot_u4) {
    __shared__ uint32_t sh_flag;
    __shared__ uint32_t sh_clear_u4;

    const int    bid        = blockIdx.x;
    const int    tid        = threadIdx.x;
    const int    r          = bid >> 2;
    const int    g          = bid & 3;
    const size_t total_u4   = static_cast<size_t>(rows) * kRowU4;
    const size_t branch_idx = (static_cast<size_t>(r) * kBranches + g) * kRowU4 + tid;

    const uint4    x_u4     = hyper_x[branch_idx];
    const uint16_t gate_u16 = static_cast<uint16_t>((gates_packed[r] >> (g * 16)) & 0xffffU);

    pdl_wait<kEnablePdl>();

    lamport_read_slot(self_sig, sh_flag, sh_clear_u4, tid);

    const uint32_t slot       = sh_flag;
    const uint32_t clear_slot = (slot == 0U) ? 2U : (slot - 1U);
    const size_t   clear_u4   = static_cast<size_t>(sh_clear_u4);

    const uint4* self_data  = self_lamp_base + static_cast<size_t>(slot) * slot_u4;
    uint4*       peer_data  = peer_lamp_base + static_cast<size_t>(slot) * slot_u4;
    uint4*       self_clear = self_lamp_base + static_cast<size_t>(clear_slot) * slot_u4;

    const size_t row_off   = static_cast<size_t>(r) * kRowU4 + tid;
    uint4        local_sum = local_sub[row_off];
    if constexpr (kPackedDual) {
        local_sum = add_bf16x2_u4(local_sum, local_sub[total_u4 + row_off]);
    }

    const int push_lo = g * kBranchPushChunk;
    if (tid >= push_lo && tid < push_lo + kBranchPushChunk) {
        st_u4_wt(peer_data + row_off, sanitize_neg_zero_u4(local_sum));
    }

    lamport_advance_slot(self_sig, slot, total_u4, tid);

    if (clear_u4 > 0) {
        const uint4  neg_zero = make_uint4(0x80008000U, 0x80008000U, 0x80008000U, 0x80008000U);
        const size_t gstride  = static_cast<size_t>(gridDim.x) * kThreads;
        for (size_t i = static_cast<size_t>(bid) * kThreads + tid; i < clear_u4; i += gstride) {
            st_u4_wt(self_clear + i, neg_zero);
        }
    }

    uint4 pr;
    do {
        pr = ld_u4_volatile(self_data + row_off);
    } while (has_neg_zero_u4(pr));

    const uint4 sub_u4 = add_bf16x2_u4(local_sum, pr);
    const uint4 inj_u4 = mul_bf16x2_u4_scalar(sub_u4, dup_bf16_bits(gate_u16));
    const uint4 res_u4 = add_bf16x2_u4(x_u4, inj_u4);

    pdl_trigger<kEnablePdl>();
    residual_out[branch_idx] = res_u4;
}

template<bool kPackedDual, bool kEnablePdl>
__global__ void __launch_bounds__(kThreads, 4)
tp2ArHcLamportInjectOnlyRowKernel(const uint4* __restrict__    hyper_x,
                                  const uint4* __restrict__    local_sub,
                                  const uint64_t* __restrict__ gates_packed,
                                  uint4* __restrict__          residual_out,
                                  uint4* __restrict__          self_lamp_base,
                                  uint4* __restrict__          peer_lamp_base,
                                  Tp2HcSignal* __restrict__    self_sig,
                                  int                          rows,
                                  size_t                       slot_u4) {
    __shared__ uint32_t sh_flag;
    __shared__ uint32_t sh_clear_u4;

    const int    bid         = blockIdx.x;
    const int    tid         = threadIdx.x;
    const size_t total_u4    = static_cast<size_t>(rows) * kRowU4;
    const bool   single_iter = (rows <= gridDim.x);

    pdl_wait<kEnablePdl>();

    lamport_read_slot(self_sig, sh_flag, sh_clear_u4, tid);

    const uint32_t slot       = sh_flag;
    const uint32_t clear_slot = (slot == 0U) ? 2U : (slot - 1U);
    const size_t   clear_u4   = static_cast<size_t>(sh_clear_u4);

    const uint4* self_data  = self_lamp_base + static_cast<size_t>(slot) * slot_u4;
    uint4*       peer_data  = peer_lamp_base + static_cast<size_t>(slot) * slot_u4;
    uint4*       self_clear = self_lamp_base + static_cast<size_t>(clear_slot) * slot_u4;

    uint4 local_sum0{};
    for (int r = bid; r < rows; r += gridDim.x) {
        const size_t row_off = static_cast<size_t>(r) * kRowU4 + tid;
        uint4        val     = local_sub[row_off];
        if constexpr (kPackedDual) {
            val = add_bf16x2_u4(val, local_sub[total_u4 + row_off]);
        }
        st_u4_wt(peer_data + row_off, sanitize_neg_zero_u4(val));
        if (r == bid) {
            local_sum0 = val;
        }
    }

    lamport_advance_slot(self_sig, slot, total_u4, tid);

    if (clear_u4 > 0) {
        const uint4  neg_zero = make_uint4(0x80008000U, 0x80008000U, 0x80008000U, 0x80008000U);
        const size_t gstride  = static_cast<size_t>(gridDim.x) * kThreads;
        for (size_t i = static_cast<size_t>(bid) * kThreads + tid; i < clear_u4; i += gstride) {
            st_u4_wt(self_clear + i, neg_zero);
        }
    }

    for (int r = bid; r < rows; r += gridDim.x) {
        const bool   last_iter = (r + gridDim.x >= rows);
        const size_t row_off   = static_cast<size_t>(r) * kRowU4 + tid;

        uint4 local_sum;
        if (single_iter || r == bid) {
            local_sum = local_sum0;
        } else {
            local_sum = local_sub[row_off];
            if constexpr (kPackedDual) {
                local_sum = add_bf16x2_u4(local_sum, local_sub[total_u4 + row_off]);
            }
        }

        uint4 pr;
        do {
            pr = ld_u4_volatile(self_data + row_off);
        } while (has_neg_zero_u4(pr));

        const uint4    sub_u4   = add_bf16x2_u4(local_sum, pr);
        const uint64_t gate_cur = gates_packed[r];

        if (last_iter) {
            pdl_trigger<kEnablePdl>();
        }
#pragma unroll
        for (int g = 0; g < kBranches; ++g) {
            const size_t   out_idx  = (static_cast<size_t>(r) * kBranches + g) * kRowU4 + tid;
            const uint16_t gate_u16 = static_cast<uint16_t>((gate_cur >> (g * 16)) & 0xffffU);
            const uint4    x_u4     = hyper_x[out_idx];
            const uint4    inj_u4   = mul_bf16x2_u4_scalar(sub_u4, dup_bf16_bits(gate_u16));
            residual_out[out_idx]   = add_bf16x2_u4(x_u4, inj_u4);
        }
    }
}

// -----------------------------------------------------------------------------
// Large-Prefill / Fallback Kernels (2-slot One-Way Push + Release/Acquire)
// Used for rows > max_lamport_rows (e.g. 1025..12288) or when lamport_buffers is empty.
// Launches up to min(kMaxSignalBlocks, max_resident_ctas) (234 CTAs = 3 CTAs/SM on H20)
// with __launch_bounds__(320, 3) -> 0 stack spill, full HBM3 saturation, and fused FP8 scale.
// -----------------------------------------------------------------------------
template<bool kPackedDual, bool kCollectAmax, bool kEnablePdl>
__global__ void __launch_bounds__(kThreads, 3)
tp2ArHcInjectNormKernel(const uint4* __restrict__    hyper_x,
                        const uint4* __restrict__    local_sub,
                        const uint64_t* __restrict__ gates_packed,
                        const uint4* __restrict__    gamma,
                        uint4* __restrict__          residual_out,
                        uint4* __restrict__          norm_out,
                        float* __restrict__          scale_out,
                        const uint4* __restrict__    self_buf_base,
                        uint4* __restrict__          peer_buf_base,
                        Tp2HcSignal* __restrict__    self_sig,
                        Tp2HcSignal* __restrict__    peer_sig,
                        int                          rows,
                        float                        eps,
                        size_t                       slot_u4) {
    __shared__ uint32_t sh_gen;
    __shared__ float    sh_warp[kBranches][kWarps];
    __shared__ float    sh_inv_rms[kBranches];

    const int    bid         = blockIdx.x;
    const int    tid         = threadIdx.x;
    const int    lane        = tid & 31;
    const int    warp        = tid >> 5;
    const size_t total_u4    = static_cast<size_t>(rows) * kRowU4;
    const bool   single_iter = (rows <= gridDim.x);

    if (tid == 0) {
        sh_gen = (self_sig->cta_gen[bid][0] += 1U);
    }

    __syncthreads();
    const uint32_t gen       = sh_gen;
    const size_t   slot_base = static_cast<size_t>(gen & 1U) * slot_u4;

    const uint4* self_buf = self_buf_base + slot_base;
    uint4*       peer_buf = peer_buf_base + slot_base;

    pdl_wait<kEnablePdl>();

    // One-way push pre-summed local data directly to peer_buf (eliminates remote read RTT).
    uint4 local_sum0{};
    for (int r = bid; r < rows; r += gridDim.x) {
        const size_t row_off = static_cast<size_t>(r) * kRowU4 + tid;
        uint4        val     = local_sub[row_off];
        if constexpr (kPackedDual) {
            val = add_bf16x2_u4(val, local_sub[total_u4 + row_off]);
        }
        st_u4_wt(peer_buf + row_off, val);
        if (r == bid) {
            local_sum0 = val;
        }
    }

    __syncthreads();
    if (tid == 0) {
        const int slot = static_cast<int>(gen & 1U);
        st_flag_release(&peer_sig->peer_ready[slot][bid][0], gen);
        while (ld_flag_acquire(&self_sig->peer_ready[slot][bid][0]) != gen) {}
    }
    __syncthreads();

    uint32_t running_amax_pair = 0U;
    for (int r = bid; r < rows; r += gridDim.x) {
        const bool   last_iter = (r + gridDim.x >= rows);
        const size_t row_off   = static_cast<size_t>(r) * kRowU4 + tid;

        uint4 local_sum;
        if (single_iter || r == bid) {
            local_sum = local_sum0;
        } else {
            local_sum = local_sub[row_off];
            if constexpr (kPackedDual) {
                local_sum = add_bf16x2_u4(local_sum, local_sub[total_u4 + row_off]);
            }
        }
        const uint4    pr       = ld_u4_volatile(self_buf + row_off);
        const uint4    sub_u4   = add_bf16x2_u4(local_sum, pr);
        const uint64_t gate_cur = gates_packed[r];

        uint4 res_u4[kBranches];
#pragma unroll
        for (int g = 0; g < kBranches; ++g) {
            const size_t   out_idx  = (static_cast<size_t>(r) * kBranches + g) * kRowU4 + tid;
            const uint16_t gate_u16 = static_cast<uint16_t>((gate_cur >> (g * 16)) & 0xffffU);
            const uint4    x_u4     = hyper_x[out_idx];
            const uint4    inj_u4   = mul_bf16x2_u4_scalar(sub_u4, dup_bf16_bits(gate_u16));
            res_u4[g]               = add_bf16x2_u4(x_u4, inj_u4);
            residual_out[out_idx]   = res_u4[g];

            float sq = sum_sq_u4(res_u4[g]);
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sq += __shfl_xor_sync(0xffffffffU, sq, offset);
            }
            if (lane == 0) {
                sh_warp[g][warp] = sq;
            }
        }

        __syncthreads();
        if (warp < kBranches) {
            float v = (lane < kWarps) ? sh_warp[warp][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xffffffffU, v, offset);
            }
            if (lane == 0) {
                sh_inv_rms[warp] = rsqrtf(v * (1.0f / 2560.0f) + eps);
            }
        }
        __syncthreads();

#pragma unroll
        for (int g = 0; g < kBranches; ++g) {
            const size_t out_idx  = (static_cast<size_t>(r) * kBranches + g) * kRowU4 + tid;
            const uint4  gamma_u4 = gamma[static_cast<size_t>(g) * kRowU4 + tid];
            const uint4  norm_u4  = norm_u4_bf16<kCollectAmax>(res_u4[g], gamma_u4, sh_inv_rms[g], running_amax_pair);
            if (last_iter && g == kBranches - 1) {
                pdl_trigger<kEnablePdl>();
            }
            norm_out[out_idx] = norm_u4;
        }
    }

    if constexpr (kCollectAmax) {
        finalize_grid_fp8_scale(running_amax_pair, &sh_warp[0][0], self_sig, scale_out, warp, lane, tid);
    }
}

template<bool kPackedDual, bool kEnablePdl>
__global__ void __launch_bounds__(kThreads, 4)
tp2ArHcInjectOnlyKernel(const uint4* __restrict__    hyper_x,
                        const uint4* __restrict__    local_sub,
                        const uint64_t* __restrict__ gates_packed,
                        uint4* __restrict__          residual_out,
                        const uint4* __restrict__    self_buf_base,
                        uint4* __restrict__          peer_buf_base,
                        Tp2HcSignal* __restrict__    self_sig,
                        Tp2HcSignal* __restrict__    peer_sig,
                        int                          rows,
                        size_t                       slot_u4) {
    __shared__ uint32_t sh_gen;

    const int    bid         = blockIdx.x;
    const int    tid         = threadIdx.x;
    const size_t total_u4    = static_cast<size_t>(rows) * kRowU4;
    const bool   single_iter = (rows <= gridDim.x);

    if (tid == 0) {
        sh_gen = (self_sig->cta_gen[bid][0] += 1U);
    }

    __syncthreads();
    const uint32_t gen       = sh_gen;
    const size_t   slot_base = static_cast<size_t>(gen & 1U) * slot_u4;

    const uint4* self_buf = self_buf_base + slot_base;
    uint4*       peer_buf = peer_buf_base + slot_base;

    pdl_wait<kEnablePdl>();

    uint4 local_sum0{};
    for (int r = bid; r < rows; r += gridDim.x) {
        const size_t row_off = static_cast<size_t>(r) * kRowU4 + tid;
        uint4        val     = local_sub[row_off];
        if constexpr (kPackedDual) {
            val = add_bf16x2_u4(val, local_sub[total_u4 + row_off]);
        }
        st_u4_wt(peer_buf + row_off, val);
        if (r == bid) {
            local_sum0 = val;
        }
    }

    __syncthreads();
    if (tid == 0) {
        const int slot = static_cast<int>(gen & 1U);
        st_flag_release(&peer_sig->peer_ready[slot][bid][0], gen);
        while (ld_flag_acquire(&self_sig->peer_ready[slot][bid][0]) != gen) {}
    }
    __syncthreads();

    for (int r = bid; r < rows; r += gridDim.x) {
        const bool   last_iter = (r + gridDim.x >= rows);
        const size_t row_off   = static_cast<size_t>(r) * kRowU4 + tid;

        uint4 local_sum;
        if (single_iter || r == bid) {
            local_sum = local_sum0;
        } else {
            local_sum = local_sub[row_off];
            if constexpr (kPackedDual) {
                local_sum = add_bf16x2_u4(local_sum, local_sub[total_u4 + row_off]);
            }
        }
        const uint4    pr       = ld_u4_volatile(self_buf + row_off);
        const uint4    sub_u4   = add_bf16x2_u4(local_sum, pr);
        const uint64_t gate_cur = gates_packed[r];

        if (last_iter) {
            pdl_trigger<kEnablePdl>();
        }
#pragma unroll
        for (int g = 0; g < kBranches; ++g) {
            const size_t   out_idx  = (static_cast<size_t>(r) * kBranches + g) * kRowU4 + tid;
            const uint16_t gate_u16 = static_cast<uint16_t>((gate_cur >> (g * 16)) & 0xffffU);
            const uint4    x_u4     = hyper_x[out_idx];
            const uint4    inj_u4   = mul_bf16x2_u4_scalar(sub_u4, dup_bf16_bits(gate_u16));
            residual_out[out_idx]   = add_bf16x2_u4(x_u4, inj_u4);
        }
    }
}

template<typename KernelFunc, typename... Args>
static void launch_maybe_pdl(KernelFunc kernel, dim3 grid, dim3 block, cudaStream_t stream, bool pdl, Args... args) {
    if (pdl) {
        cudaLaunchAttribute attr[1];
        attr[0].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim          = grid;
        cfg.blockDim         = block;
        cfg.dynamicSmemBytes = 0;
        cfg.stream           = stream;
        cfg.attrs            = attr;
        cfg.numAttrs         = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
    } else {
        kernel<<<grid, block, 0, stream>>>(args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

static int get_device_sm_count(int dev) {
    static int cached_dev      = -1;
    static int cached_sm_count = 0;
    if (cached_dev == dev && cached_sm_count > 0) {
        return cached_sm_count;
    }
    int sm_count = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
    cached_dev      = dev;
    cached_sm_count = sm_count;
    return sm_count;
}

template<typename KernelFunc>
static int get_max_resident_ctas(KernelFunc kernel, int dev) {
    const int sm_count = get_device_sm_count(dev);
    int       active   = 1;
    cudaError_t err    = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, kThreads, 0);
    if (err != cudaSuccess || active < 1) {
        active = 1;
    }
    return sm_count * active;
}

template<bool kEnablePdl>
__global__ void __launch_bounds__(kGatherThreads, 4)
tp2P2pAllGatherKernel(const uint4* __restrict__ shard,
                      uint4* __restrict__       out,
                      uint4* __restrict__       self_buf,
                      const uint4* __restrict__ peer_buf,
                      Tp2HcSignal* __restrict__ self_sig,
                      Tp2HcSignal* __restrict__ peer_sig,
                      int                       rank,
                      size_t                    num_u4,
                      size_t                    cta_cap_u4) {
    __shared__ uint32_t sh_gen;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid == 0) {
        sh_gen = (self_sig->cta_gen[bid][0] += 1U);
    }
    __syncthreads();

    const uint32_t gen      = sh_gen;
    const int      slot     = static_cast<int>(gen & 1U);
    const size_t   cta_base = (static_cast<size_t>(slot) * kMaxBlocks + bid) * cta_cap_u4;

    const size_t self_out_base = (rank == 0) ? 0 : num_u4;
    const size_t peer_out_base = (rank == 0) ? num_u4 : 0;
    const size_t stride        = static_cast<size_t>(gridDim.x) * kGatherThreads;

    pdl_wait<kEnablePdl>();

    size_t iter = 0;
    for (size_t idx = static_cast<size_t>(bid) * kGatherThreads + tid; idx < num_u4; idx += stride, ++iter) {
        const uint4 v = shard[idx];
        st_u4_volatile(self_buf + cta_base + iter * kGatherThreads + tid, v);
        out[self_out_base + idx] = v;
    }

    __syncthreads();
    if (tid == 0) {
        st_flag_release(&peer_sig->peer_ready[slot][bid][0], gen);
        while (ld_flag_acquire(&self_sig->peer_ready[slot][bid][0]) != gen) {}
    }
    __syncthreads();

    iter = 0;
    for (size_t idx = static_cast<size_t>(bid) * kGatherThreads + tid; idx < num_u4; idx += stride, ++iter) {
        out[peer_out_base + idx] = ld_u4_volatile(peer_buf + cta_base + iter * kGatherThreads + tid);
    }

    pdl_trigger<kEnablePdl>();
}

static size_t resolve_cta_cap_u4(int64_t buffer_bytes, size_t* out_slot_bytes) {
    const int64_t total_bytes = (buffer_bytes > 0) ? buffer_bytes : tp2ArHcBufferSize(kMaxSingleRows);
    constexpr int64_t kUnitBytes = static_cast<int64_t>(2) * kMaxBlocks * sizeof(uint4);
    TORCH_CHECK(total_bytes >= kUnitBytes && total_bytes % kUnitBytes == 0,
                "TP2 IPC buffer_bytes must be a positive multiple of 2 * 128 * 16");
    const size_t slot_bytes = static_cast<size_t>(total_bytes / 2);
    if (out_slot_bytes) {
        *out_slot_bytes = slot_bytes;
    }
    return (slot_bytes / sizeof(uint4)) / kMaxBlocks;
}

static size_t resolve_lamport_slot_u4(int64_t lamport_buffer_bytes) {
    constexpr int64_t kUnitBytes = static_cast<int64_t>(kLamportSlots) * static_cast<int64_t>(kRowU4 * sizeof(uint4));
    if (lamport_buffer_bytes <= 0 || lamport_buffer_bytes % kUnitBytes != 0) {
        return 0;
    }
    return static_cast<size_t>((lamport_buffer_bytes / kLamportSlots) / sizeof(uint4));
}

#undef DINLINE
}  // namespace tp2_hc_ipc

int64_t tp2ArHcMetaSize() {
    return sizeof(tp2_hc_ipc::Tp2HcSignal);
}

int64_t tp2ArHcBufferSize(int64_t max_single_rows) {
    TORCH_CHECK(max_single_rows >= tp2_hc_ipc::kMaxBlocks && max_single_rows % tp2_hc_ipc::kMaxBlocks == 0,
                "max_single_rows must be a positive multiple of 128");
    return static_cast<int64_t>(2) * max_single_rows * static_cast<int64_t>(tp2_hc_ipc::kRowU4 * sizeof(uint4));
}

int64_t tp2ArHcLamportBufferSize(int64_t max_lamport_rows) {
    TORCH_CHECK(max_lamport_rows >= 1, "max_lamport_rows must be positive");
    return static_cast<int64_t>(tp2_hc_ipc::kLamportSlots) * max_lamport_rows
           * static_cast<int64_t>(tp2_hc_ipc::kRowU4 * sizeof(uint4));
}

void tp2ArHcInitLamportBuffer(int64_t buffer_ptr, int64_t buffer_bytes) {
    TORCH_CHECK(buffer_ptr != 0 && buffer_bytes > 0 && buffer_bytes % static_cast<int64_t>(sizeof(uint4)) == 0,
                "Invalid Lamport buffer pointer or byte size");
    const size_t num_u4 = static_cast<size_t>(buffer_bytes) / sizeof(uint4);
    auto         stream = c10::cuda::getCurrentCUDAStream().stream();
    const dim3   block(256);
    const dim3   grid(static_cast<unsigned>(std::min<size_t>((num_u4 + 255) / 256, 256)));
    tp2_hc_ipc::tp2ArHcInitLamportKernel<<<grid, block, 0, stream>>>(reinterpret_cast<uint4*>(buffer_ptr), num_u4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t tp2P2pGatherBufferSize(int64_t max_single_rows) {
    return tp2ArHcBufferSize(max_single_rows);
}

torch::Tensor tp2P2pAllGather(const torch::Tensor&        shard,
                              const std::vector<int64_t>& buffers,
                              const std::vector<int64_t>& signals,
                              int64_t                     rank,
                              bool                        enable_pdl,
                              int64_t                     buffer_bytes) {
    TORCH_CHECK(shard.is_cuda() && shard.is_contiguous() && shard.dim() >= 1,
                "tp2P2pAllGather shard must be a contiguous CUDA tensor");
    size_t       slot_bytes  = 0;
    const size_t cta_cap_u4  = tp2_hc_ipc::resolve_cta_cap_u4(buffer_bytes, &slot_bytes);
    const size_t shard_bytes = static_cast<size_t>(shard.numel()) * static_cast<size_t>(shard.element_size());
    TORCH_CHECK(shard_bytes > 0 && shard_bytes <= slot_bytes && shard_bytes % sizeof(uint4) == 0,
                "tp2P2pAllGather shard bytes must be a non-zero multiple of 16 within IPC slot capacity");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(shard.data_ptr()) % sizeof(uint4) == 0,
                "tp2P2pAllGather shard must be 16-byte aligned");
    TORCH_CHECK(rank >= 0 && rank < 2 && buffers.size() == 2 && signals.size() == 2,
                "tp2P2pAllGather requires 2 rank pointers");

    c10::cuda::CUDAGuard guard(shard.device());
    auto                 out_sizes = shard.sizes().vec();
    out_sizes[0] *= 2;
    auto out = torch::empty(out_sizes, shard.options());

    auto*       self_buf = reinterpret_cast<uint4*>(buffers[rank]);
    const auto* peer_buf = reinterpret_cast<const uint4*>(buffers[1 - rank]);
    auto*       self_sig = reinterpret_cast<tp2_hc_ipc::Tp2HcSignal*>(signals[rank]);
    auto*       peer_sig = reinterpret_cast<tp2_hc_ipc::Tp2HcSignal*>(signals[1 - rank]);

    const size_t num_u4   = shard_bytes / sizeof(uint4);
    const int    sm_count = tp2_hc_ipc::get_device_sm_count(shard.get_device());
    const int    max_ctas = std::min(sm_count, tp2_hc_ipc::kMaxBlocks);
    const size_t need_blk = (num_u4 + tp2_hc_ipc::kGatherThreads - 1) / tp2_hc_ipc::kGatherThreads;
    const dim3   grid(static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(need_blk, max_ctas))));
    const dim3   block(tp2_hc_ipc::kGatherThreads);
    auto         stream = c10::cuda::getCurrentCUDAStream(shard.get_device()).stream();

    const auto* s_ptr = reinterpret_cast<const uint4*>(shard.data_ptr());
    auto*       o_ptr = reinterpret_cast<uint4*>(out.data_ptr());
    const int   r_i32 = static_cast<int>(rank);

    if (enable_pdl) {
        tp2_hc_ipc::launch_maybe_pdl(tp2_hc_ipc::tp2P2pAllGatherKernel<true>,
                                     grid, block, stream, true,
                                     s_ptr, o_ptr, self_buf, peer_buf, self_sig, peer_sig, r_i32, num_u4, cta_cap_u4);
    } else {
        tp2_hc_ipc::launch_maybe_pdl(tp2_hc_ipc::tp2P2pAllGatherKernel<false>,
                                     grid, block, stream, false,
                                     s_ptr, o_ptr, self_buf, peer_buf, self_sig, peer_sig, r_i32, num_u4, cta_cap_u4);
    }

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>>
tp2ArHcInjectNorm(const torch::Tensor&        hyper_x,
                  const torch::Tensor&        local_sub,
                  const torch::Tensor&        gates,
                  const torch::Tensor&        gamma,
                  const std::vector<int64_t>& buffers,
                  const std::vector<int64_t>& signals,
                  int64_t                     rank,
                  double                      eps,
                  bool                        packed_dual,
                  bool                        collect_amax,
                  bool                        enable_pdl,
                  int64_t                     buffer_bytes,
                  const std::vector<int64_t>& lamport_buffers,
                  int64_t                     lamport_buffer_bytes) {
    TORCH_CHECK(hyper_x.is_cuda() && local_sub.is_cuda() && gates.is_cuda() && gamma.is_cuda(),
                "tp2ArHcInjectNorm inputs must be CUDA tensors");
    TORCH_CHECK(hyper_x.scalar_type() == torch::kBFloat16 && local_sub.scalar_type() == torch::kBFloat16
                    && gates.scalar_type() == torch::kBFloat16 && gamma.scalar_type() == torch::kBFloat16,
                "tp2ArHcInjectNorm inputs must be BF16");
    TORCH_CHECK(hyper_x.is_contiguous() && local_sub.is_contiguous() && gates.is_contiguous() && gamma.is_contiguous(),
                "tp2ArHcInjectNorm inputs must be contiguous");
    size_t slot_bytes = 0;
    (void)tp2_hc_ipc::resolve_cta_cap_u4(buffer_bytes, &slot_bytes);
    const int64_t max_rows_ipc      = static_cast<int64_t>(slot_bytes / (tp2_hc_ipc::kRowU4 * sizeof(uint4)));
    const int64_t rows              = hyper_x.numel() / (tp2_hc_ipc::kBranches * tp2_hc_ipc::kHidden);
    const int64_t expected_sub_rows = packed_dual ? (2 * rows) : rows;
    TORCH_CHECK(rows >= 1 && rows <= max_rows_ipc
                    && hyper_x.size(-1) == tp2_hc_ipc::kBranches * tp2_hc_ipc::kHidden,
                "tp2ArHcInjectNorm rows exceed IPC slot capacity");
    TORCH_CHECK(local_sub.dim() == 2 && local_sub.size(0) == expected_sub_rows
                    && local_sub.size(1) == tp2_hc_ipc::kHidden,
                "tp2ArHcInjectNorm local_sub shape mismatch");
    TORCH_CHECK(gates.numel() == rows * tp2_hc_ipc::kBranches && gates.size(-1) == tp2_hc_ipc::kBranches,
                "tp2ArHcInjectNorm gates shape mismatch");
    TORCH_CHECK(gamma.dim() == 1 && gamma.numel() == tp2_hc_ipc::kBranches * tp2_hc_ipc::kHidden,
                "tp2ArHcInjectNorm gamma shape mismatch");
    TORCH_CHECK(rank >= 0 && rank < 2 && buffers.size() == 2 && signals.size() == 2,
                "tp2ArHcInjectNorm requires 2 rank pointers");

    c10::cuda::CUDAGuard guard(hyper_x.device());
    const int            dev      = hyper_x.get_device();
    auto                 residual = torch::empty_like(hyper_x);
    auto                 normed   = torch::empty_like(hyper_x);
    c10::optional<torch::Tensor> scale_opt;
    float*                       scale_ptr = nullptr;
    if (collect_amax) {
        auto scale = torch::empty({1}, torch::TensorOptions().dtype(torch::kFloat32).device(hyper_x.device()));
        scale_ptr  = scale.data_ptr<float>();
        scale_opt  = std::move(scale);
    }

    auto*       self_sig = reinterpret_cast<tp2_hc_ipc::Tp2HcSignal*>(signals[rank]);
    auto*       peer_sig = reinterpret_cast<tp2_hc_ipc::Tp2HcSignal*>(signals[1 - rank]);
    const auto* self_buf = reinterpret_cast<const uint4*>(buffers[rank]);
    auto*       peer_buf = reinterpret_cast<uint4*>(buffers[1 - rank]);

    const size_t  lamport_slot_u4  = tp2_hc_ipc::resolve_lamport_slot_u4(lamport_buffer_bytes);
    const int64_t max_lamport_rows = static_cast<int64_t>(lamport_slot_u4 / tp2_hc_ipc::kRowU4);
    const bool    use_lamport      = (lamport_buffers.size() == 2 && lamport_slot_u4 > 0 && rows <= max_lamport_rows);

    const dim3  block(tp2_hc_ipc::kThreads);
    auto        stream = c10::cuda::getCurrentCUDAStream(dev).stream();
    const auto* x_ptr  = reinterpret_cast<const uint4*>(hyper_x.data_ptr());
    const auto* s_ptr  = reinterpret_cast<const uint4*>(local_sub.data_ptr());
    const auto* g_ptr  = reinterpret_cast<const uint64_t*>(gates.data_ptr());
    const auto* w_ptr  = reinterpret_cast<const uint4*>(gamma.data_ptr());
    auto*       r_ptr  = reinterpret_cast<uint4*>(residual.data_ptr());
    auto*       n_ptr  = reinterpret_cast<uint4*>(normed.data_ptr());
    const int   n_i32  = static_cast<int>(rows);
    const float eps_f  = static_cast<float>(eps);

    if (use_lamport) {
        auto* self_lamp = reinterpret_cast<uint4*>(lamport_buffers[rank]);
        auto* peer_lamp = reinterpret_cast<uint4*>(lamport_buffers[1 - rank]);

#define DISPATCH_LAMPORT(PACKED, COLLECT, PDL)                                                                         \
    do {                                                                                                               \
        static const int max_res_branch =                                                                              \
            tp2_hc_ipc::get_max_resident_ctas(tp2_hc_ipc::tp2ArHcLamportBranchKernel<PACKED, COLLECT, PDL>, dev);      \
        if ((rows * tp2_hc_ipc::kBranches) <= max_res_branch) {                                                        \
            const dim3 grid(static_cast<unsigned>(rows * tp2_hc_ipc::kBranches));                                      \
            tp2_hc_ipc::launch_maybe_pdl(tp2_hc_ipc::tp2ArHcLamportBranchKernel<PACKED, COLLECT, PDL>,                 \
                                         grid, block, stream, PDL,                                                     \
                                         x_ptr, s_ptr, g_ptr, w_ptr, r_ptr, n_ptr, scale_ptr,                          \
                                         self_lamp, peer_lamp, self_sig, n_i32, eps_f, lamport_slot_u4);               \
        } else {                                                                                                       \
            static const int max_res_row =                                                                             \
                tp2_hc_ipc::get_max_resident_ctas(tp2_hc_ipc::tp2ArHcLamportRowKernel<PACKED, COLLECT, PDL>, dev);     \
            const dim3 grid(static_cast<unsigned>(std::max<int64_t>(1, std::min<int64_t>(rows, max_res_row))));        \
            tp2_hc_ipc::launch_maybe_pdl(tp2_hc_ipc::tp2ArHcLamportRowKernel<PACKED, COLLECT, PDL>,                    \
                                         grid, block, stream, PDL,                                                     \
                                         x_ptr, s_ptr, g_ptr, w_ptr, r_ptr, n_ptr, scale_ptr,                          \
                                         self_lamp, peer_lamp, self_sig, n_i32, eps_f, lamport_slot_u4);               \
        }                                                                                                              \
    } while (0)

        if (packed_dual) {
            if (collect_amax) {
                if (enable_pdl) DISPATCH_LAMPORT(true, true, true);
                else            DISPATCH_LAMPORT(true, true, false);
            } else {
                if (enable_pdl) DISPATCH_LAMPORT(true, false, true);
                else            DISPATCH_LAMPORT(true, false, false);
            }
        } else {
            if (collect_amax) {
                if (enable_pdl) DISPATCH_LAMPORT(false, true, true);
                else            DISPATCH_LAMPORT(false, true, false);
            } else {
                if (enable_pdl) DISPATCH_LAMPORT(false, false, true);
                else            DISPATCH_LAMPORT(false, false, false);
            }
        }
#undef DISPATCH_LAMPORT
        return {std::move(residual), std::move(normed), std::move(scale_opt)};
    }

    // Large-prefill / fallback 2-slot one-way push + release/acquire path.
    // Launches up to min(kMaxSignalBlocks, max_resident_ctas) (234 CTAs = 3 CTAs/SM on H20).
    const size_t slot_u4 = slot_bytes / sizeof(uint4);

#define DISPATCH_NORM(PACKED, COLLECT, PDL)                                                                            \
    do {                                                                                                               \
        static const int max_res_norm =                                                                                \
            tp2_hc_ipc::get_max_resident_ctas(tp2_hc_ipc::tp2ArHcInjectNormKernel<PACKED, COLLECT, PDL>, dev);         \
        const int  cap  = std::min(tp2_hc_ipc::kMaxSignalBlocks, max_res_norm);                                        \
        const dim3 grid(static_cast<unsigned>(std::max<int64_t>(1, std::min<int64_t>(rows, cap))));                    \
        tp2_hc_ipc::launch_maybe_pdl(tp2_hc_ipc::tp2ArHcInjectNormKernel<PACKED, COLLECT, PDL>,                        \
                                     grid, block, stream, PDL,                                                         \
                                     x_ptr, s_ptr, g_ptr, w_ptr, r_ptr, n_ptr, scale_ptr,                              \
                                     self_buf, peer_buf, self_sig, peer_sig, n_i32, eps_f, slot_u4);                   \
    } while (0)

    if (packed_dual) {
        if (collect_amax) {
            if (enable_pdl) DISPATCH_NORM(true, true, true);
            else            DISPATCH_NORM(true, true, false);
        } else {
            if (enable_pdl) DISPATCH_NORM(true, false, true);
            else            DISPATCH_NORM(true, false, false);
        }
    } else {
        if (collect_amax) {
            if (enable_pdl) DISPATCH_NORM(false, true, true);
            else            DISPATCH_NORM(false, true, false);
        } else {
            if (enable_pdl) DISPATCH_NORM(false, false, true);
            else            DISPATCH_NORM(false, false, false);
        }
    }
#undef DISPATCH_NORM

    return {std::move(residual), std::move(normed), std::move(scale_opt)};
}

torch::Tensor tp2ArHcInjectOnly(const torch::Tensor&        hyper_x,
                                const torch::Tensor&        local_sub,
                                const torch::Tensor&        gates,
                                const std::vector<int64_t>& buffers,
                                const std::vector<int64_t>& signals,
                                int64_t                     rank,
                                bool                        packed_dual,
                                bool                        enable_pdl,
                                int64_t                     buffer_bytes,
                                const std::vector<int64_t>& lamport_buffers,
                                int64_t                     lamport_buffer_bytes) {
    TORCH_CHECK(hyper_x.is_cuda() && local_sub.is_cuda() && gates.is_cuda(),
                "tp2ArHcInjectOnly inputs must be CUDA tensors");
    TORCH_CHECK(hyper_x.scalar_type() == torch::kBFloat16 && local_sub.scalar_type() == torch::kBFloat16
                    && gates.scalar_type() == torch::kBFloat16,
                "tp2ArHcInjectOnly inputs must be BF16");
    TORCH_CHECK(hyper_x.is_contiguous() && local_sub.is_contiguous() && gates.is_contiguous(),
                "tp2ArHcInjectOnly inputs must be contiguous");
    size_t slot_bytes = 0;
    (void)tp2_hc_ipc::resolve_cta_cap_u4(buffer_bytes, &slot_bytes);
    const int64_t max_rows_ipc      = static_cast<int64_t>(slot_bytes / (tp2_hc_ipc::kRowU4 * sizeof(uint4)));
    const int64_t rows              = hyper_x.numel() / (tp2_hc_ipc::kBranches * tp2_hc_ipc::kHidden);
    const int64_t expected_sub_rows = packed_dual ? (2 * rows) : rows;
    TORCH_CHECK(rows >= 1 && rows <= max_rows_ipc
                    && hyper_x.size(-1) == tp2_hc_ipc::kBranches * tp2_hc_ipc::kHidden,
                "tp2ArHcInjectOnly rows exceed IPC slot capacity");
    TORCH_CHECK(local_sub.dim() == 2 && local_sub.size(0) == expected_sub_rows
                    && local_sub.size(1) == tp2_hc_ipc::kHidden,
                "tp2ArHcInjectOnly local_sub shape mismatch");
    TORCH_CHECK(gates.numel() == rows * tp2_hc_ipc::kBranches && gates.size(-1) == tp2_hc_ipc::kBranches,
                "tp2ArHcInjectOnly gates shape mismatch");
    TORCH_CHECK(rank >= 0 && rank < 2 && buffers.size() == 2 && signals.size() == 2,
                "tp2ArHcInjectOnly requires 2 rank pointers");

    c10::cuda::CUDAGuard guard(hyper_x.device());
    const int            dev      = hyper_x.get_device();
    auto                 residual = torch::empty_like(hyper_x);

    auto*       self_sig = reinterpret_cast<tp2_hc_ipc::Tp2HcSignal*>(signals[rank]);
    auto*       peer_sig = reinterpret_cast<tp2_hc_ipc::Tp2HcSignal*>(signals[1 - rank]);
    const auto* self_buf = reinterpret_cast<const uint4*>(buffers[rank]);
    auto*       peer_buf = reinterpret_cast<uint4*>(buffers[1 - rank]);

    const size_t  lamport_slot_u4  = tp2_hc_ipc::resolve_lamport_slot_u4(lamport_buffer_bytes);
    const int64_t max_lamport_rows = static_cast<int64_t>(lamport_slot_u4 / tp2_hc_ipc::kRowU4);
    const bool    use_lamport      = (lamport_buffers.size() == 2 && lamport_slot_u4 > 0 && rows <= max_lamport_rows);

    const dim3  block(tp2_hc_ipc::kThreads);
    auto        stream = c10::cuda::getCurrentCUDAStream(dev).stream();
    const auto* x_ptr  = reinterpret_cast<const uint4*>(hyper_x.data_ptr());
    const auto* s_ptr  = reinterpret_cast<const uint4*>(local_sub.data_ptr());
    const auto* g_ptr  = reinterpret_cast<const uint64_t*>(gates.data_ptr());
    auto*       r_ptr  = reinterpret_cast<uint4*>(residual.data_ptr());
    const int   n_i32  = static_cast<int>(rows);

    if (use_lamport) {
        auto* self_lamp = reinterpret_cast<uint4*>(lamport_buffers[rank]);
        auto* peer_lamp = reinterpret_cast<uint4*>(lamport_buffers[1 - rank]);

#define DISPATCH_LAMPORT_ONLY(PACKED, PDL)                                                                             \
    do {                                                                                                               \
        static const int max_res_branch =                                                                              \
            tp2_hc_ipc::get_max_resident_ctas(tp2_hc_ipc::tp2ArHcLamportInjectOnlyBranchKernel<PACKED, PDL>, dev);     \
        if ((rows * tp2_hc_ipc::kBranches) <= max_res_branch) {                                                        \
            const dim3 grid(static_cast<unsigned>(rows * tp2_hc_ipc::kBranches));                                      \
            tp2_hc_ipc::launch_maybe_pdl(tp2_hc_ipc::tp2ArHcLamportInjectOnlyBranchKernel<PACKED, PDL>,                \
                                         grid, block, stream, PDL,                                                     \
                                         x_ptr, s_ptr, g_ptr, r_ptr, self_lamp, peer_lamp, self_sig, n_i32,            \
                                         lamport_slot_u4);                                                             \
        } else {                                                                                                       \
            static const int max_res_row =                                                                             \
                tp2_hc_ipc::get_max_resident_ctas(tp2_hc_ipc::tp2ArHcLamportInjectOnlyRowKernel<PACKED, PDL>, dev);    \
            const dim3 grid(static_cast<unsigned>(std::max<int64_t>(1, std::min<int64_t>(rows, max_res_row))));        \
            tp2_hc_ipc::launch_maybe_pdl(tp2_hc_ipc::tp2ArHcLamportInjectOnlyRowKernel<PACKED, PDL>,                   \
                                         grid, block, stream, PDL,                                                     \
                                         x_ptr, s_ptr, g_ptr, r_ptr, self_lamp, peer_lamp, self_sig, n_i32,            \
                                         lamport_slot_u4);                                                             \
        }                                                                                                              \
    } while (0)

        if (packed_dual) {
            if (enable_pdl) DISPATCH_LAMPORT_ONLY(true, true);
            else            DISPATCH_LAMPORT_ONLY(true, false);
        } else {
            if (enable_pdl) DISPATCH_LAMPORT_ONLY(false, true);
            else            DISPATCH_LAMPORT_ONLY(false, false);
        }
#undef DISPATCH_LAMPORT_ONLY
        return residual;
    }

    const size_t slot_u4 = slot_bytes / sizeof(uint4);

#define DISPATCH_ONLY(PACKED, PDL)                                                                                     \
    do {                                                                                                               \
        static const int max_res_only =                                                                                \
            tp2_hc_ipc::get_max_resident_ctas(tp2_hc_ipc::tp2ArHcInjectOnlyKernel<PACKED, PDL>, dev);                  \
        const int  cap  = std::min(tp2_hc_ipc::kMaxSignalBlocks, max_res_only);                                        \
        const dim3 grid(static_cast<unsigned>(std::max<int64_t>(1, std::min<int64_t>(rows, cap))));                    \
        tp2_hc_ipc::launch_maybe_pdl(tp2_hc_ipc::tp2ArHcInjectOnlyKernel<PACKED, PDL>,                                 \
                                     grid, block, stream, PDL,                                                         \
                                     x_ptr, s_ptr, g_ptr, r_ptr, self_buf, peer_buf, self_sig, peer_sig,               \
                                     n_i32, slot_u4);                                                                  \
    } while (0)

    if (packed_dual) {
        if (enable_pdl) DISPATCH_ONLY(true, true);
        else            DISPATCH_ONLY(true, false);
    } else {
        if (enable_pdl) DISPATCH_ONLY(false, true);
        else            DISPATCH_ONLY(false, false);
    }
#undef DISPATCH_ONLY

    return residual;
}

}  // namespace rtp_llm
