// Adapted from cuda_extension 94fd20eb9179301e8cfb14a48ce0070cc447de9f.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

// CUTLASS/CuTe headers
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>

namespace rtp_dsv41_qb {
namespace wq_b {
constexpr int BLOCK_K=128, BLOCK_N=128, CLUSTER_SIZE=2, NUM_MULTICAST=2;
constexpr int CLUSTER_BLOCK_N=256, LOAD_BLOCK_N=128, FP8_ELEM_SIZE=1;
constexpr int NUM_UTCCP_ALIGNED=128, SMEM_B_PER_STAGE=16384, SMEM_SFB_PER_STAGE=512;
constexpr int NUM_SFB_TMEM_COLS=4;
constexpr int NUM_TMA_STORE_STAGES=2, NUM_STORE_THREADS=128, NUM_NON_EPI_THREADS=128;
constexpr int STORE_BLOCK_N=128, STORE_BLOCK_N_ATOM=64, SWIZZLE_CD=128;
constexpr int SMEM_CAPACITY=232448;
constexpr uint32_t UE8M0_ONE=0x7f;
static constexpr int div_up(int a, int b)  { return (a + b - 1) / b; }
static constexpr int align_up(int a, int b) { return div_up(a, b) * b; }
static constexpr int num_aligned_tmem_cols(int c) {
    if (c <= 32)  return 32;
    if (c <= 64)  return 64;
    if (c <= 128) return 128;
    if (c <= 256) return 256;
    return 512;
}

static constexpr int store_m_for(int m) {
    if (m <= 16) return 16;
    if (m == 32) return 32;
    if (m == 64) return 64;
    if (m == 96) return 48;
    if (m == 128) return 64;
    if (m == 160) return 40;
    if (m == 192) return 32;
    if (m == 224) return 56;
    if (m == 240) return 48;
    return 64;
}

// ---- Compile-time helpers parameterised on M (UMMA_N = M, no padding) ----
// One tile covers BM rows; the persistent scheduler also iterates M tiles.
// Smaller M shrinks SMEM_A and the TMEM accumulator footprint, so the pipeline
// deepens automatically (M=32 -> ~11 stages, matching DeepGEMM's pick).
template <int M_> struct SwapDims {
    static_assert(M_ >= 16 && M_ <= 256 && M_ % 16 == 0,
                  "CSA swap-AB batch tile must be a multiple of 16 in [16,256]");

    static constexpr int UMMA_N           = M_;
    static constexpr int LOAD_BLOCK_M     = M_ / NUM_MULTICAST;
    static constexpr int SMEM_A_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * FP8_ELEM_SIZE;

    // SFA stays UTCCP-aligned to 128 rows (rows >= M carry UE8M0_ONE padding;
    // the MMA only consumes the first M).
    static constexpr int SF_BLOCK_M         = align_up(M_, NUM_UTCCP_ALIGNED);
    static constexpr int SMEM_SFA_PER_STAGE = SF_BLOCK_M * (int)sizeof(uint32_t);
    static constexpr int NUM_SFA_TMEM_COLS  = SF_BLOCK_M / 32;

    static constexpr int SMEM_PER_STAGE = SMEM_A_PER_STAGE + SMEM_B_PER_STAGE +
                                          SMEM_SFA_PER_STAGE + SMEM_SFB_PER_STAGE;

    // Use larger drain batches to reduce epilogue synchronization and TMA
    // transaction overhead. For M>=160, keep the largest tested batch that
    // does not reduce the A/B pipeline depth; M256 is the single-stage exception.
    static constexpr int STORE_M = store_m_for(M_);
    static_assert(M_ % STORE_M == 0 && STORE_M % 8 == 0,
                  "STORE_M must tile M exactly in 8-row TMEM load atoms");
    static constexpr int SMEM_CD_PER_STAGE =
        STORE_M * STORE_BLOCK_N * sizeof(__nv_bfloat16);
    static constexpr int SMEM_CD_TOTAL = SMEM_CD_PER_STAGE * NUM_TMA_STORE_STAGES;

    static constexpr int NUM_EPI_STAGES      = M_ == 256 ? 1 : 2;
    // TMEM layout: [accum (UMMA_N*NUM_EPI_STAGES)] [SFA cols] [SFB cols]
    static constexpr int NUM_ACCUM_TMEM_COLS = UMMA_N * NUM_EPI_STAGES;
    static constexpr int TMEM_START_SFA      = NUM_ACCUM_TMEM_COLS;
    static constexpr int TMEM_START_SFB      = NUM_ACCUM_TMEM_COLS + NUM_SFA_TMEM_COLS;
    static constexpr int NUM_TMEM_COLS_RAW =
        NUM_ACCUM_TMEM_COLS + NUM_SFA_TMEM_COLS + NUM_SFB_TMEM_COLS;
    static_assert(NUM_TMEM_COLS_RAW <= 512,
                  "CSA TMEM allocation exceeds the 512-column limit");
    static constexpr int NUM_TMEM_COLS = num_aligned_tmem_cols(NUM_TMEM_COLS_RAW);

    // Number of pipeline stages fitting the SMEM budget.
    // Overhead: smem_cd + barriers(full/empty/with_sf per stage + tmem full/empty) + tmem_ptr.
    // (The fused indexer compressor is fully register-resident -- no smem term.)
    static constexpr int SMEM_BARRIERS = (16 * 3 + NUM_EPI_STAGES * 2) * 8;
    static constexpr int SMEM_OVERHEAD = SMEM_CD_TOTAL + SMEM_BARRIERS + 8;
    static constexpr int STAGES_RAW    = (SMEM_CAPACITY - SMEM_OVERHEAD) / SMEM_PER_STAGE;
    static constexpr int NUM_STAGES    = STAGES_RAW > 12 ? 12 : STAGES_RAW;
};

}
// ======================== Barrier alias ========================
namespace mma_desc {
using Barrier = cutlass::arch::ClusterTransactionBarrier;
} // namespace mma_desc

// ======================== PTX Wrappers ========================
// Block-scale MMA + descriptors live in cluster_mma_fp8.cuh. Here we keep only the
// helpers used by the producer / SF layout / epilogue / init in this kernel.
namespace ptx {

// TMEM alloc/dealloc for 2SM
__device__ __forceinline__ void tcgen05_alloc_2sm(uint32_t smem_addr, uint32_t num_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem_addr), "r"(num_cols));
}
__device__ __forceinline__ void tcgen05_dealloc_2sm(uint32_t taddr, uint32_t num_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        :: "r"(taddr), "r"(num_cols));
}

// Fences
__device__ __forceinline__ void tcgen05_fence_before_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}
__device__ __forceinline__ void tcgen05_fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// TMEM load: 32dp32b, x4 (4 FP32 per lane) — used by the non-swap FP32 store
__device__ __forceinline__ void tmem_load_32dp32b4x(
    uint32_t tmem_addr,
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
        : "r"(tmem_addr));
}

// TMEM load: 32dp32b, x8 (8 FP32 per lane) — used by the swap-AB FP32 store
__device__ __forceinline__ void tmem_load_32dp32b8x(
    uint32_t tmem_addr,
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3,
    uint32_t& v4, uint32_t& v5, uint32_t& v6, uint32_t& v7) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3),
          "=r"(v4), "=r"(v5), "=r"(v6), "=r"(v7)
        : "r"(tmem_addr));
}

// TMEM load: 16dp256b, x1 (4 FP32 per lane). Two loads, with datapaths 0/16,
// expose an 8-row x 32-col warp tile as four same-row values per lane.
__device__ __forceinline__ void tmem_load_16dp256b1x(
    uint32_t tmem_addr,
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3) {
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
        : "r"(tmem_addr));
}
__device__ __forceinline__ void tmem_load_fence() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// Shared memory load/store (used by the SF layout warp and the epilogue)
__device__ __forceinline__ uint32_t ld_shared_u32(const uint32_t* ptr) {
    uint32_t v;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr));
    return v;
}
__device__ __forceinline__ void st_shared_u32(void* ptr, uint32_t v) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(v) : "memory");
}

// Warp-cooperative BF16 transpose into a 128B-swizzled TMA store tile.
__device__ __forceinline__ void stmatrix_x4_trans(
    void* smem_ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile(
        "stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [%0], {%1,%2,%3,%4};"
        :: "l"(__cvta_generic_to_shared(smem_ptr)),
           "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}
__device__ __forceinline__ void st_shared_v4_u32(void* ptr, uint32_t v0, uint32_t v1,
                                                 uint32_t v2, uint32_t v3) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

// Cluster utilities
__device__ __forceinline__ uint32_t block_rank_in_cluster() {
    uint32_t rank;
    asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(rank));
    return rank;
}
__device__ __forceinline__ void cluster_sync() {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}
__device__ __forceinline__ uint32_t get_lane_idx() {
    uint32_t lane;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(lane));
    return lane;
}
__device__ __forceinline__ bool elect_one_sync() {
    uint32_t pred;
    asm volatile("{\n\t.reg .pred p;\n\t"
        "elect.sync _|p, 0xffffffff;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}" : "=r"(pred));
    return pred != 0;
}

__device__ __forceinline__ long long rdclock() {
    long long t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

// %globaltimer, NOT %clock64: device-wide and in NANOSECONDS, so a timeout can be
// stated in real time. clock64 is per-SM and its rate is not architectural, which
// makes it fine for the relative prof stamps above and wrong for a deadline.
__device__ __forceinline__ unsigned long long rdtimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

} // namespace ptx

// ======================== TMA Copy Helpers ========================
namespace tma {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// Plain (per-CTA) TMA 2D load for FP8 operands. Each CTA loads its own box;
// the cta_group::2 MMA combines the pair across SMs. (Aligned with w1_merged.)
__device__ __forceinline__
void copy_2d_fp8(void const* desc_ptr, Barrier* barrier_ptr,
                 __nv_fp8_e4m3* smem_ptr, uint32_t k_idx, uint32_t mn_idx) {
    cute::SM90_TMA_LOAD_2D::copy(
        desc_ptr,
        reinterpret_cast<uint64_t*>(barrier_ptr),
        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
        smem_ptr, k_idx, mn_idx);
}

__device__ __forceinline__
void store_2d(void const* desc_ptr, void* smem_ptr,
              uint32_t col_idx, uint32_t row_idx) {
    cute::SM90_TMA_STORE_2D::copy(desc_ptr, smem_ptr, col_idx, row_idx);
}

} // namespace tma

}
