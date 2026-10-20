// Adapted from cuda_extension 94fd20eb9179301e8cfb14a48ce0070cc447de9f.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

namespace rtp_dsv41_qb {

namespace cluster_mma_fp8 {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// ======================== Internal helpers ========================
namespace detail {

// K-major 128B-swizzle SMEM descriptor for FP8 operands.
// num_non_contiguous = 128/16 = 8 ; SBO = 8 * BLOCK_K * sizeof(e4m3) = 8*BLOCK_K.
template <int BLOCK_K>
__device__ __forceinline__
cute::UMMA::SmemDescriptor make_smem_desc_k_major(void* smem_ptr) {
    cute::UMMA::SmemDescriptor desc;
    desc.version_     = 1;
    desc.lbo_mode_    = 0;
    desc.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
    constexpr uint32_t SBO = 8u * BLOCK_K * sizeof(__nv_fp8_e4m3); // 8*BLOCK_K
    desc.stride_byte_offset_  = SBO >> 4;
    desc.leading_byte_offset_ = 0;
    desc.base_offset_         = 0;
    return desc;
}

// SF (UTCCP source) descriptor: SWIZZLE_NONE, one 8x128b atom. SBO = 8*16 = 128.
__device__ __forceinline__
cute::UMMA::SmemDescriptor make_sf_desc() {
    cute::UMMA::SmemDescriptor desc;
    desc.version_     = 1;
    desc.lbo_mode_    = 0;
    desc.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_NONE);
    desc.start_address_       = 0;
    desc.stride_byte_offset_  = (8 * 16) >> 4;   // 8
    desc.leading_byte_offset_ = 0;
    desc.base_offset_         = 0;
    return desc;
}

__device__ __forceinline__
void replace_sf_desc_addr(cute::UMMA::SmemDescriptor& desc, const void* smem_ptr) {
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(const_cast<void*>(smem_ptr));
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
}

__device__ __forceinline__
uint64_t sf_desc_bits(const cute::UMMA::SmemDescriptor& d) {
    return *reinterpret_cast<const uint64_t*>(&d);
}

// Advance descriptor .lo by one UMMA_K step within a stage.
// K-major fp8: UMMA_K=32 elems -> 32 bytes -> 32/16 = 2 units.
__device__ __forceinline__
uint32_t advance_lo_k(uint32_t base_lo, uint32_t kk) {
    return base_lo + kk * (32u * sizeof(__nv_fp8_e4m3) / 16u); // kk * 2
}

// Base InstrDescriptorBlockScaled. Operand order is selected at issue time.
template <int UMMA_N_T>
__device__ __forceinline__
cute::UMMA::InstrDescriptorBlockScaled make_block_scaled_idesc() {
    return cute::UMMA::make_instr_desc_block_scaled<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, cutlass::float_ue8m0_t,
        /*UMMA_M=*/256, UMMA_N_T,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
}

// Runtime descriptor (hi 32 bits) with per-sub-block SF ids applied.
__device__ __forceinline__
uint64_t make_runtime_idesc_with_sf_id(cute::UMMA::InstrDescriptorBlockScaled desc,
                                       uint32_t a_sf_id, uint32_t b_sf_id) {
    desc.a_sf_id_ = a_sf_id;
    desc.b_sf_id_ = b_sf_id;
    return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

// UTCCP (2cta): copy one 32x128b SF atom from a smem descriptor into a TMEM column.
__device__ __forceinline__ void utccp_4x32_2cta(uint32_t tmem_col, uint64_t sf_desc) {
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(tmem_col), "l"(sf_desc) : "memory");
}

// tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale (only hi 32 bits of idesc used).
//   desc_a / tmem_sfa : hardware A-operand value + scale
//   desc_b / tmem_sfb : hardware B-operand value + scale
__device__ __forceinline__ void mma_2sm_block_scale(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint64_t runtime_idesc, uint32_t accum,
    uint32_t tmem_sfa, uint32_t tmem_sfb) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
           "r"(static_cast<uint32_t>(runtime_idesc >> 32)), "r"(accum),
           "r"(tmem_sfa), "r"(tmem_sfb));
}

__device__ __forceinline__ void arrive_multicast_2sm(Barrier* bar, uint16_t mask) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    if (cute::elect_one_sync()) {
        asm volatile(
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
            :: "r"(addr), "h"(mask) : "memory");
    }
}

__device__ __forceinline__ void fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// Profiling helpers (only used on the kProfWait path).
__device__ __forceinline__ long long rdclock() {
    long long t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory"); return t;
}
__device__ __forceinline__ uint32_t laneid() {
    uint32_t l; asm volatile("mov.u32 %0, %%laneid;" : "=r"(l)); return l;
}

__device__ __forceinline__ bool elect_one_sync() {
    uint32_t pred;
    asm volatile("{\n\t.reg .pred p;\n\t"
        "elect.sync _|p, 0xffffffff;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}" : "=r"(pred));
    return pred != 0;
}

} // namespace detail

// ======================== Cluster FP8 block-scale MMA engine ========================
// BLOCK_K   : K per pipeline stage (multiple of UMMA_K(32); e.g. 128 -> 4 SF sub-blocks).
// NUM_STAGES: number of SMEM pipeline stages (bounds the per-lane descriptor table).
template <int BLOCK_K, int NUM_STAGES,
          int BLOCK_M_T = 128, int BLOCK_N_T = 128, bool kSwapAB = true>
struct ClusterMmaFP8BlockScale {
    // ---- Geometry ----
    static constexpr int BM = BLOCK_M_T, BN = BLOCK_N_T, NUM_MULTICAST = 2;
    static constexpr int UMMA_M = 256;
    static constexpr int UMMA_N = kSwapAB ? BM : BN;
    static constexpr int UMMA_K = 32, GRAN_K = 32;
    // V4.1: each K128 stage contains four independent K32 exponents.
    static constexpr int LOAD_BLOCK_M = BM / (kSwapAB ? NUM_MULTICAST : 1);
    static constexpr int LOAD_BLOCK_N = BN / (kSwapAB ? 1 : NUM_MULTICAST);
    static constexpr int STEPS_PER_STAGE = BLOCK_K / UMMA_K;          // = SF sub-blocks per stage
    static constexpr uint16_t CTA_MASK = (1 << NUM_MULTICAST) - 1;    // 0b11

    // ---- Operand SMEM per-stage byte strides (FP8 = 1 byte, tight K-major) ----
    static constexpr int SMEM_ACT_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * (int)sizeof(__nv_fp8_e4m3);
    static constexpr int SMEM_WGT_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * (int)sizeof(__nv_fp8_e4m3);

    // ---- Scale-factor layout ----
    static constexpr int NUM_UTCCP_ALIGNED = 128;
    static constexpr int SF_BLOCK_M        = ((BM + NUM_UTCCP_ALIGNED - 1) / NUM_UTCCP_ALIGNED) * NUM_UTCCP_ALIGNED;
    static constexpr int SF_BLOCK_N        = ((BN + NUM_UTCCP_ALIGNED - 1) / NUM_UTCCP_ALIGNED) * NUM_UTCCP_ALIGNED;
    static constexpr int NUM_SF_ATOMS_M    = SF_BLOCK_M / NUM_UTCCP_ALIGNED;
    static constexpr int NUM_SF_ATOMS_N    = SF_BLOCK_N / NUM_UTCCP_ALIGNED;
    // TMEM columns each SF operand occupies (caller lays these out after the accumulator).
    static constexpr int NUM_SF_TMEM_COLS_ACT = SF_BLOCK_M / 32;
    static constexpr int NUM_SF_TMEM_COLS_WGT = SF_BLOCK_N / 32;
    // Per-stage SF smem byte strides (packed uint32: 4 UE8M0 per word).
    static constexpr int SMEM_SF_ACT_PER_STAGE = SF_BLOCK_M * (int)sizeof(uint32_t);
    static constexpr int SMEM_SF_WGT_PER_STAGE = SF_BLOCK_N * (int)sizeof(uint32_t);

    static_assert(BLOCK_K % UMMA_K == 0, "BLOCK_K must be a multiple of UMMA_K(32)");
    static_assert(BLOCK_K == 128, "one SF word covers four K32 groups");
    // BM is fully parameterised: SMEM strides scale with LOAD_BLOCK_M, the SF path
    // UTCCPs align(BM,128) rows regardless (padding rows carry UE8M0_ONE), and the
    // K-major smem descriptor's SBO depends only on BLOCK_K. Real constraints:
    static_assert(BM % 16 == 0 && BM <= 256, "invalid (swap) UMMA N = BM");
    static_assert((BN % 16) == 0 && BN <= 256, "invalid non-swap UMMA N");

    // Descriptor state precomputed once per MMA warp.
    struct DescState {
        uint32_t act_lo, wgt_lo;   // per-lane, stage-indexed .lo (valid for lane < NUM_STAGES)
        uint32_t act_hi, wgt_hi;   // constant .hi
        cute::UMMA::InstrDescriptorBlockScaled instr_desc;
    };

    // Build the per-stage descriptor tables. Call once at MMA-warp entry, all 32 lanes.
    //   smem_act_base = activation SMEM base
    //   smem_wgt_base = weight SMEM base
    static __device__ __forceinline__
    DescState init_desc(__nv_fp8_e4m3* smem_act_base, __nv_fp8_e4m3* smem_wgt_base, uint32_t lane_id) {
        auto act_desc = detail::make_smem_desc_k_major<BLOCK_K>(smem_act_base);
        auto wgt_desc = detail::make_smem_desc_k_major<BLOCK_K>(smem_wgt_base);

        DescState ds;
        ds.act_lo = (lane_id < NUM_STAGES)
            ? act_desc.lo + lane_id * (SMEM_ACT_PER_STAGE / 16) : 0u;
        ds.wgt_lo = (lane_id < NUM_STAGES)
            ? wgt_desc.lo + lane_id * (SMEM_WGT_PER_STAGE / 16) : 0u;
        ds.act_hi     = act_desc.hi;
        ds.wgt_hi     = wgt_desc.hi;
        ds.instr_desc = detail::make_block_scaled_idesc<UMMA_N>();
        return ds;
    }

    template <bool kProfWait = false>
    static __device__ __forceinline__
    void run_tile(const DescState& ds,
                  Barrier* with_sf_full_barriers, Barrier* empty_barriers,
                  Barrier& tmem_full, Barrier& tmem_empty,
                  const uint8_t* smem_sf_act_base, const uint8_t* smem_sf_wgt_base,
                  uint32_t tmem_c, uint32_t tmem_sf_act, uint32_t tmem_sf_wgt,
                  int num_k_tiles, uint32_t accum_phase,
                  uint32_t& stage_idx, uint32_t& phase,
                  long long* wait_out = nullptr) {
        long long wait_acc = 0, tw = 0;
        const bool prof0 = kProfWait && (detail::laneid() == 0);

        if (prof0) tw = detail::rdclock();
        tmem_empty.wait(accum_phase ^ 1);
        if (prof0) wait_acc += detail::rdclock() - tw;
        detail::fence_after_sync();

        for (int k = 0; k < num_k_tiles; ++k) {
            if (prof0) tw = detail::rdclock();
            with_sf_full_barriers[stage_idx].wait(phase);
            if (prof0) wait_acc += detail::rdclock() - tw;
            detail::fence_after_sync();

            uint32_t a_base = __shfl_sync(0xffffffff, ds.act_lo, stage_idx);
            uint32_t b_base = __shfl_sync(0xffffffff, ds.wgt_lo, stage_idx);

            if (detail::elect_one_sync()) {
                // Refresh scales every K128 stage.
                {
                    auto sf_desc = detail::make_sf_desc();
                    const auto* sf_act = reinterpret_cast<const uint32_t*>(
                        smem_sf_act_base + stage_idx * SMEM_SF_ACT_PER_STAGE);
                    const auto* sf_wgt = reinterpret_cast<const uint32_t*>(
                        smem_sf_wgt_base + stage_idx * SMEM_SF_WGT_PER_STAGE);
                    #pragma unroll
                    for (int i = 0; i < NUM_SF_ATOMS_M; ++i) {
                        detail::replace_sf_desc_addr(sf_desc, sf_act + i * NUM_UTCCP_ALIGNED);
                        detail::utccp_4x32_2cta(tmem_sf_act + i * 4, detail::sf_desc_bits(sf_desc));
                    }
                    #pragma unroll
                    for (int i = 0; i < NUM_SF_ATOMS_N; ++i) {
                        detail::replace_sf_desc_addr(sf_desc, sf_wgt + i * NUM_UTCCP_ALIGNED);
                        detail::utccp_4x32_2cta(tmem_sf_wgt + i * 4, detail::sf_desc_bits(sf_desc));
                    }
                }

                #pragma unroll
                for (int kk = 0; kk < STEPS_PER_STAGE; ++kk) {
                    uint64_t rdesc = detail::make_runtime_idesc_with_sf_id(ds.instr_desc, kk, kk);
                    uint32_t a_lo = detail::advance_lo_k(a_base, kk);
                    uint32_t b_lo = detail::advance_lo_k(b_base, kk);
                    uint64_t act_full = (static_cast<uint64_t>(ds.act_hi) << 32) | a_lo;
                    uint64_t wgt_full = (static_cast<uint64_t>(ds.wgt_hi) << 32) | b_lo;

                    uint32_t accum_flag = (k > 0 || kk > 0) ? 1 : 0;
                    if constexpr (kSwapAB) {
                        // A-operand = weight, B-operand = activation.
                        detail::mma_2sm_block_scale(tmem_c, wgt_full, act_full, rdesc, accum_flag,
                                                    tmem_sf_wgt, tmem_sf_act);
                    } else {
                        // A-operand = activation, B-operand = weight.
                        detail::mma_2sm_block_scale(tmem_c, act_full, wgt_full, rdesc, accum_flag,
                                                    tmem_sf_act, tmem_sf_wgt);
                    }
                }
            }
            __syncwarp();

            detail::arrive_multicast_2sm(&empty_barriers[stage_idx], CTA_MASK);
            if (k == num_k_tiles - 1) {
                detail::arrive_multicast_2sm(&tmem_full, CTA_MASK);
            }
            __syncwarp();

            stage_idx = (stage_idx + 1) % NUM_STAGES;
            if (stage_idx == 0) phase ^= 1;
        }

        if (prof0 && wait_out) *wait_out = wait_acc;
    }
};

} // namespace cluster_mma_fp8

}  // namespace rtp_dsv41_qb
