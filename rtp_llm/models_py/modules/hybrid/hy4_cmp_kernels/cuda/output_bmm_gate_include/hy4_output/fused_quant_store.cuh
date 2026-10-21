#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace deep_gemm::epilogue {

__device__ __forceinline__ float load_swizzled_bf16(
    const std::uint8_t* base, std::uint32_t row, std::uint32_t col) {
    constexpr std::uint32_t kAtomColumns = 64;
    constexpr std::uint32_t kAtomRows = 8;
    constexpr std::uint32_t kSwizzleBytes = 128;
    constexpr std::uint32_t kElementBytes = 2;
    const std::uint32_t atom_n = col / kAtomColumns;
    const std::uint32_t atom_col = col % kAtomColumns;
    const std::uint32_t atom_m = row / kAtomRows;
    const std::uint32_t atom_row = row % kAtomRows;
    const std::uint32_t byte_offset =
        atom_n * 16 * kSwizzleBytes + atom_m * kAtomRows * kSwizzleBytes +
        atom_row * kSwizzleBytes + ((atom_col / 8) ^ atom_row) * 16 +
        (atom_col % 8) * kElementBytes;
    const std::uint16_t raw =
        *reinterpret_cast<const std::uint16_t*>(base + byte_offset);
    return __uint_as_float(static_cast<std::uint32_t>(raw) << 16);
}

template <std::uint32_t kBlockM, std::uint32_t kBlockN,
          std::uint32_t kStoreBlockM, std::uint32_t kStoreBlockN,
          std::uint32_t kSwizzleCDMode, std::uint32_t kNumTMAStoreStages,
          std::uint32_t kNumStoreThreads, deep_gemm::GemmType kGemmType,
          bool kWithAccumulation, typename cd_dtype_t, typename pattern_cd_t>
CUTLASS_DEVICE void store_fused_quant(
    const deep_gemm::utils::PatternVisitor<pattern_cd_t>& smem_cd,
    const std::uint32_t& tmem_base_addr,
    const std::uint32_t& base_m_idx, const std::uint32_t& base_n_idx,
    const std::uint32_t& batch_idx, const std::uint32_t& effective_m,
    const std::uint32_t& shape_m, const std::uint32_t& shape_n,
    const std::uint32_t& epilogue_warp_idx,
    const std::uint32_t& lane_idx,
    const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
    std::uint8_t* quant_output, std::int32_t* packed_scale,
    int scale_stride, int output_stride, const __nv_bfloat16* gate, __nv_bfloat16* expanded) {
    static_assert((kBlockM == 16 || kBlockM == 32 || kBlockM == 64 ||
                   kBlockM == 128) &&
                  kBlockN == 128);
    static_assert(kStoreBlockM == 16 && kStoreBlockN == 128);
    static_assert(kBlockM % kStoreBlockM == 0);
    static_assert(kSwizzleCDMode == 128);
    static_assert(kNumStoreThreads == 128 || kNumStoreThreads == 256 ||
                  kNumStoreThreads == 384 || kNumStoreThreads == 512);
    static_assert(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>);
    static_assert(kGemmType == deep_gemm::GemmType::Batched &&
                  !kWithAccumulation);

    constexpr std::uint32_t kSwizzleAtomRows = 8;
    constexpr std::uint32_t kStoreBlockNAtom =
        kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr std::uint32_t kWarpsPerAtom = kStoreBlockNAtom / 32;
    constexpr std::uint32_t kThreadsPerWarpgroup = 128;
    constexpr std::uint32_t kWarpsPerWarpgroup =
        kThreadsPerWarpgroup / 32;
    constexpr std::uint32_t kNumEpilogueWarpgroups =
        kNumStoreThreads / kThreadsPerWarpgroup;
    static_assert(kNumTMAStoreStages >= kNumEpilogueWarpgroups);

    const std::uint32_t epilogue_warpgroup_idx =
        epilogue_warp_idx / kWarpsPerWarpgroup;
    const std::uint32_t local_warp_idx =
        epilogue_warp_idx % kWarpsPerWarpgroup;
    const std::uint32_t local_thread = local_warp_idx * 32 + lane_idx;
    auto* stage = reinterpret_cast<std::uint8_t*>(
        smem_cd[epilogue_warpgroup_idx]);
    const std::uint32_t num_stores = effective_m / kStoreBlockM;
    bool released_tmem = false;

#pragma unroll
    for (std::uint32_t store = epilogue_warpgroup_idx; store < num_stores;
         store += kNumEpilogueWarpgroups) {
        cutlass::arch::NamedBarrier::sync(
            kThreadsPerWarpgroup, epilogue_warpgroup_idx);
#pragma unroll
        for (std::uint32_t atom_m = 0; atom_m < 2; ++atom_m) {
            const std::uint32_t tmem_addr =
                tmem_base_addr + store * kStoreBlockM +
                atom_m * kSwizzleAtomRows;
            std::uint32_t values[kSwizzleAtomRows];
            cute::SM100_TMEM_LOAD_16dp256b1x::copy(
                tmem_addr, values[0], values[1], values[2], values[3]);
            cute::SM100_TMEM_LOAD_16dp256b1x::copy(
                tmem_addr | 0x00100000, values[4], values[5], values[6],
                values[7]);
            cutlass::arch::fence_view_async_tmem_load();

            const std::uint32_t row = lane_idx % 8;
            const std::uint32_t col =
                (local_warp_idx % 2) * 4 + lane_idx / 8;
            const std::uint32_t outer_offset =
                (local_warp_idx / kWarpsPerAtom) *
                kStoreBlockM * kSwizzleCDMode;
            const std::uint32_t inner_offset =
                atom_m * kSwizzleAtomRows * kSwizzleCDMode;
            auto* smem_ptr = stage +
                             outer_offset + inner_offset +
                             row * 128 + (col ^ row) * 16;
            deep_gemm::ptx::SM90_U32x4_STSM_T<int>::copy(
                deep_gemm::math::cast_into_bf16_and_pack(values[0], values[1]),
                deep_gemm::math::cast_into_bf16_and_pack(values[2], values[3]),
                deep_gemm::math::cast_into_bf16_and_pack(values[4], values[5]),
                deep_gemm::math::cast_into_bf16_and_pack(values[6], values[7]),
                smem_ptr);
        }

        if (store + kNumEpilogueWarpgroups >= num_stores) {
            deep_gemm::ptx::tcgen05_before_thread_sync();
            tmem_empty_barrier->arrive(0u);
            released_tmem = true;
        }
        cutlass::arch::NamedBarrier::sync(
            kThreadsPerWarpgroup, epilogue_warpgroup_idx);

        const std::uint32_t subgroup = local_thread / 16;
        const std::uint32_t subgroup_lane = local_thread % 16;

#pragma unroll
        for (std::uint32_t row_phase = 0; row_phase < 2; ++row_phase) {
            const std::uint32_t local_row = row_phase * 8 + subgroup;
            const std::uint32_t global_row =
                base_m_idx + store * kStoreBlockM + local_row;
            float values[8];
            float local_absmax = 1.0e-4f;
#pragma unroll
            for (std::uint32_t value = 0; value < 8; ++value) {
                const std::uint32_t col = subgroup_lane * 8 + value;
                const std::uint32_t offset = global_row * output_stride + batch_idx * shape_n + base_n_idx + col;
                const float a = load_swizzled_bf16(stage, local_row, col);
                float g = 0.0f;
                if (global_row < shape_m) {
                    expanded[offset] = __float2bfloat16_rn(a);
                    g = __bfloat162float(gate[offset]);
                }
                // Match tl.sigmoid, then its BF16 boundary before BF16 multiply.
                float e, sig;
                const float arg = __fmul_rn(-g, 1.4426950408889634f);
                asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(arg));
                const float denominator = __fadd_rn(1.0f, e);
                asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(sig) : "f"(denominator));
                sig = __bfloat162float(__float2bfloat16_rn(sig));
                values[value] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(a, sig)));
                local_absmax = fmaxf(local_absmax, fabsf(values[value]));
            }
            // Four lanes, eight adjacent values each: exactly one group32.
            local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, 2, 4));
            local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, 1, 4));
            const float initial_scale = __fdiv_rn(local_absmax, 448.0f);
            const unsigned bits = __float_as_uint(initial_scale);
            const unsigned exponent = ((bits >> 23) & 255u) + ((bits & 0x7fffffu) != 0);
            const float scale = __uint_as_float(exponent << 23);
            if (global_row < shape_m) {
#pragma unroll
                for (std::uint32_t value = 0; value < 8; ++value) {
                    const float v = fminf(fmaxf(__fdiv_rn(values[value], scale), -448.0f), 448.0f);
                    quant_output[global_row * output_stride + batch_idx * shape_n + base_n_idx + subgroup_lane * 8 + value] =
                        static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
                }
                if (subgroup_lane % 4 == 0) {
                    const std::uint32_t group = batch_idx * (shape_n / 32) + base_n_idx / 32 + subgroup_lane / 4;
                    reinterpret_cast<std::uint8_t*>(packed_scale)[(group / 4) * scale_stride * 4 + global_row * 4 + group % 4] = exponent;
                }
            }
        }
        cutlass::arch::NamedBarrier::sync(
            kThreadsPerWarpgroup, epilogue_warpgroup_idx);
    }

    if (!released_tmem) {
        deep_gemm::ptx::tcgen05_before_thread_sync();
        tmem_empty_barrier->arrive(0u);
    }
}

}  // namespace deep_gemm::epilogue
