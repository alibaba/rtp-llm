// Adapted from RTP kernel bdafb874 q_b_proj.cuh; see qb_include/NOTICE.md.
#pragma once

#include <cutlass/arch/barrier.h>
#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <rtp_kernel/glm5/deep_gemm/fp8_gemm.cuh>
#include <rtp_kernel/glm5/norm_quant.cuh>

#include <cuda_bf16.h>

#include <cstdint>

namespace hy4_qb {

struct Args {
  const float* cos_sin;
  const std::int32_t* positions;
  __nv_bfloat16* q_nope;
  __nv_bfloat16* sparse_query;
  std::int32_t rows;
  std::int32_t heads;
};

using rtp_kernel::glm5::detail::Bf16x8;
using rtp_kernel::glm5::detail::pack_bf16x8;
using rtp_kernel::glm5::detail::unpack_bf16x8;

template <typename SmemCd>
__device__ __forceinline__ Bf16x8 load_chunk(
    const SmemCd& smem_cd, std::uint32_t store_stage,
    int row, int sublane) {
  const int n_atom = sublane / 8;
  const int n_chunk = sublane % 8;
  const int byte_offset = n_atom * 16 * 128 + row * 128 +
                          (n_chunk ^ (row % 8)) * 16;
  const auto* pointer =
      reinterpret_cast<const std::uint8_t*>(smem_cd[store_stage]) + byte_offset;
  return *reinterpret_cast<const Bf16x8*>(pointer);
}

template <std::uint32_t kThreadsPerStore, typename SmemCd>
__device__ __forceinline__ void stage_bf16(
    const SmemCd& smem_cd,
    std::uint32_t store_stage,
    std::uint32_t tmem_base,
    std::uint32_t epilogue_warp,
    std::uint32_t lane,
    std::uint32_t barrier_group,
    const cutlass::arch::ClusterTransactionBarrier* empty_barrier,
    bool release_tmem) {
  static_assert(kThreadsPerStore == 256);
  constexpr std::uint32_t kWarpsPerWarpgroup = 4;
  constexpr std::uint32_t kRowsPerWarpgroup = 8;

  cutlass::arch::NamedBarrier::sync(
      kThreadsPerStore, barrier_group * 2);
  const std::uint32_t warpgroup = epilogue_warp / kWarpsPerWarpgroup;
  const std::uint32_t warp = epilogue_warp % kWarpsPerWarpgroup;
  const std::uint32_t tile_m = warpgroup * kRowsPerWarpgroup;
  const std::uint32_t address = tmem_base + tile_m;
  std::uint32_t values[8];
  cute::SM100_TMEM_LOAD_16dp256b1x::copy(
      address, values[0], values[1], values[2], values[3]);
  cute::SM100_TMEM_LOAD_16dp256b1x::copy(
      address | 0x00100000, values[4], values[5], values[6], values[7]);
  cutlass::arch::fence_view_async_tmem_load();

  const std::uint32_t n_atom = warp / 2;
  auto* smem_base = reinterpret_cast<std::uint8_t*>(smem_cd[store_stage]) +
                    n_atom * 16 * 128 + tile_m * 128;
  const std::uint32_t row = lane % kRowsPerWarpgroup;
  const std::uint32_t column = (warp % 2) * 4 + lane / kRowsPerWarpgroup;
  auto* destination = smem_base + row * 128 + (column ^ row) * 16;
  deep_gemm::ptx::SM90_U32x4_STSM_T<int>::copy(
      deep_gemm::math::cast_into_bf16_and_pack(values[0], values[1]),
      deep_gemm::math::cast_into_bf16_and_pack(values[2], values[3]),
      deep_gemm::math::cast_into_bf16_and_pack(values[4], values[5]),
      deep_gemm::math::cast_into_bf16_and_pack(values[6], values[7]),
      destination);

  if (release_tmem) {
    deep_gemm::ptx::tcgen05_before_thread_sync();
    empty_barrier->arrive(0u);
  }
  cutlass::arch::NamedBarrier::sync(
      kThreadsPerStore, barrier_group * 2 + 1);
}

template <int kHeadDim, int kNopeDim, int kPeDim, int kSparseQueryDim,
          typename SmemCd>
__device__ __forceinline__ void finalize(
    const SmemCd& smem_cd,
    std::uint32_t store_stage,
    const Args& args,
    int base_m,
    int base_n,
    int epilogue_warp,
    int lane) {
  const int sublane = lane & 15;
  const int local_row = epilogue_warp * 2 + (lane >> 4);
  const int row = base_m + local_row;
  if (row >= args.rows) {
    return;
  }

  const int head = base_n / kHeadDim;
  const int half = (base_n / (kHeadDim / 2)) & 1;
  const int token_head = row * args.heads + head;
  const int n0 = sublane * 8;
  static_assert(kHeadDim == kNopeDim + kPeDim);
  static_assert(kSparseQueryDim >= kPeDim);
  const Bf16x8 input = load_chunk(
      smem_cd, store_stage, local_row, sublane);
  if (half == 0) {
    *reinterpret_cast<Bf16x8*>(
        args.q_nope + token_head * kHeadDim + n0) = input;
    return;
  }
  if (sublane < 8) {
    *reinterpret_cast<Bf16x8*>(
        args.q_nope + token_head * kHeadDim + kHeadDim / 2 + n0) = input;
    return;
  }

  const int pe0 = (sublane - 8) * 8;
  float values[8];
  unpack_bf16x8(input, values);
  const float* cos = args.cos_sin + args.positions[row] * kPeDim + pe0 / 2;
  const float* sin = cos + kPeDim / 2;
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const float x0 = values[i];
    const float x1 = values[i + 1];
    values[i] = __fmaf_rn(x0, cos[i / 2], -__fmul_rn(x1, sin[i / 2]));
    values[i + 1] = __fmaf_rn(x1, cos[i / 2], __fmul_rn(x0, sin[i / 2]));
  }
  *reinterpret_cast<Bf16x8*>(args.q_nope + token_head * kHeadDim + kNopeDim + pe0) = pack_bf16x8(values);
  *reinterpret_cast<Bf16x8*>(
      args.sparse_query + token_head * kSparseQueryDim +
      (kSparseQueryDim - kPeDim) + pe0) =
      pack_bf16x8(values);
}

template <int kHeadDim, int kNopeDim, int kPeDim, int kSparseQueryDim,
          int kEpilogueWarpgroups>
struct Epilogue {
  static constexpr bool kParallelizeNormalStore = true;

  template <std::uint32_t BLOCK_M, std::uint32_t BLOCK_N,
            std::uint32_t STORE_BLOCK_M, std::uint32_t STORE_BLOCK_N,
            std::uint32_t kSwizzleCDMode,
            std::uint32_t kNumTMAStoreStages,
            std::uint32_t kNumUMMAStoreThreads,
            deep_gemm::GemmType kGemmType, bool kWithAccumulation,
            typename cd_dtype_t, typename transform_t,
            typename pattern_cd_t>
  CUTLASS_DEVICE static void store_swap_ab(
      const deep_gemm::utils::PatternVisitor<pattern_cd_t>& smem_cd,
      std::uint32_t& store_stage,
      std::uint32_t tmem_base,
      std::uint32_t base_m,
      std::uint32_t base_n,
      std::uint32_t,
      std::uint32_t effective_m,
      std::uint32_t epilogue_warp,
      std::uint32_t lane,
      const cutlass::arch::ClusterTransactionBarrier* empty_barrier,
      const cute::TmaDescriptor&,
      const void* opaque_args) {
    static_assert(BLOCK_M % 16 == 0 && BLOCK_M <= 128 &&
                  BLOCK_N == 128 && STORE_BLOCK_M == 16 &&
                  STORE_BLOCK_N == 128 && kSwizzleCDMode == 128 &&
                  kNumTMAStoreStages == 2 &&
                  (kNumUMMAStoreThreads == 256 ||
                   kNumUMMAStoreThreads == 512) &&
                  kGemmType == deep_gemm::GemmType::Normal &&
                  !kWithAccumulation &&
                  cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>);
    (void)sizeof(transform_t);
    const auto& args = *static_cast<const Args*>(opaque_args);
    const std::uint32_t stores = effective_m / STORE_BLOCK_M;
    constexpr std::uint32_t kThreadsPerStore = 256;
    constexpr std::uint32_t kWarpsPerStore = kThreadsPerStore / 32;
    constexpr std::uint32_t kConcurrentStores =
        kNumUMMAStoreThreads / kThreadsPerStore;

    if constexpr (kConcurrentStores == 1) {
      for (std::uint32_t store_idx = 0; store_idx < stores; ++store_idx) {
        const bool last = store_idx + 1 == stores;
        stage_bf16<kThreadsPerStore>(
            smem_cd, store_stage, tmem_base + store_idx * STORE_BLOCK_M,
            epilogue_warp, lane, 0, empty_barrier, last);
        finalize<kHeadDim, kNopeDim, kPeDim, kSparseQueryDim>(
            smem_cd, store_stage, args,
            static_cast<int>(base_m + store_idx * STORE_BLOCK_M),
            static_cast<int>(base_n), static_cast<int>(epilogue_warp),
            static_cast<int>(lane));
        store_stage = (store_stage + 1) % kNumTMAStoreStages;
      }
    } else {
      const std::uint32_t store_group = epilogue_warp / kWarpsPerStore;
      const std::uint32_t local_warp = epilogue_warp % kWarpsPerStore;
      store_stage = store_group;
      bool released_tmem = false;
      for (std::uint32_t store_idx = store_group; store_idx < stores;
           store_idx += kConcurrentStores) {
        const bool last = store_idx + kConcurrentStores >= stores;
        stage_bf16<kThreadsPerStore>(
            smem_cd, store_stage, tmem_base + store_idx * STORE_BLOCK_M,
            local_warp, lane, store_group, empty_barrier, last);
        finalize<kHeadDim, kNopeDim, kPeDim, kSparseQueryDim>(
            smem_cd, store_stage, args,
            static_cast<int>(base_m + store_idx * STORE_BLOCK_M),
            static_cast<int>(base_n), static_cast<int>(local_warp),
            static_cast<int>(lane));
        released_tmem = last;
      }
      if (!released_tmem) {
        deep_gemm::ptx::tcgen05_before_thread_sync();
        empty_barrier->arrive(0u);
      }
    }
  }

  template <std::uint32_t BLOCK_M, std::uint32_t BLOCK_N,
            std::uint32_t STORE_BLOCK_M, std::uint32_t STORE_BLOCK_N,
            std::uint32_t kSwizzleCDMode,
            std::uint32_t kNumTMAStoreStages,
            std::uint32_t kNumUMMAStoreThreads,
            deep_gemm::GemmType kGemmType, bool kWithAccumulation,
            typename cd_dtype_t, typename transform_t,
            typename pattern_cd_t>
  CUTLASS_DEVICE static void store_normal(
      const deep_gemm::utils::PatternVisitor<pattern_cd_t>& smem_cd,
      std::uint32_t& store_stage,
      std::uint32_t tmem_base,
      std::uint32_t base_m,
      std::uint32_t base_n,
      std::uint32_t,
      std::uint32_t epilogue_warp,
      std::uint32_t lane,
      const cutlass::arch::ClusterTransactionBarrier* empty_barrier,
      const cute::TmaDescriptor&,
      const void* opaque_args) {
    static_assert(BLOCK_M == 128 && BLOCK_N == 224 &&
                  STORE_BLOCK_M == 128 && STORE_BLOCK_N == 32 &&
                  kSwizzleCDMode == 64 &&
                  kNumTMAStoreStages >= kEpilogueWarpgroups &&
                  kNumUMMAStoreThreads ==
                      128 * kEpilogueWarpgroups &&
                  kGemmType == deep_gemm::GemmType::Normal &&
                  !kWithAccumulation &&
                  cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>);
    (void)sizeof(transform_t);
    const auto& args = *static_cast<const Args*>(opaque_args);
    constexpr int kChunksPerStore = STORE_BLOCK_N / 8;
    constexpr int kStores = BLOCK_N / STORE_BLOCK_N;
    constexpr int kWarpsPerStore = 4;
    const int store_group = epilogue_warp / kWarpsPerStore;
    const int local_warp = epilogue_warp % kWarpsPerStore;
    bool released_tmem = false;

#pragma unroll
    for (int store = store_group; store < kStores;
         store += kEpilogueWarpgroups) {
      if constexpr (kEpilogueWarpgroups > 1) {
        store_stage = store_group;
      }
      auto* smem_base = reinterpret_cast<std::uint8_t*>(
          smem_cd[store_stage]);
#pragma unroll
      for (int chunk = 0; chunk < kChunksPerStore; ++chunk) {
        const int bank_group = chunk + lane * kChunksPerStore;
        const int row = bank_group / 8;
        int column = bank_group % 8;
        column ^= row % 4;
        const std::uint32_t address =
            tmem_base + store * STORE_BLOCK_N + chunk * 8;
        std::uint32_t values[8];
        cute::SM100_TMEM_LOAD_32dp32b8x::copy(
            address, values[0], values[1], values[2], values[3], values[4],
            values[5], values[6], values[7]);
        cutlass::arch::fence_view_async_tmem_load();
        auto* destination = smem_base + local_warp * 32 * 64 +
                            row * 128 + column * 16;
        deep_gemm::ptx::st_shared(
            destination,
            deep_gemm::math::cast_into_bf16_and_pack(values[0], values[1]),
            deep_gemm::math::cast_into_bf16_and_pack(values[2], values[3]),
            deep_gemm::math::cast_into_bf16_and_pack(values[4], values[5]),
            deep_gemm::math::cast_into_bf16_and_pack(values[6], values[7]));
      }

      const bool last = store + kEpilogueWarpgroups >= kStores;
      if (last) {
        deep_gemm::ptx::tcgen05_before_thread_sync();
        empty_barrier->arrive(0u);
        released_tmem = true;
      }
      cutlass::arch::NamedBarrier::sync(
          128, store_group * 2);

      const int thread = local_warp * 32 + lane;
      const int row_in_wave = thread / 4;
      const int chunk = thread % 4;
      const int column = static_cast<int>(base_n) + store * STORE_BLOCK_N;
      const int head = column / kHeadDim;
      const int head_column = column % kHeadDim;
#pragma unroll
      for (int wave = 0; wave < 4; ++wave) {
        const int local_row = wave * 32 + row_in_wave;
        const int row = static_cast<int>(base_m) + local_row;
        const int chunk_column = column + chunk * 8;
        if (row < args.rows && chunk_column < args.heads * kHeadDim) {
          const int token_head = row * args.heads + head;
          const int warp = local_row / 32;
          const int lane = local_row % 32;
          const int physical_row = lane / 2;
          const int physical_column =
              (chunk + (lane % 2) * 4) ^ (physical_row % 4);
          const Bf16x8 values = *reinterpret_cast<const Bf16x8*>(
              smem_base + warp * 32 * 64 + physical_row * 128 +
              physical_column * 16);
          if (head_column < kNopeDim) {
            *reinterpret_cast<Bf16x8*>(
                args.q_nope + token_head * kHeadDim + head_column +
                chunk * 8) = values;
          } else {
            float rotated[8];
            unpack_bf16x8(values, rotated);
            const int pe0 = head_column - kNopeDim + chunk * 8;
            const float* cos =
                args.cos_sin + args.positions[row] * kPeDim + pe0 / 2;
            const float* sin = cos + kPeDim / 2;
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
              const float x0 = rotated[i];
              const float x1 = rotated[i + 1];
              rotated[i] = __fmaf_rn(x0, cos[i / 2], -__fmul_rn(x1, sin[i / 2]));
              rotated[i + 1] = __fmaf_rn(x1, cos[i / 2], __fmul_rn(x0, sin[i / 2]));
            }
            *reinterpret_cast<Bf16x8*>(args.q_nope + token_head * kHeadDim + kNopeDim + pe0) = pack_bf16x8(rotated);
            *reinterpret_cast<Bf16x8*>(
                args.sparse_query + token_head * kSparseQueryDim +
                (kSparseQueryDim - kPeDim) + pe0) =
                pack_bf16x8(rotated);
          }
        }
      }
      cutlass::arch::NamedBarrier::sync(
          128, store_group * 2 + 1);
      if constexpr (kEpilogueWarpgroups == 1) {
        store_stage = (store_stage + 1) % kNumTMAStoreStages;
      }
    }
    if (!released_tmem) {
      deep_gemm::ptx::tcgen05_before_thread_sync();
      empty_barrier->arrive(0u);
    }
  }
};

template <int kInputDim, int kHeads, int kHeadDim, int kNopeDim,
          int kPeDim, int kSparseQueryDim, int kBlockM, int kBlockN,
          int kStages, int kWorkers, bool kSwapAb,
          int kEpilogueWarpgroups>
CUTLASS_GLOBAL void __launch_bounds__(
    128 * (1 + kEpilogueWarpgroups), 1) q_b_proj_impl(
    std::uint32_t rows,
    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
    const __grid_constant__ cute::TmaDescriptor tensor_map_cd,
    const float* cos_sin,
    const std::int32_t* positions,
    __nv_bfloat16* q_nope,
    __nv_bfloat16* sparse_query) {
  Args args{cos_sin, positions, q_nope, sparse_query, static_cast<int>(rows),
            kHeads};
  using Config = deep_gemm::Fp8GemmConfig<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      32, 32, 128,
      0, kHeads * kHeadDim, kInputDim,
      kBlockM, kBlockN, 128, 1,
      128, 128, kSwapAb ? 128 : 64, kStages,
      128, 128 * kEpilogueWarpgroups, 2, kSwapAb,
      Epilogue<kHeadDim, kNopeDim, kPeDim, kSparseQueryDim,
               kEpilogueWarpgroups>,
      true, true, kWorkers,
      kSwapAb, true, deep_gemm::GemmType::Normal, false,
      cutlass::float_e4m3_t, cutlass::float_e4m3_t,
      cutlass::bfloat16_t,
      deep_gemm::epilogue::transform::EpilogueIdentity>;
  Config::template run<deep_gemm::NoAuxiliaryRole>(
      nullptr, &args, deep_gemm::NoAuxiliaryRole::Arguments{},
      rows, kHeads * kHeadDim, kInputDim,
      tensor_map_a, tensor_map_b, tensor_map_sfa, tensor_map_sfb,
      tensor_map_cd);
}

}  // namespace hy4_qb
