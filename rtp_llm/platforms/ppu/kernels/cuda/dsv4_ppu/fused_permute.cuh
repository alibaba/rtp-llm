#pragma once

#include <cuda_pipeline.h>
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm80.h"

// Fused permute(1,0,2) for a paired (A, SFA) tensor set.
// Optimization: cp.async (async prefetch to SMEM) + cache-global stores (ppu.st.global.cg).
// Single-buffered: load full tile, then store full tile.
// Supports non-16-byte-aligned dim2 via vectorized path + scalar tail.

namespace deep_gemm {

using permute_vec_t = __attribute__((__vector_size__(16))) char;

__device__ __forceinline__ void cp_async_cg_16(void* dst, const void* src) {
    cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(dst, src);
}

__device__ __forceinline__ void st_global_cg_16(void* dst, const permute_vec_t& val) {
    const int4* v = reinterpret_cast<const int4*>(&val);
    asm volatile("ppu.st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};\n" ::
                 "l"(dst), "r"(v->x), "r"(v->y), "r"(v->z), "r"(v->w));
}

__device__ __forceinline__ void st_global_cg_4(void* dst, int val) {
    asm volatile("ppu.st.global.cg.b32 [%0], %1;\n" :: "l"(dst), "r"(val));
}

__device__ __forceinline__ long long a_src_offset(int d0, int d1, int dim1, int dim2_a, int d2_vec, int vec_size) {
    return ((long long)d0 * dim1 + d1) * dim2_a + d2_vec * vec_size;
}

__device__ __forceinline__ long long a_dst_offset(int d1, int d0, int dim0, int dim2_a, int d2_vec, int vec_size) {
    return (long long)d1 * dim0 * dim2_a + (long long)d0 * dim2_a + d2_vec * vec_size;
}

__device__ __forceinline__ long long sfa_src_offset(int d0, int d1, int dim1, int dim2_sfa_bytes, int byte_offset) {
    return ((long long)d0 * dim1 + d1) * dim2_sfa_bytes + byte_offset;
}

__device__ __forceinline__ long long sfa_dst_offset(int d1, int d0, int dim0, int dim2_sfa_bytes, int byte_offset) {
    return (long long)d1 * dim0 * dim2_sfa_bytes + (long long)d0 * dim2_sfa_bytes + byte_offset;
}

template <int VEC_SIZE, int THREADS_PER_BLOCK, int ROWS_PER_ITER, int DIM1, int NUM_D2_VECS_A, int TAIL_BYTES_A, int DIM2_SFA_BYTES, int D1_TILE, bool USE_FAST_PATH>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_permute_kernel(
    const char* __restrict__ src_a,
    char* __restrict__       dst_a,
    const char* __restrict__ src_sfa,
    char* __restrict__       dst_sfa,
    int dim0, int dim2_a)
{
    const int d0_base = blockIdx.y * ROWS_PER_ITER;

    // ===== Tensor A =====
    if constexpr (USE_FAST_PATH) {
        constexpr int D2_BLOCKS = (NUM_D2_VECS_A + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const int d1 = blockIdx.x / D2_BLOCKS;
        const int d2_block = blockIdx.x - d1 * D2_BLOCKS;
        const int d2_vec = d2_block * THREADS_PER_BLOCK + threadIdx.x;

        // Vectorized 16-byte path
        if (d2_vec < NUM_D2_VECS_A) {
            #pragma unroll
            for (int d0_offset = 0; d0_offset < ROWS_PER_ITER; ++d0_offset) {
                int d0 = d0_base + d0_offset;
                if (d0 < dim0) {
                    long long src_off = a_src_offset(d0, d1, DIM1, dim2_a, d2_vec, VEC_SIZE);
                    long long dst_off = a_dst_offset(d1, d0, dim0, dim2_a, d2_vec, VEC_SIZE);
                    permute_vec_t val;
                    __builtin_memcpy(&val, src_a + src_off, VEC_SIZE);
                    st_global_cg_16(dst_a + dst_off, val);
                }
            }
        }

        // Scalar tail for non-16-byte-aligned dim2_a_bytes
        if constexpr (TAIL_BYTES_A > 0) {
            if (threadIdx.x < TAIL_BYTES_A) {
                int tail_byte_offset = NUM_D2_VECS_A * VEC_SIZE + threadIdx.x;
                #pragma unroll
                for (int d0_offset = 0; d0_offset < ROWS_PER_ITER; ++d0_offset) {
                    int d0 = d0_base + d0_offset;
                    if (d0 < dim0) {
                        long long src_off = ((long long)d0 * DIM1 + d1) * dim2_a + tail_byte_offset;
                        long long dst_off = (long long)d1 * dim0 * dim2_a + (long long)d0 * dim2_a + tail_byte_offset;
                        char val = src_a[src_off];
                        dst_a[dst_off] = val;
                    }
                }
            }
        }
    } else {
        // SMEM transpose path: cp.async + cache-global stores.
        constexpr int TILE_VECS = ROWS_PER_ITER * D1_TILE * NUM_D2_VECS_A;
        constexpr int LOAD_ITERS = (TILE_VECS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        constexpr int VECS_PER_SLICE = ROWS_PER_ITER * NUM_D2_VECS_A;
        constexpr int STORE_ITERS = (VECS_PER_SLICE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        constexpr int VECS_PER_TILE_ROW = D1_TILE * NUM_D2_VECS_A;
        extern __shared__ char smem[];

        const int d1_group = blockIdx.x;
        const int d1_start = d1_group * D1_TILE;

        #pragma unroll
        for (int i = 0; i < LOAD_ITERS; ++i) {
            int vec_idx = i * THREADS_PER_BLOCK + threadIdx.x;
            if (vec_idx < TILE_VECS) {
                int d0_offset = vec_idx / VECS_PER_TILE_ROW;
                int rem = vec_idx - d0_offset * VECS_PER_TILE_ROW;
                int d1_local = rem / NUM_D2_VECS_A;
                int d2_vec = rem - d1_local * NUM_D2_VECS_A;
                int d0 = d0_base + d0_offset;
                if (d0 < dim0) {
                    int d1 = d1_start + d1_local;
                    long long src_off = a_src_offset(d0, d1, DIM1, dim2_a, d2_vec, VEC_SIZE);
                    cp_async_cg_16(smem + vec_idx * VEC_SIZE, src_a + src_off);
                }
            }
        }
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();

        #pragma unroll
        for (int d1_local = 0; d1_local < D1_TILE; ++d1_local) {
            int d1 = d1_start + d1_local;
            #pragma unroll
            for (int i = 0; i < STORE_ITERS; ++i) {
                int store_vec_idx = i * THREADS_PER_BLOCK + threadIdx.x;
                if (store_vec_idx < VECS_PER_SLICE) {
                    int d0_offset = store_vec_idx / NUM_D2_VECS_A;
                    int d2_vec = store_vec_idx - d0_offset * NUM_D2_VECS_A;
                    int d0 = d0_base + d0_offset;
                    if (d0 < dim0) {
                        int smem_idx = (d0_offset * D1_TILE + d1_local) * NUM_D2_VECS_A + d2_vec;
                        permute_vec_t val;
                        __builtin_memcpy(&val, smem + smem_idx * VEC_SIZE, VEC_SIZE);
                        long long dst_off = a_dst_offset(d1, d0, dim0, dim2_a, d2_vec, VEC_SIZE);
                        st_global_cg_16(dst_a + dst_off, val);
                    }
                }
            }
        }

        // Scalar tail for non-16-byte-aligned dim2_a_bytes (SMEM path)
        if constexpr (TAIL_BYTES_A > 0) {
            if (threadIdx.x < TAIL_BYTES_A) {
                int tail_byte_offset = NUM_D2_VECS_A * VEC_SIZE + threadIdx.x;
                #pragma unroll
                for (int d1_local = 0; d1_local < D1_TILE; ++d1_local) {
                    int d1 = d1_start + d1_local;
                    #pragma unroll
                    for (int d0_offset = 0; d0_offset < ROWS_PER_ITER; ++d0_offset) {
                        int d0 = d0_base + d0_offset;
                        if (d0 < dim0) {
                            long long src_off = ((long long)d0 * DIM1 + d1) * dim2_a + tail_byte_offset;
                            long long dst_off = (long long)d1 * dim0 * dim2_a + (long long)d0 * dim2_a + tail_byte_offset;
                            char val = src_a[src_off];
                            dst_a[dst_off] = val;
                        }
                    }
                }
            }
        }
    }


    // ===== SFA: direct load/store =====
    constexpr int SFA_VEC = (DIM2_SFA_BYTES >= 16) ? 16 : DIM2_SFA_BYTES;
    constexpr int VECS_PER_D1_SFA = DIM2_SFA_BYTES / SFA_VEC;
    auto copy_sfa_row = [&](int d1_s) {
        if constexpr (DIM2_SFA_BYTES % 16 == 0) {
            if (threadIdx.x * 16 >= DIM2_SFA_BYTES) return;
            int byte_offset = threadIdx.x * 16;
            #pragma unroll
            for (int d0_offset = 0; d0_offset < ROWS_PER_ITER; ++d0_offset) {
                int d0 = d0_base + d0_offset;
                if (d0 < dim0) {
                    long long src_off = sfa_src_offset(d0, d1_s, DIM1, DIM2_SFA_BYTES, byte_offset);
                    long long dst_off = sfa_dst_offset(d1_s, d0, dim0, DIM2_SFA_BYTES, byte_offset);
                    permute_vec_t val;
                    __builtin_memcpy(&val, src_sfa + src_off, 16);
                    st_global_cg_16(dst_sfa + dst_off, val);
                }
            }
        } else {
            if (threadIdx.x * 4 >= DIM2_SFA_BYTES) return;
            int byte_offset = threadIdx.x * 4;
            #pragma unroll
            for (int d0_offset = 0; d0_offset < ROWS_PER_ITER; ++d0_offset) {
                int d0 = d0_base + d0_offset;
                if (d0 < dim0) {
                    long long src_off = sfa_src_offset(d0, d1_s, DIM1, DIM2_SFA_BYTES, byte_offset);
                    long long dst_off = sfa_dst_offset(d1_s, d0, dim0, DIM2_SFA_BYTES, byte_offset);
                    int val;
                    __builtin_memcpy(&val, src_sfa + src_off, 4);
                    st_global_cg_4(dst_sfa + dst_off, val);
                }
            }
        }
    };

    if constexpr (USE_FAST_PATH) {
        constexpr int D2_BLOCKS = (NUM_D2_VECS_A + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (blockIdx.x % D2_BLOCKS == 0) {
            copy_sfa_row(blockIdx.x / D2_BLOCKS);
        }
    } else {
        constexpr int TOTAL_VECS_SFA = D1_TILE * VECS_PER_D1_SFA;
        const int d1_start = blockIdx.x * D1_TILE;
        if (threadIdx.x < TOTAL_VECS_SFA) {
            int d1_local = threadIdx.x / VECS_PER_D1_SFA;
            int local_vec_s = threadIdx.x - d1_local * VECS_PER_D1_SFA;
            int d1_s = d1_start + d1_local;
            int byte_offset = local_vec_s * SFA_VEC;

            #pragma unroll
            for (int d0_offset = 0; d0_offset < ROWS_PER_ITER; ++d0_offset) {
                int d0 = d0_base + d0_offset;
                if (d0 < dim0) {
                    long long src_off = sfa_src_offset(d0, d1_s, DIM1, DIM2_SFA_BYTES, byte_offset);
                    long long dst_off = sfa_dst_offset(d1_s, d0, dim0, DIM2_SFA_BYTES, byte_offset);
                    if constexpr (SFA_VEC == 16) {
                        permute_vec_t val;
                        __builtin_memcpy(&val, src_sfa + src_off, 16);
                        st_global_cg_16(dst_sfa + dst_off, val);
                    } else {
                        int val;
                        __builtin_memcpy(&val, src_sfa + src_off, 4);
                        st_global_cg_4(dst_sfa + dst_off, val);
                    }
                }
            }
        }
    }
}

}  // namespace deep_gemm
