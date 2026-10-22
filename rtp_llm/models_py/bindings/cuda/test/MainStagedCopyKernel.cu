// Test-only kernel snapshot from main e73826e27a6005640bb75ce2a73f235bfa51c55b.
#include "rtp_llm/models_py/bindings/cuda/test/MainStagedCopyKernel.h"

#include <cstdint>

namespace rtp_llm::crc_copy_benchmark::main_baseline::sDevMPS {

#define THREADS_PER_BLOCK 512

template<typename T>
__device__ __forceinline__ void dsv4_copy_typed_region(
    const char* src_base, char* dst_base, size_t region_size, const size_t tid, const size_t num_threads) {
    const size_t vec_bytes   = sizeof(T);
    const size_t vec_num     = region_size / vec_bytes;
    const size_t vec_payload = vec_num * vec_bytes;

    size_t element_idx = tid;
#pragma unroll 4
    while (element_idx < vec_num) {
        reinterpret_cast<T*>(dst_base)[element_idx] = reinterpret_cast<const T*>(src_base)[element_idx];
        element_idx += num_threads;
    }

    for (size_t byte_idx = vec_payload + tid; byte_idx < region_size; byte_idx += num_threads) {
        dst_base[byte_idx] = src_base[byte_idx];
    }
}

__device__ __forceinline__ void
dsv4_copy_region(const char* src_base, char* dst_base, size_t region_size, const size_t tid, const size_t num_threads) {
    const auto addr_mask = reinterpret_cast<uintptr_t>(src_base) | reinterpret_cast<uintptr_t>(dst_base);
    if ((addr_mask & (sizeof(int4) - 1)) == 0) {
        dsv4_copy_typed_region<int4>(src_base, dst_base, region_size, tid, num_threads);
        return;
    }
    if ((addr_mask & (sizeof(int2) - 1)) == 0) {
        dsv4_copy_typed_region<int2>(src_base, dst_base, region_size, tid, num_threads);
        return;
    }
    if ((addr_mask & (sizeof(unsigned int) - 1)) == 0) {
        dsv4_copy_typed_region<unsigned int>(src_base, dst_base, region_size, tid, num_threads);
        return;
    }
    if ((addr_mask & (sizeof(unsigned short) - 1)) == 0) {
        dsv4_copy_typed_region<unsigned short>(src_base, dst_base, region_size, tid, num_threads);
        return;
    }

    for (size_t byte_idx = tid; byte_idx < region_size; byte_idx += num_threads) {
        dst_base[byte_idx] = src_base[byte_idx];
    }
}

__global__ void
dsv4_memory_cache_gather_copy_var_nooffset_kernel(const void**  src_ptrs,
                                                  const size_t* sizes,        // 每个源要拷贝的字节数
                                                  const size_t* dst_offsets,  // 目标内存中的前缀和偏移量
                                                  void*         dst,
                                                  int           num_srcs) {
    if (blockIdx.x >= num_srcs)
        return;
    const size_t tid         = threadIdx.x;
    const size_t num_threads = blockDim.x;

    for (int src_idx = blockIdx.x; src_idx < num_srcs; src_idx += gridDim.x) {
        const size_t cur_size       = sizes[src_idx];
        const size_t cur_dst_offset = dst_offsets[src_idx];

        const char* src_base = reinterpret_cast<const char*>(src_ptrs[src_idx]);
        char*       dst_base = reinterpret_cast<char*>(dst) + cur_dst_offset;

        dsv4_copy_region(src_base, dst_base, cur_size, tid, num_threads);
    }
}

__global__ void dsv4_memory_cache_scatter_copy_var_nooffset_kernel(const void*   src,
                                                                   const size_t* src_offsets,  // 源内存中的前缀和偏移量
                                                                   const size_t* sizes,        // 每个目标要拷贝的字节数
                                                                   void**        dst_ptrs,
                                                                   int           num_dsts) {
    if (blockIdx.x >= num_dsts)
        return;
    const size_t tid         = threadIdx.x;
    const size_t num_threads = blockDim.x;

    for (int dst_idx = blockIdx.x; dst_idx < num_dsts; dst_idx += gridDim.x) {
        const size_t cur_size       = sizes[dst_idx];
        const size_t cur_src_offset = src_offsets[dst_idx];

        const char* src_base = reinterpret_cast<const char*>(src) + cur_src_offset;
        char*       dst_base = reinterpret_cast<char*>(dst_ptrs[dst_idx]);

        dsv4_copy_region(src_base, dst_base, cur_size, tid, num_threads);
    }
}

void launch_dsv4_memory_cache_gather_copy_var_nooffset(
    const void**  src_ptrs,     // 源指针数组（每个指针已指向实际数据起始位置）
    const size_t* sizes,        // 每个源要拷贝的字节数数组
    const size_t* dst_offsets,  // 目标内存中的前缀和偏移量数组
    void*         dst,          // 目标内存起始地址
    int           num_srcs,     // 源数量
    int           block_num,
    cudaStream_t  stream) {
    if (block_num == 0) {
        block_num = num_srcs;
    }
    dsv4_memory_cache_gather_copy_var_nooffset_kernel<<<block_num, THREADS_PER_BLOCK, 0, stream>>>(
        src_ptrs, sizes, dst_offsets, dst, num_srcs);
}

void launch_dsv4_memory_cache_scatter_copy_var_nooffset(
    const void*   src,          // 源内存起始地址
    const size_t* src_offsets,  // 源内存中的前缀和偏移量数组
    const size_t* sizes,        // 每个目标要拷贝的字节数数组
    void**        dst_ptrs,     // 目标指针数组（每个指针已指向实际写入起始位置）
    int           num_dsts,     // 目标数量
    int           block_num,
    cudaStream_t  stream) {
    if (block_num == 0) {
        block_num = num_dsts;
    }
    dsv4_memory_cache_scatter_copy_var_nooffset_kernel<<<block_num, THREADS_PER_BLOCK, 0, stream>>>(
        src, src_offsets, sizes, dst_ptrs, num_dsts);
}

#undef THREADS_PER_BLOCK

}  // namespace rtp_llm::crc_copy_benchmark::main_baseline::sDevMPS
