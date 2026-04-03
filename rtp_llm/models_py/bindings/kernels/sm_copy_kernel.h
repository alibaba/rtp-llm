#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <chrono>
#include <cmath>

// 支持N卡 & PPU
// COPY FROM tair mempool (pace)，就不改成RTP的namespace了，保留原namespace
// RTP只需要gather和scatter操作
//
// Var-length *_var_nooffset* (gather_copy_var_nooffset_kernel / scatter_copy_var_nooffset_kernel, their
// launch_* and warmup_sm_copy_var_nooffset_kernels): not referenced by production inference here.
// Split KV: warmup_sm_copy_split_kernels in initExecCtx; launch_*_copy_split from execNoBlockCopy (split KV path).
// Remaining entry points stay in the library for
// standalone microbenchmarks (e.g. rtp_llm/cpp/devices/cuda_impl/tests/sm_copy_kernel_benchmark.cc via nvcc
// build scripts; kernel lives under models_py/bindings/kernels): A/B split staging vs var-length device scatter, and
// possible future non-split layouts. Do not add call sites without a real consumer; otherwise prefer YAGNI and drop or
// split into a benchmark-only target.

namespace sDevMPS {
/**
 * @brief Scatter from contiguous src to num_dsts pairs (dst_kv_cache[i], dst_kv_scale[i]).
 * Src layout: [kv0_cache, kv0_scale, kv1_cache, kv1_scale, ...]; stride = kv_cache_size + kv_scale_size per dst.
 */
void launch_scatter_copy_split(const void*  src,
                               void**       dst_kv_cache_ptrs,
                               void**       dst_kv_scale_ptrs,
                               size_t       kv_cache_size,
                               size_t       kv_scale_size,
                               int          num_dsts,
                               int          block_num,
                               cudaStream_t stream);

/**
 * @brief Gather from num_srcs pairs (src_kv_cache[i], src_kv_scale[i]) to contiguous dst.
 * Dst layout: [kv0_cache, kv0_scale, kv1_cache, kv1_scale, ...].
 */
void launch_gather_copy_split(const void** src_kv_cache_ptrs,
                              const void** src_kv_scale_ptrs,
                              size_t       kv_cache_size,
                              size_t       kv_scale_size,
                              void*        dst,
                              int          num_srcs,
                              int          block_num,
                              cudaStream_t stream);

/**
 * @brief JIT-load split KV/scale gather+scatter kernels on \p stream (e.g. before NCCL init).
 * @return true on success; frees temp allocations before return.
 */
bool warmup_sm_copy_split_kernels(cudaStream_t stream);

/**
 * Optional JIT warmup for *_var_nooffset kernels only (not used on the production split-KV path).
 * Intended for sm_copy_kernel_benchmark / manual profiling; not invoked by \ref warmup_sm_copy_split_kernels.
 */
bool warmup_sm_copy_var_nooffset_kernels(cudaStream_t stream);

/**
 * @brief 启动一个 CUDA 内核，执行变长 Gather-Copy（每个源的 GPU 指针已在 Host 端预偏移到数据起点）。
 *
 * @note No in-tree production caller today; see file header for intended use (benchmark / experiments).
 *
 * 调用方在 Host 端预先将每条源的 device 指针指到实际数据起点，Kernel 内不再对 src 做 offset 加法，
 * 从而减少参数与地址计算。
 *
 * 数据流：
 *   src_ptrs[i] --[sizes[i] bytes]--> dst + dst_offsets[i]
 *
 * 约定：
 *   - src_ptrs[i] 已指向本段拷贝起点（例如 base_ptr[i] + page_byte_offset）
 *   - dst_offsets 仍为设备端前缀和，指明写入 contiguous dst 的各段起点
 *
 * 典型应用场景：当调用方可以在 Host 端方便地预计算偏移时，使用此优化版本可减少 Kernel 开销。
 *
 * @param src_ptrs      指向源内存指针的数组，长度为 num_srcs。每个元素是一个 void*，指向一个源缓冲区。
 *                      **注意**：每个指针应已预先偏移到实际数据的起始位置（即 src_ptrs[i] = base_ptr[i] + offset[i]）。
 *                      这个指针数组，且里面每个元素指向的地址，均需 SM 可访问。
 * @param sizes         每个源要拷贝的字节数数组（设备端），长度为 num_srcs。sizes[i] 表示从第 i 个源
 *                      拷贝的字节数。
 * @param dst_offsets   目标内存中的前缀和偏移量数组（设备端），长度为 num_srcs。dst_offsets[i] 表示
 *                      第 i 个源的数据在目标内存中的起始位置。通常由 Host 端预先计算：
 *                      dst_offsets[0] = 0, dst_offsets[i] = dst_offsets[i-1] + sizes[i-1] (i > 0)
 * @param dst           目标内存的起始地址。必须已分配足够空间（sum(sizes) 字节），该地址需 SM 可访问。
 * @param num_srcs      源缓冲区的数量，即 src_ptrs 数组的长度。必须 >= 0
 * @param block_num     Block 数，用于控制使用的 SM 数，传入值为 0 时默认使用所有的 SM
 * @param stream        CUDA 流，用于异步执行。0 表示使用默认流
 *
 * @note 前缀和计算应在 Host 端完成，避免 Kernel 内部的全局同步开销。
 * @note 调用方需在 Host 端预计算偏移后的指针数组，并将偏移后的指针拷贝到 Device。
 * @see launch_scatter_copy_var_nooffset 逆向 Scatter（同源 contiguous / 多 dest 指针预偏移）
 */
void launch_gather_copy_var_nooffset(const void**  src_ptrs,
                                     const size_t* sizes,
                                     const size_t* dst_offsets,
                                     void*         dst,
                                     int           num_srcs,
                                     int           block_num,
                                     cudaStream_t  stream);

/**
 * @brief 启动一个 CUDA 内核，执行变长 Scatter-Copy（每个目标的 GPU 指针已在 Host 端预偏移到写入起点）。
 *
 * @note No in-tree production caller today; see file header for intended use (benchmark / experiments).
 *
 * 源一侧仍通过 src + src_offsets[i] 定位各段；各 dst_ptrs[i] 已指向本段写入起点，Kernel 内不对 dst 做 offset 加法。
 *
 * 数据流：
 *   src + src_offsets[i] --[sizes[i] bytes]--> dst_ptrs[i]
 *
 * 约定：
 *   - dst_ptrs[i] 已指向本段写入起点
 *   - src_offsets 为设备端前缀和，指明读自 contiguous src 的各段起点
 *
 * 典型应用场景：当调用方可以在 Host 端方便地预计算偏移时，使用此优化版本可减少 Kernel 开销。
 *
 * @param src           源内存的起始地址。必须包含至少 sum(sizes) 字节的有效数据，该地址需 SM 可访问。
 * @param src_offsets   源内存中的前缀和偏移量数组（设备端），长度为 num_dsts。src_offsets[i] 表示
 *                      第 i 个目标的数据在源内存中的起始位置。通常由 Host 端预先计算：
 *                      src_offsets[0] = 0, src_offsets[i] = src_offsets[i-1] + sizes[i-1] (i > 0)
 * @param sizes         每个目标要拷贝的字节数数组（设备端），长度为 num_dsts。sizes[i] 表示拷贝到
 *                      第 i 个目标的字节数。
 * @param dst_ptrs      指向目标内存指针的数组，长度为 num_dsts。每个元素是一个 void*，指向一个目标缓冲区。
 *                      **注意**：每个指针应已预先偏移到实际写入的起始位置（即 dst_ptrs[i] = base_ptr[i] + offset[i]）。
 *                      这个指针数组，且里面每个元素指向的地址，均需 SM 可访问。
 * @param num_dsts      目标缓冲区的数量，即 dst_ptrs 数组的长度。必须 >= 0。
 * @param block_num     Block 数，用于控制使用的 SM 数，传入值为 0 时默认使用所有的 SM
 * @param stream        CUDA 流，用于异步执行。0 表示使用默认流
 *
 * @note 前缀和计算应在 Host 端完成，避免 Kernel 内部的全局同步开销。
 * @note 调用方需在 Host 端预计算偏移后的指针数组，并将偏移后的指针拷贝到 Device。
 * @note 与 launch_gather_copy_var_nooffset 互为逆操作（在相同参数下）。
 * @see launch_gather_copy_var_nooffset 逆向 Gather（多源指针预偏移 / 同一 contiguous dst）
 */
void launch_scatter_copy_var_nooffset(const void*   src,
                                      const size_t* src_offsets,
                                      const size_t* sizes,
                                      void**        dst_ptrs,
                                      int           num_dsts,
                                      int           block_num,
                                      cudaStream_t  stream);

}  // namespace sDevMPS