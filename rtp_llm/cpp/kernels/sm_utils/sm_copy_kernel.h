#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <chrono>
#include <cmath>

// 支持N卡 & PPU
// COPY FROM tair mempool (pace)，就不改成RTP的namespace了，保留原namespace
// RTP只需要gather和scatter操作

namespace sDevMPS {
/**
 * @brief 启动一个 CUDA 内核，执行 Gather-Copy 操作。
 *
 * 该函数将多个源内存区域（每个由 src_ptrs[i] 指向）中相同偏移 offset 处、长度为 size 字节的数据，
 * 依次复制到连续的目标内存 dst 中。最终 dst 中的数据布局为：
 * [src_ptrs[0][offset:offset+size], src_ptrs[1][offset:offset+size], ..., src_ptrs[num_srcs-1][...]]
 *
 * 典型应用场景：从多个分散的缓冲区中提取相同位置的数据块，合并成一个连续缓冲区（如多个Layer的KVCache等）。
 *
 * @param src_ptrs      指向源内存指针的数组，长度为 num_srcs。每个元素是一个 void*，指向一个源缓冲区。
 *                      注意：这个指针数组，且里面每个元素指向的地址，均需SM可访问。
 * @param offset        每个源缓冲区中数据起始的统一偏移量（以字节为单位）。所有源都从 offset 处开始读取。
 * @param size          每个源要复制的数据大小（字节数）。所有源复制相同长度。
 * @param dst           目标内存的起始地址。必须已分配足够空间（num_srcs * size字节），该地址需SM可访问。
 * @param num_srcs      源缓冲区的数量，即 src_ptrs 数组的长度。必须 >= 0
 * @param block_num     Block数，用于控制使用的SM数，传入值为0时默认使用所有的SM
 * @param stream        CUDA 流，用于异步执行。0 表示使用默认流
 *
 */
void launch_gather_copy(
    const void** src_ptrs, size_t offset, size_t size, void* dst, int num_srcs, int block_num, cudaStream_t stream);

/**
 * @brief 启动一个 CUDA 内核，执行 Scatter-Copy 操作。
 *
 * 该函数将连续源内存 src 中的数据，按块分割后分别复制到多个目标内存区域（由 dst_ptrs[i] 指向）中，
 * 每个目标从其 offset 偏移处开始写入 size 字节。数据布局为：
 * src[0:size] -> dst_ptrs[0][offset:offset+size]
 * src[size:2*size] -> dst_ptrs[1][offset:offset+size]
 * ...
 *
 * 典型应用场景：将一个大缓冲区的数据分发到多个独立缓冲区。
 *
 * @param src           源内存的起始地址。必须包含至少 num_dsts * size 字节的有效数据，该地址需SM可访问。
 * @param offset        每个目标缓冲区中写入起始的统一偏移量（字节）。所有目标都从 offset 处开始写入。
 * @param size          每个目标要写入的数据大小（字节）。所有目标写入相同长度。
 * @param dst_ptrs      指向目标内存指针的数组，长度为 num_dsts。每个元素是一个 void*，指向一个目标缓冲区。
 *                      注意：这个指针数组，且里面每个元素指向的地址，均需SM可访问。
 * @param num_dsts      目标缓冲区的数量，即 dst_ptrs 数组的长度。必须 >= 0。
 * @param block_num     Block数，用于控制使用的SM数，传入值为0时默认使用所有的SM
 * @param stream        CUDA 流，用于异步执行。0 表示使用默认流
 *
 * @note 与 launch_gather_copy 互为逆操作（在相同参数下）。
 * @note 要求所有目标缓冲区在 [offset, offset + size) 范围内不重叠，否则行为未定义。
 */
void launch_scatter_copy(
    const void* src, size_t offset, size_t size, void** dst_ptrs, int num_dsts, int block_num, cudaStream_t stream);

}  // namespace sDevMPS