#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace rtp_llm::crc_copy_benchmark::main_baseline::sDevMPS {

void launch_dsv4_memory_cache_gather_copy_var_nooffset(const void**  src_ptrs,
                                                       const size_t* sizes,
                                                       const size_t* dst_offsets,
                                                       void*         dst,
                                                       int           num_srcs,
                                                       int           block_num,
                                                       cudaStream_t  stream);
void launch_dsv4_memory_cache_scatter_copy_var_nooffset(const void*   src,
                                                        const size_t* src_offsets,
                                                        const size_t* sizes,
                                                        void**        dst_ptrs,
                                                        int           num_dsts,
                                                        int           block_num,
                                                        cudaStream_t  stream);

}  // namespace rtp_llm::crc_copy_benchmark::main_baseline::sDevMPS
