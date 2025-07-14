#pragma once

#include <assert.h>
#include <vector>

#if USEING_CUDA
#include <cuda_runtime.h>
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

void InvokeMultiCopyKernel(const std::vector<void*>&  h_src_ptrs,
                           const std::vector<void*>&  h_dst_ptrs,
                           const std::vector<size_t>& h_copy_sizes,
                           cudaStream_t               stream);

void InvokeMultiMergeCopyKernel(const void*                h_dst_ptr,
                                const std::vector<void*>&  h_src_ptrs,
                                const std::vector<size_t>& h_copy_sizes,
                                const std::vector<size_t>& h_begin_offsets,
                                cudaStream_t               stream);

}  // namespace rtp_llm
