#pragma once

#include <ATen/ATen.h>
#if USING_ROCM
#include <hip/hip_runtime.h>
#elif USING_CUDA
#include <cuda_runtime_api.h>
#else
#error "attention_input_metadata requires CUDA or ROCm"
#endif

namespace rtp_llm {

#if USING_ROCM
using AttentionInputMetadataStream = hipStream_t;
#elif USING_CUDA
using AttentionInputMetadataStream = cudaStream_t;
#else
#error "attention_input_metadata requires CUDA or ROCm"
#endif

void invokeBuildAttentionInputMetadata(const at::Tensor&            input_lengths,
                                       const at::Tensor&            prefix_lengths,
                                       at::Tensor&                  cu_seqlens,
                                       at::Tensor&                  cu_kv_seqlens,
                                       at::Tensor&                  padding_offset,
                                       AttentionInputMetadataStream stream);

}  // namespace rtp_llm
