#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

namespace rtp_llm {

void invokeBuildAttentionInputMetadata(const at::Tensor& input_lengths,
                                       const at::Tensor& prefix_lengths,
                                       at::Tensor&       cu_seqlens,
                                       at::Tensor&       cu_kv_seqlens,
                                       at::Tensor&       padding_offset,
                                       cudaStream_t      stream);

}  // namespace rtp_llm
