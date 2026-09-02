#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>

namespace rtp_llm {

void invokeFlashMlaStateMergeSegmented(const at::Tensor& output,
                                       const at::Tensor& output_lse,
                                       const at::Tensor& partial_output,
                                       const at::Tensor& partial_lse,
                                       const at::Tensor& partial_q_indptr,
                                       const at::Tensor& destination_starts,
                                       cudaStream_t      stream);

}  // namespace rtp_llm
