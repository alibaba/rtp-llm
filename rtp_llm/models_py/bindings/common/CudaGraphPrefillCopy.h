#pragma once
#include <torch/extension.h>

#if USING_CUDA
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#endif

namespace torch_ext {

// CUDA Graph copy functions
// CUDA implementation: rtp_llm/models_py/bindings/cuda/CudaGraphPrefillCopy.cc
// ROCm stub implementation: rtp_llm/models_py/bindings/rocm/CudaGraphPrefillCopy.cc
void cuda_graph_copy_small2large(at::Tensor& input_tensor,
                                 at::Tensor& output_tensor,
                                 at::Tensor& batch_size,
                                 int64_t     max_batch_size,
                                 int64_t     max_seq_len,
                                 at::Tensor& input_lengths,
                                 int64_t     hidden_size,
                                 at::Tensor& cu_seq_len);

void cuda_graph_copy_large2small(at::Tensor& input_tensor,
                                 at::Tensor& output_tensor,
                                 at::Tensor& batch_size,
                                 int64_t     max_batch_size,
                                 int64_t     max_seq_len,
                                 at::Tensor& input_lengths,
                                 int64_t     hidden_size,
                                 at::Tensor& cu_seq_len);

}  // namespace torch_ext
