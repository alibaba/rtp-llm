#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

namespace torch_ext {

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
