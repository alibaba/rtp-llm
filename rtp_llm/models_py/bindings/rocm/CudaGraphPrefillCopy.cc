#include "rtp_llm/models_py/bindings/rocm/CudaGraphPrefillCopy.h"

namespace torch_ext {

// Stub implementations for CUDA Graph copy functions in ROCm environment
// These are empty no-op stubs since CUDA Graph is CUDA-specific

void cuda_graph_copy_small2large(at::Tensor& input_tensor,
                                 at::Tensor& output_tensor,
                                 at::Tensor& batch_size,
                                 int64_t     max_batch_size,
                                 int64_t     max_seq_len,
                                 at::Tensor& input_lengths,
                                 int64_t     hidden_size,
                                 at::Tensor& cu_seq_len) {
    // Empty stub implementation for ROCm
    // CUDA Graph is not supported in ROCm environment
}

void cuda_graph_copy_large2small(at::Tensor& input_tensor,
                                 at::Tensor& output_tensor,
                                 at::Tensor& batch_size,
                                 int64_t     max_batch_size,
                                 int64_t     max_seq_len,
                                 at::Tensor& input_lengths,
                                 int64_t     hidden_size,
                                 at::Tensor& cu_seq_len) {
    // Empty stub implementation for ROCm
    // CUDA Graph is not supported in ROCm environment
}

}  // namespace torch_ext
