#pragma once

#include <cuda_runtime_api.h>
#include <torch/torch.h>
#include "rtp_llm/cpp/cuda/deep_gemm/utils.h"

namespace rtp_llm {

class DeepGemmPlugin {
public:
    static size_t getPaddingSize(size_t m, DeepGemmType gemm_type);
    static size_t paddingMasked(const size_t& token_num);

    static void gemmFp8(const torch::Tensor& lhs_kernel,
                        const torch::Tensor& lhs_scales,
                        const torch::Tensor& rhs_kernel,
                        const torch::Tensor& rhs_scales,
                        torch::Tensor&       output,
                        int                  user_deep_gemm_num_sm,
                        cudaStream_t         stream);
    static void groupedGemmFp8Contiguous(const torch::Tensor& lhs_kernel,
                                         const torch::Tensor& lhs_scales,
                                         const torch::Tensor& rhs_kernel,
                                         const torch::Tensor& rhs_scales,
                                         torch::Tensor&       output,
                                         const torch::Tensor& m_indices,
                                         int                  user_deep_gemm_num_sm,
                                         bool                 use_64_padding,
                                         cudaStream_t         stream);
    static void groupedGemmFp8Masked(const torch::Tensor& lhs_kernel,
                                     const torch::Tensor& lhs_scales,
                                     const torch::Tensor& rhs_kernel,
                                     const torch::Tensor& rhs_scales,
                                     torch::Tensor&       output,
                                     const torch::Tensor& masked_m,
                                     int                  expected_m,
                                     int                  user_deep_gemm_num_sm,
                                     cudaStream_t         stream);
    static void groupedGemmFp8Masked_V2(const torch::Tensor& lhs_kernel,
                                        const torch::Tensor& lhs_scales,
                                        const torch::Tensor& rhs_kernel,
                                        const torch::Tensor& rhs_scales,
                                        torch::Tensor&       output,
                                        const torch::Tensor& masked_m,
                                        int                  expected_m,
                                        int                  user_deep_gemm_num_sm,
                                        cudaStream_t         stream);

private:
    static inline int getNumSms(int user_deep_gemm_num_sm);
};

}  // namespace rtp_llm
